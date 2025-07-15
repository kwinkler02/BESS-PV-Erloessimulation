import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO

# ── 0) Seite konfigurieren ──────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("BESS-Vermarktungserlöse")

# ── 1) Session-State initialisieren ─────────────────────────────────────────
if "run" not in st.session_state:
    st.session_state.run = False

# ── 2) Sidebar: Simulation & Reset ──────────────────────────────────────────
st.sidebar.header("Aktionen")
if st.sidebar.button("▶️ Simulation starten"):
    st.session_state.run = True
if st.sidebar.button("🔄 Neue Eingabe"):
    for key in ["run", "price_file", "pv_file"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()

# ── 3) Sidebar: Input-Dateien & Systemparameter ─────────────────────────────
st.sidebar.markdown("---")
price_file = st.sidebar.file_uploader(
    "Strommarktpreise hochladen\n(1. Spalte Zeitstempel, 2. Spalte Preis €/MWh)",
    type=["csv", "xls", "xlsx"],
    key="price_file"
)
pv_file = st.sidebar.file_uploader(
    "PV-Lastgang hochladen\n(1. Spalte Zeitstempel, 2. Spalte Einspeisung kWh)",
    type=["csv", "xls", "xlsx"],
    key="pv_file"
)

st.sidebar.markdown("---")
start_soc  = st.sidebar.number_input("Start-SoC (kWh)",       0.0, 1e6, 0.0, step=1.0)
cap        = st.sidebar.number_input("Kapazität (kWh)",       0.0, 1e6, 4472.0, step=1.0)
bat_kw     = st.sidebar.number_input("Batterieleistung (kW)", 0.0, 1e6, 559.0, step=1.0)
grid_kw    = st.sidebar.number_input("Netzanschlussleistung (kW)", 0.0, 1e6, 757.5, step=1.0)
eff_pct    = st.sidebar.slider("Round-Trip Efficiency (%)", 0, 100, 91) / 100.0
max_cycles = st.sidebar.number_input("Zyklen/Jahr", 0.0, 1e4, 548.0, step=1.0)

# ── 4) Abbruch, wenn nicht gestartet oder Dateien fehlen ────────────────────
if not st.session_state.run:
    st.info("Bitte auf **Simulation starten** klicken.")
    st.stop()
if price_file is None or pv_file is None:
    st.error("Beide Dateien (Preise & PV) müssen hochgeladen sein.")
    st.stop()

# ── 5) DataLoader-Funktionen ────────────────────────────────────────────────
@st.cache_data
def load_price_df(upl):
    if upl.name.lower().endswith(".csv"):
        df = pd.read_csv(upl,
                         usecols=[0, 1],
                         names=["Zeitstempel", "Preis_€/MWh"],
                         header=0, sep=";", decimal=",")
    else:
        df = pd.read_excel(upl,
                           usecols=[0, 1],
                           names=["Zeitstempel", "Preis_€/MWh"],
                           header=0, engine="openpyxl")
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], dayfirst=True)
    df["Preis_€/MWh"] = pd.to_numeric(df["Preis_€/MWh"], errors="raise")
    return df

@st.cache_data
def load_pv_df(upl):
    if upl.name.lower().endswith(".csv"):
        df = pd.read_csv(upl,
                         usecols=[0, 1],
                         names=["Zeitstempel", "PV_kWh"],
                         header=0, sep=";", decimal=",")
    else:
        df = pd.read_excel(upl,
                           usecols=[0, 1],
                           names=["Zeitstempel", "PV_kWh"],
                           header=0, engine="openpyxl")
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], dayfirst=True)
    df["PV_kWh"] = pd.to_numeric(df["PV_kWh"], errors="raise")
    return df

# ── 6) DataFrames einlesen ───────────────────────────────────────────────────
price_df = load_price_df(price_file)
pv_df    = load_pv_df(pv_file)

timestamps = price_df["Zeitstempel"]
prices_mwh = price_df["Preis_€/MWh"].to_numpy()
pv_feed    = pv_df["PV_kWh"].to_numpy()

T = len(prices_mwh)
if len(pv_feed) != T:
    st.error(f"Zeilenanzahl Preise ({T}) ≠ PV-Lastgang ({len(pv_feed)})")
    st.stop()

# ── 7) Einheiten & Limits ───────────────────────────────────────────────────
prices       = prices_mwh / 1000.0   # €/kWh
interval_h   = 0.25                  # 15 min
batt_max_kwh = bat_kw * interval_h
grid_max_kwh = grid_kw * interval_h
pv_use       = np.minimum(pv_feed, grid_max_kwh)

# ── 8) Fortschritt im Hauptbereich ──────────────────────────────────────────
progress_bar  = st.empty()
progress_text = st.empty()
def set_progress(pct: int):
    progress_bar.progress(pct)
    progress_text.markdown(f"**Fortschritt:** {pct}%")
set_progress(0)

# ── 9) Solver-Funktion ──────────────────────────────────────────────────────
def solve_arbitrage(pv_vec, pct_target):
    model = pulp.LpProblem("BESS_Arbitrage", pulp.LpMaximize)
    c_on   = pulp.LpVariable.dicts("c_on", range(T), cat="Binary")
    d_on   = pulp.LpVariable.dicts("d_on", range(T), cat="Binary")
    charge = pulp.LpVariable.dicts("charge", range(T), lowBound=0, upBound=batt_max_kwh)
    discharge = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=batt_max_kwh)
    SoC    = pulp.LpVariable.dicts("SoC", range(T), lowBound=0, upBound=cap)

    # Zielfunktion
    model += pulp.lpSum(prices[t] * discharge[t] - prices[t] * charge[t] for t in range(T))
    eff = eff_pct ** 0.5

    # Constraints
    for t in range(T):
        model += c_on[t] + d_on[t] <= 1
        model += charge[t] <= batt_max_kwh * c_on[t]
        model += charge[t] >= interval_h * c_on[t]
        model += discharge[t] <= batt_max_kwh * d_on[t]
        model += discharge[t] >= interval_h * d_on[t]
        model += pv_vec[t] + charge[t] + discharge[t] <= grid_max_kwh
        prev = start_soc if t == 0 else SoC[t-1]
        model += SoC[t] == prev + eff * charge[t] - discharge[t] / eff

    model += pulp.lpSum((charge[t] + discharge[t]) / (2 * cap) for t in range(T)) <= max_cycles

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    model.solve(solver)
    set_progress(pct_target)

    obj = pulp.value(model.objective) or 0.0
    ch  = np.array([charge[t].value() for t in range(T)])
    dh  = np.array([discharge[t].value() for t in range(T)])
    soc = np.array([SoC[t].value() for t in range(T)])
    return obj, ch, dh, soc

# ── 10) Mit & ohne PV optimieren ─────────────────────────────────────────────
with st.spinner("Berechne mit PV…"):
    obj_w, ch_w, dh_w, soc_w = solve_arbitrage(pv_use, pct_target=50)
with st.spinner("Berechne ohne PV…"):
    obj_n, ch_n, dh_n, soc_n = solve_arbitrage(np.zeros(T), pct_target=90)
set_progress(100)

# ── 11) Kennzahlen ──────────────────────────────────────────────────────────
loss_abs = obj_n - obj_w
loss_pct = abs(loss_abs) / obj_n * 100 if obj_n != 0 else 0.0
c1, c2, c3 = st.columns(3)
c1.metric("Gewinn ohne PV", f"€{obj_n:,.2f}")
c2.metric("Gewinn mit PV",   f"€{obj_w:,.2f}")
c3.metric(
    "Verlust durch PV",
    f"-€{abs(loss_abs):,.2f}",
    f"{-loss_pct:.2f}%"
)

# ── 12) Ergebnis-Tabelle & Download ─────────────────────────────────────────
loss_factor = 1 - eff_pct ** 0.5
cycles_w    = (ch_w + dh_w) / (2 * cap)

out = pd.DataFrame({
    "Zeitstempel":               timestamps,
    "Preis (€/MWh)":             prices_mwh,
    "PV-Einspeisung (kWh)":      pv_feed,
    "PV-genutzt (kWh)":          pv_use,
    "Ladebetrieb":               (ch_w > 0).astype(int),
    "Entladebetrieb":            (dh_w > 0).astype(int),
    "Ladeenergie (kWh)":         ch_w,
    "Entladeenergie (kWh)":      dh_w,
    "SoC (kWh)":                 soc_w,
    "Verluste (kWh)":            -(ch_w + dh_w) * loss_factor,
    "Äq-Zyklen Int.":            cycles_w,
    "Kum. Äq-Zyklen":            np.cumsum(cycles_w),
    "Netzanschluss Last (kWh)":  pv_use + ch_w + dh_w,
    "Netzlast (kW)":             (pv_use + ch_w + dh_w) / interval_h,
})

st.dataframe(out, height=400, use_container_width=True)

buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name="Ergebnis")
buf.seek(0)
st.download_button(
    "📥 Ergebnis als Excel herunterladen",
    data=buf,
    file_name="Optimierungsergebnis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
