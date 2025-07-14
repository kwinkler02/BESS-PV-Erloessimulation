import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO

# 0) Seite konfigurieren
st.set_page_config(layout="wide")
st.title("BESS-Vermarktungserlöse")

# --------------------------------------------------
# Session State für Run-Flag und Reset
if "run" not in st.session_state:
    st.session_state.run = False

# 1) Buttons oben in der Sidebar
st.sidebar.header("Aktionen")
if st.sidebar.button("🚀 Simulation starten"):
    st.session_state.run = True
if st.sidebar.button("🔄 Neue Eingabe"):
    for k in ["run", "price", "pv"]:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# 2) Input-Dateien
st.sidebar.header("📂 Input-Dateien")
price_file = st.sidebar.file_uploader(
    "🔸 Day-Ahead-Preise (Datum, Preis €/MWh)",
    type=["xlsx", "xls", "csv"], key="price"
)
pv_file = st.sidebar.file_uploader(
    "🔸 PV-Lastgang (Datum, Einspeisung kWh)",
    type=["xlsx", "xls", "csv"], key="pv"
)

# 3) Systemparameter
st.sidebar.header("⚙️ Systemparameter")
start_soc   = st.sidebar.number_input("Start-SoC (kWh)", min_value=0.0, value=0.0, step=1.0)
cap         = st.sidebar.number_input("Kapazität (kWh)", min_value=0.0, value=4472.0, step=1.0)
bat_power   = st.sidebar.number_input("Max-Leistung Batterie (kW)", min_value=0.0, value=559.0, step=1.0)
grid_power  = st.sidebar.number_input("Max-Leistung Netzanschluss (kW)", min_value=0.0, value=757.5, step=1.0)
eff_pct     = st.sidebar.number_input("Round-trip efficiency (%)", min_value=0.0, max_value=100.0, value=91.3, step=0.1)/100.0
max_cycles  = st.sidebar.number_input("Max Äq-Zyklen/Jahr", min_value=0.0, value=548.0, step=1.0)

# 4) Abbruch, wenn noch nicht gestartet
if not st.session_state.run:
    st.info("Bitte auf **Simulation starten** klicken, um die Optimierung auszuführen.")
    st.stop()

# 5) Fortschrittsbalken initialisieren
progress = st.sidebar.progress(0)

# 6) Dateien einlesen
def load_df(upl):
    if upl.name.lower().endswith(".csv"):
        return pd.read_csv(upl, header=0)
    else:
        return pd.read_excel(upl, header=0, engine="openpyxl")

if price_file is None or pv_file is None:
    st.error("Bitte beide Dateien hochladen.")
    st.stop()

progress.progress(10)
df_price = load_df(price_file)
df_pv    = load_df(pv_file)

# 7) Zeitstempel, Preise und PV-Feed extrahieren
try:
    timestamps = pd.to_datetime(df_price.iloc[:,0], dayfirst=True)
    prices_mwh = pd.to_numeric(df_price.iloc[:,1], errors="raise").values
    pv_feed    = pd.to_numeric(df_pv   .iloc[:,1], errors="raise").values
except Exception:
    st.error("Fehler beim Einlesen der Dateien – bitte Format prüfen.")
    st.stop()

T = len(prices_mwh)
if len(pv_feed) != T:
    st.error(f"Reihenlängen stimmen nicht: Preise={T}, PV={len(pv_feed)}")
    st.stop()

prices = prices_mwh / 1000.0  # €/MWh → €/kWh

# 8) PV-Nutzung vorab berechnen
interval_h   = 0.25
grid_max_kwh = grid_power * interval_h
batt_max_kwh = bat_power  * interval_h
pv_use       = np.minimum(pv_feed, grid_max_kwh)

# 9) Optimierungsfunktion
def solve_arbitrage(pv_vec, prog_offset):
    model = pulp.LpProblem("BESS_Arbitrage", pulp.LpMaximize)
    c_on      = pulp.LpVariable.dicts("charge_on",    range(T), cat="Binary")
    d_on      = pulp.LpVariable.dicts("discharge_on", range(T), cat="Binary")
    charge    = pulp.LpVariable.dicts("charge",       range(T),
                                      lowBound=0, upBound=batt_max_kwh)
    discharge = pulp.LpVariable.dicts("discharge",    range(T),
                                      lowBound=0, upBound=batt_max_kwh)
    SoC       = pulp.LpVariable.dicts("SoC",          range(T),
                                      lowBound=0, upBound=cap)
    # Zielfunktion
    model += pulp.lpSum(prices[t]*discharge[t] - prices[t]*charge[t]
                        for t in range(T)), "Maximiere_Gewinn"
    eff = eff_pct**0.5
    for t in range(T):
        model += c_on[t] + d_on[t] <= 1
        model += charge[t]    <= batt_max_kwh * c_on[t]
        model += charge[t]    >= interval_h   * c_on[t]
        model += discharge[t] <= batt_max_kwh * d_on[t]
        model += discharge[t] >= interval_h   * d_on[t]
        model += pv_vec[t] + charge[t] + discharge[t] <= grid_max_kwh
        prev = start_soc if t==0 else SoC[t-1]
        model += SoC[t] == prev + eff*charge[t] - discharge[t]/eff
    model += pulp.lpSum((charge[t]+discharge[t])/(2*cap)
                        for t in range(T)) <= max_cycles
    # hier keine per-Step Updates, sondern nur grobe Indikatoren
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    model.solve(solver)
    # nach jedem solve grob den Fortschritt setzen
    progress.progress(prog_offset)
    obj = pulp.value(model.objective) or 0.0
    ch  = np.array([charge[t].value()    for t in range(T)])
    dh  = np.array([discharge[t].value() for t in range(T)])
    soc = np.array([SoC[t].value()       for t in range(T)])
    return obj, ch, dh, soc

# 10) Jetzt rechnen mit spinners und progress
with st.spinner("⏳ Berechne mit PV…"):
    obj_w, ch_w, dh_w, soc_w = solve_arbitrage(pv_use, prog_offset=50)
with st.spinner("⏳ Berechne ohne PV…"):
    obj_n, ch_n, dh_n, soc_n = solve_arbitrage(np.zeros(T), prog_offset=80)

progress.progress(100)

# 11) Kennzahlen anzeigen
loss_abs = obj_n - obj_w
loss_rel = loss_abs / abs(obj_n) if obj_n != 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("💰 Gewinn ohne PV",   f"€{obj_n:,.2f}")
c2.metric("💰 Gewinn mit PV",     f"€{obj_w:,.2f}")
c3.metric("📉 Verlust durch PV",  f"-€{loss_abs:,.2f}", f"{-loss_rel*100:.2f}%")

# 12) Ergebnis-Tabelle & Download
loss_factor = 1 - eff_pct**0.5
cycles_w    = (ch_w + dh_w)/(2*cap)

out = pd.DataFrame({
    "Zeitstempel":               timestamps,
    "Preis (€/MWh)":             prices_mwh,
    "PV-Einspeisung (kWh)":      pv_feed,
    "PV-genutzt (kWh)":          pv_use,
    "Ladebetrieb":               (ch_w>0).astype(int),
    "Entladebetrieb":            (dh_w>0).astype(int),
    "Ladeenergie (kWh)":         ch_w,
    "Entladeenergie (kWh)":      dh_w,
    "SoC (kWh)":                 soc_w,
    "Verluste (kWh)":            -(ch_w+dh_w)*(1-eff_pct**0.5),
    "Äq-Zyklen Int.":            cycles_w,
    "Kum. Äq-Zyklen":            np.cumsum(cycles_w),
    "Netzbelastung (kWh)":       pv_use + ch_w + dh_w,
    "Netzlast (kW)":             (pv_use + ch_w + dh_w) * 4,
})

st.dataframe(out, height=450, use_container_width=True)

buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name="Ergebnis")
buf.seek(0)

st.download_button(
    "📥 Download Ergebnis-Excel",
    data=buf,
    file_name="Optimierungsergebnis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

