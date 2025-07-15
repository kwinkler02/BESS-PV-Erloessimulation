import streamlit as st 
import pandas as pd
import numpy as np
import pulp
from io import BytesIO
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import locale

# deutsches Locale für Monatsnamen & Zahlen
try:
    locale.setlocale(locale.LC_TIME,    "de_DE.UTF-8")
    locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
except locale.Error:
    pass

# Euro-Formatter
try:
    locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
    def fmt_euro(x):
        return locale.currency(x, symbol=True, grouping=True)
except locale.Error:
    def fmt_euro(x):
        s = f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s + " €"

# ── 0) Seite konfigurieren ──────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("UBESS-Vermarktungserlöse")

# ── 1) Session-State initialisieren ─────────────────────────────────────────
if "run" not in st.session_state:
    st.session_state.run = False

# ── 2) Sidebar: Simulation & Reset ──────────────────────────────────────────
st.sidebar.markdown("### Aktionen")
if st.sidebar.button("▶️ Simulation starten"):
    st.session_state.run = True
if st.sidebar.button("🔄 Neue Eingabe"):
    for key in ["run", "price_file", "pv_file"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()

# ── 3) Sidebar: Datei-Uploads & Eingaben ────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Datei-Uploads")
price_file = st.sidebar.file_uploader(
    "Strommarktpreise hochladen\n(1. Spalte Zeitstempel, 2. Spalte Preis €/MWh)",
    type=["csv", "xls", "xlsx"], key="price_file"
)
pv_file = st.sidebar.file_uploader(
    "PV-Lastgang hochladen\n(1. Spalte Zeitstempel, 2. Spalte Einspeisung kWh)",
    type=["csv", "xls", "xlsx"], key="pv_file"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Eingaben")
start_soc  = st.sidebar.number_input("Start-SoC (kWh)",       0.0, 1e6, 0.0, step=1.0)
cap        = st.sidebar.number_input("Kapazität (kWh)",       0.0, 1e6, 4472.0, step=1.0)
bat_kw     = st.sidebar.number_input("Batterieleistung (kW)", 0.0, 1e6, 559.0, step=1.0)
grid_kw    = st.sidebar.number_input("Netzanschlussleistung (kW)", 0.0, 1e6, 757.5, step=1.0)
eff_pct    = st.sidebar.number_input(
    "Round-Trip Efficiency (%)", 0.0, 100.0, 91.0, step=0.1, format="%.1f"
) / 100.0
max_cycles = st.sidebar.number_input("Zyklen/Jahr", 0.0, 1e4, 548.0, step=1.0)

# ── 4) Abbruch, wenn nicht gestartet oder Dateien fehlen ────────────────────
if not st.session_state.run:
    st.info("Bitte auf **Simulation starten** klicken.")
    st.stop()
if price_file is None or pv_file is None:
    st.error("Beide Dateien (Preise & PV) müssen hochgeladen sein.")
    st.stop()

# ── 5) Data-Loader ───────────────────────────────────────────────────────────
@st.cache_data
def load_price_df(upl):
    if upl.name.lower().endswith(".csv"):
        df = pd.read_csv(upl, usecols=[0,1], names=["Zeitstempel","Preis_€/MWh"],
                         header=0, sep=";", decimal=",")
    else:
        df = pd.read_excel(upl, usecols=[0,1], names=["Zeitstempel","Preis_€/MWh"],
                           header=0, engine="openpyxl")
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], dayfirst=True)
    df["Preis_€/MWh"] = pd.to_numeric(df["Preis_€/MWh"], errors="raise")
    return df

@st.cache_data
def load_pv_df(upl):
    if upl.name.lower().endswith(".csv"):
        df = pd.read_csv(upl, usecols=[0,1], names=["Zeitstempel","PV_kWh"],
                         header=0, sep=";", decimal=",")
    else:
        df = pd.read_excel(upl, usecols=[0,1], names=["Zeitstempel","PV_kWh"],
                           header=0, engine="openpyxl")
    df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], dayfirst=True)
    df["PV_kWh"]      = pd.to_numeric(df["PV_kWh"], errors="raise")
    return df

# ── 6) Compute-Funktion ──────────────────────────────────────────────────────
def compute_results(price_file, pv_file,
                    start_soc, cap, bat_kw, grid_kw,
                    eff_pct, max_cycles):
    price_df   = load_price_df(price_file)
    pv_df      = load_pv_df(pv_file)
    timestamps = price_df["Zeitstempel"]
    prices_mwh = price_df["Preis_€/MWh"].to_numpy()
    pv_feed    = pv_df["PV_kWh"].to_numpy()
    T = len(prices_mwh)
    if len(pv_feed) != T:
        st.error(f"Preise ({T}) ≠ PV-Daten ({len(pv_feed)})")
        st.stop()

    prices       = prices_mwh / 1000.0
    interval_h   = 0.25
    batt_max_kwh = bat_kw * interval_h
    grid_max_kwh = grid_kw * interval_h
    pv_use       = np.minimum(pv_feed, grid_max_kwh)

    def solve(pv_vec):
        model = pulp.LpProblem("BESS", pulp.LpMaximize)
        c  = pulp.LpVariable.dicts("c", range(T), cat="Binary")
        d  = pulp.LpVariable.dicts("d", range(T), cat="Binary")
        ch = pulp.LpVariable.dicts("ch", range(T), lowBound=0, upBound=batt_max_kwh)
        dh = pulp.LpVariable.dicts("dh", range(T), lowBound=0, upBound=batt_max_kwh)
        SoC= pulp.LpVariable.dicts("SoC",range(T), lowBound=0, upBound=cap)

        model += pulp.lpSum(prices[t]*dh[t] - prices[t]*ch[t] for t in range(T))
        eff = eff_pct**0.5
        for t in range(T):
            model += c[t] + d[t] <= 1
            model += ch[t] <= batt_max_kwh * c[t]
            model += ch[t] >= interval_h * c[t]
            model += dh[t] <= batt_max_kwh * d[t]
            model += dh[t] >= interval_h * d[t]
            model += pv_vec[t] + ch[t] + dh[t] <= grid_max_kwh
            prev = start_soc if t==0 else SoC[t-1]
            model += SoC[t] == prev + eff*ch[t] - dh[t]/eff
        model += pulp.lpSum((ch[t]+dh[t])/(2*cap) for t in range(T)) <= max_cycles

        pulp.PULP_CBC_CMD(msg=False, timeLimit=120).solve(model)
        obj  = pulp.value(model.objective) or 0.0
        ch_v = np.array([ch[t].value() for t in range(T)])
        dh_v = np.array([dh[t].value() for t in range(T)])
        return obj, ch_v, dh_v

    obj_w, ch_w, dh_w = solve(pv_use)
    obj_n, ch_n, dh_n = solve(np.zeros(T))
    return timestamps, prices_mwh, pv_feed, obj_w, ch_w, dh_w, obj_n, ch_n, dh_n, interval_h

# ── 7) Progress-Bar anlegen ─────────────────────────────────────────────────
progress_bar  = st.progress(0)
progress_text = st.empty()
def set_progress(pct: int):
    progress_bar.progress(pct)
    progress_text.markdown(f"**Fortschritt:** {pct}%")

# ── 8) Ergebnisse berechnen & Fortschritt updaten ────────────────────────────
set_progress(5)
(timestamps, prices_mwh, pv_feed,
 obj_w, ch_w, dh_w,
 obj_n, ch_n, dh_n,
 interval_h) = compute_results(
    price_file, pv_file,
    start_soc, cap, bat_kw, grid_kw,
    eff_pct, max_cycles
)
set_progress(100)

# ── 9) Kennzahlen ───────────────────────────────────────────────────────────
loss_abs = obj_n - obj_w
loss_pct = abs(loss_abs)/obj_n*100 if obj_n!=0 else 0.0
c1, c2, c3 = st.columns(3)
c1.metric("Gewinn ohne PV", fmt_euro(obj_n))
c2.metric("Gewinn mit PV",   fmt_euro(obj_w))
c3.metric(
    "Verlust durch PV",
    "-" + fmt_euro(abs(loss_abs)),
    f"{-loss_pct:,.2f} %",
    delta_color="inverse"
)

# ── 10) Nicht-kumuliert „Erlöse“ pro Tag ─────────────────────────────────────
rev_w = prices_mwh/1000 * dh_w - prices_mwh/1000 * ch_w
rev_n = prices_mwh/1000 * dh_n - prices_mwh/1000 * ch_n
df_daily = pd.DataFrame({
    "Datum":   timestamps.dt.floor("D"),
    "ohne PV": rev_n,
    "mit PV":  rev_w
})
daily_sum = df_daily.groupby("Datum").sum().sort_index()

st.subheader("Erlöse")
fig1, ax1 = plt.subplots(figsize=(8,3))
ax1.plot(daily_sum.index, daily_sum["ohne PV"], label="ohne PV")
ax1.plot(daily_sum.index, daily_sum["mit PV"],  label="mit PV")
ax1.yaxis.set_major_locator(mticker.MultipleLocator(10_000))
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"{int(x):,d}".replace(",",".")+" €")
)
ax1.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig1.autofmt_xdate()
ax1.legend()
st.pyplot(fig1)

# ── 11) Kumuliert „Kumulierte Erlöse“ ───────────────────────────────────────
st.subheader("Kumulierte Erlöse")
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(daily_sum.index, daily_sum["ohne PV"].cumsum(), label="ohne PV")
ax2.plot(daily_sum.index, daily_sum["mit PV"].cumsum(),  label="mit PV")
ax2.yaxis.set_major_locator(mticker.MultipleLocator(10_000))
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"{int(x):,d}".replace(",",".")+" €")
)
ax2.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig2.autofmt_xdate()
ax2.legend()
st.pyplot(fig2)

# ── 12) Tabelle & Download ───────────────────────────────────────────────────
loss_factor = 1 - eff_pct**0.5
cycles_w    = (ch_w + dh_w)/(2*cap)
out = pd.DataFrame({
    "Zeitstempel":         timestamps,
    "Preis (€/MWh)":       prices_mwh,
    "PV-Einspeisung (kWh)":pv_feed,
    "PV-genutzt (kWh)":    np.minimum(pv_feed, grid_kw*interval_h),
    "Ladeaktiv":           (ch_w>0).astype(int),
    "Entladeaktiv":        (dh_w>0).astype(int),
    "Lade-kWh":            ch_w,
    "Entlade-kWh":         dh_w,
    "Verlust (kWh)":       -(ch_w+dh_w)*loss_factor,
    "Kum. Zyklen":         np.cumsum(cycles_w),
    "Netzlast (kWh)":      np.minimum(pv_feed, grid_kw*interval_h)+ch_w+dh_w
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
    mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
)
