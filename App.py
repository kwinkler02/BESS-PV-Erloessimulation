import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import locale

# ── 1) Progress-Bar vorbereiten ──────────────────────────────────────────────
progress_bar  = st.sidebar.progress(0)
progress_text = st.sidebar.empty()
def set_progress(pct: int):
    progress_bar.progress(pct)
    progress_text.markdown(f"**Fortschritt:** {pct}%")

# ── 2) Deutsches Locale & Euro-Formatter ────────────────────────────────────
try:
    locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
    def fmt_euro(x):
        return locale.currency(x, symbol=True, grouping=True)
except locale.Error:
    def fmt_euro(x):
        s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s + " €"

# ── 3) Solver-Funktion mit Progress-Callback ─────────────────────────────────
def compute_results(price_file, pv_file,
                    start_soc, cap, bat_kw, grid_kw,
                    eff_pct, max_cycles,
                    progress_callback):
    # -- 3.1) Data-Loader --
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

    price_df = load_price_df(price_file)
    pv_df    = load_pv_df(pv_file)

    timestamps = price_df["Zeitstempel"]
    prices_mwh = price_df["Preis_€/MWh"].to_numpy()
    pv_feed    = pv_df["PV_kWh"].to_numpy()
    T = len(prices_mwh)
    if len(pv_feed) != T:
        st.error(f"Preisdaten ({T}) ≠ PV-Daten ({len(pv_feed)})")
        st.stop()

    # -- 3.2) Parameter & Limits --
    prices     = prices_mwh / 1000.0
    interval_h = 0.25
    batt_max   = bat_kw   * interval_h
    grid_max   = grid_kw  * interval_h
    pv_use     = np.minimum(pv_feed, grid_max)

    # -- 3.3) Solve inklusive progress_callback --
    def solve(pv_vec):
        m = pulp.LpProblem("BESS", pulp.LpMaximize)
        c  = pulp.LpVariable.dicts("c",  range(T), cat="Binary")
        d  = pulp.LpVariable.dicts("d",  range(T), cat="Binary")
        ch = pulp.LpVariable.dicts("ch", range(T), lowBound=0, upBound=batt_max)
        dh = pulp.LpVariable.dicts("dh", range(T), lowBound=0, upBound=batt_max)
        soc= pulp.LpVariable.dicts("soc",range(T), lowBound=0, upBound=cap)

        m += pulp.lpSum(prices[t]*dh[t] - prices[t]*ch[t] for t in range(T))
        eff = eff_pct**0.5

        for t in range(T):
            m += c[t] + d[t] <= 1
            m += ch[t] <= batt_max * c[t]
            m += ch[t] >= interval_h * c[t]
            m += dh[t] <= batt_max * d[t]
            m += dh[t] >= interval_h * d[t]
            m += pv_vec[t] + ch[t] + dh[t] <= grid_max
            prev = start_soc if t==0 else soc[t-1]
            m += soc[t] == prev + eff*ch[t] - dh[t]/eff

            # fein granularen Fortschritt melden
            if progress_callback and (t % max(1, T//50) == 0):
                pct = 5 + int(45 * t / T)
                progress_callback(pct)

        m += pulp.lpSum((ch[t]+dh[t])/(2*cap) for t in range(T)) <= max_cycles

        # vor dem Solve 50%
        if progress_callback: progress_callback(50)
        pulp.PULP_CBC_CMD(msg=False, timeLimit=120).solve(m)
        # nach dem Solve 90%
        if progress_callback: progress_callback(90)

        obj  = pulp.value(m.objective) or 0.0
        ch_v = np.array([ch[t].value() for t in range(T)])
        dh_v = np.array([dh[t].value() for t in range(T)])
        return obj, ch_v, dh_v

    # -- 3.4) Ausführen und End-Progress setzen --
    obj_w, ch_w, dh_w = solve(pv_use)
    obj_n, ch_n, dh_n = solve(np.zeros(T))
    if progress_callback: progress_callback(100)
    return (timestamps, prices_mwh, pv_feed,
            obj_w, ch_w, dh_w, obj_n, ch_n, dh_n, interval_h)

# ── 4) Streamlit-Seite starten ───────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("UBESS-Vermarktungserlöse")

# ── 5) ▶️ Simulation starten (ganz oben in der Sidebar) ─────────────────────
if st.sidebar.button("▶️ Simulation starten"):
    if not (st.session_state.get("price_file") and st.session_state.get("pv_file")):
        st.sidebar.error("Bitte zuerst beide Dateien hochladen.")
    else:
        st.session_state.results = compute_results(
            price_file   = st.session_state.price_file,
            pv_file      = st.session_state.pv_file,
            start_soc    = st.session_state.start_soc,
            cap          = st.session_state.cap,
            bat_kw       = st.session_state.bat_kw,
            grid_kw      = st.session_state.grid_kw,
            eff_pct      = st.session_state.eff_pct,
            max_cycles   = st.session_state.max_cycles,
            progress_callback = set_progress
        )

# ── 6) Datei-Uploads & Eingaben ───────────────────────────────────────────────
st.sidebar.markdown("### Datei-Uploads")
st.session_state.price_file = st.sidebar.file_uploader(
    "Strommarkt-Preise (Zeit, Preis €/MWh)", type=["csv","xls","xlsx"]
)
st.session_state.pv_file    = st.sidebar.file_uploader(
    "PV-Lastgang (Zeit, kWh)",           type=["csv","xls","xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Eingaben")
st.session_state.start_soc  = st.sidebar.number_input("Start-SoC (kWh)", 0.0, 1e6, 0.0, step=1.0)
st.session_state.cap        = st.sidebar.number_input("Kapazität (kWh)", 0.0, 1e6, 4472.0, step=1.0)
st.session_state.bat_kw     = st.sidebar.number_input("Batterie (kW)",   0.0, 1e6, 559.0, step=1.0)
st.session_state.grid_kw    = st.sidebar.number_input("Netzanschluss (kW)",0.0,1e6,757.5,step=1.0)
st.session_state.eff_pct    = st.sidebar.number_input(
    "Round-Trip Eff. (%)", 0.0,100.0,91.0,step=0.1,format="%.1f"
) / 100.0
st.session_state.max_cycles = st.sidebar.number_input("Zyklen/Jahr",0.0,1e4,548.0,step=1.0)

# ── 7) Warten bis Simulation läuft ───────────────────────────────────────────
if "results" not in st.session_state:
    st.info("Bitte auf **Simulation starten** klicken.")
    st.stop()

# ── 8) Ergebnisse entpacken ──────────────────────────────────────────────────
(
    timestamps, prices_mwh, pv_feed,
    obj_w, ch_w, dh_w,
    obj_n, ch_n, dh_n,
    interval_h
) = st.session_state.results

# ── 9) Kennzahlen ───────────────────────────────────────────────────────────
loss_abs = obj_n - obj_w
loss_pct = abs(loss_abs)/obj_n*100 if obj_n else 0.0
c1, c2, c3 = st.columns(3)
c1.metric("Gewinn ohne PV", fmt_euro(obj_n))
c2.metric("Gewinn mit PV",   fmt_euro(obj_w))
c3.metric(
    "Verlust durch PV",
    "-" + fmt_euro(abs(loss_abs)),
    f"{-loss_pct:.2f} %",
    delta_color="normal"
)

# ── 10) Chart 1: Erlöse monatsweise ──────────────────────────────────────────
rev_w = prices_mwh/1000 * dh_w - prices_mwh/1000 * ch_w
rev_n = prices_mwh/1000 * dh_n - prices_mwh/1000 * ch_n
dfm = (
    pd.DataFrame({
        "Datum":   timestamps.dt.floor("D"),
        "ohne PV": rev_n,
        "mit PV":  rev_w
    })
    .groupby(lambda i: timestamps.dt.floor("D")[i].to_period("M"))
    [["ohne PV","mit PV"]]
    .sum()
)
dfm.index = dfm.index.to_timestamp()

st.subheader("Erlöse (monatsweise)")
fig1, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(dfm.index, dfm["ohne PV"], label="ohne PV")
ax1.plot(dfm.index, dfm["mit PV"],  label="mit PV")

ax1.yaxis.set_major_locator(mticker.MultipleLocator(1_000))
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,d}".replace(",",".")+" €")
)
ax1.grid(axis="y", linestyle="--", alpha=0.3)

ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

fig1.autofmt_xdate()
ax1.legend(loc="upper left")
st.pyplot(fig1, use_container_width=True)

# ── 11) Chart 2: Kumulierte Erlöse ──────────────────────────────────────────
cum = (
    pd.DataFrame({
        "Datum":   timestamps.dt.floor("D"),
        "ohne PV": rev_n,
        "mit PV":  rev_w
    })
    .groupby("Datum")[["ohne PV","mit PV"]]
    .sum()
    .sort_index()
    .cumsum()
)

st.subheader("Kumulierte Erlöse")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(cum.index, cum["ohne PV"], label="ohne PV")
ax2.plot(cum.index, cum["mit PV"],  label="mit PV")

ax2.yaxis.set_major_locator(mticker.MultipleLocator(10_000))
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,d}".replace(",",".")+" €")
)
ax2.grid(axis="y", linestyle="--", alpha=0.3)

ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

fig2.autofmt_xdate()
ax2.legend(loc="upper left")
st.pyplot(fig2, use_container_width=True)

# ── 12) Ergebnis-Tabelle & Download ─────────────────────────────────────────
loss_factor = 1 - st.session_state.eff_pct**0.5
cycles_w    = (ch_w + dh_w)/(2*st.session_state.cap)
out = pd.DataFrame({
    "Zeitstempel":         timestamps,
    "Preis (€/MWh)":       prices_mwh,
    "PV-Einspeisung (kWh)":pv_feed,
    "PV-genutzt (kWh)":    np.minimum(pv_feed, st.session_state.grid_kw*interval_h),
    "Ladeaktiv":           (ch_w>0).astype(int),
    "Entladeaktiv":        (dh_w>0).astype(int),
    "Lade-kWh":            ch_w,
    "Entlade-kWh":         dh_w,
    "Verlust (kWh)":       -(ch_w+dh_w)*loss_factor,
    "Kum. Zyklen":         np.cumsum(cycles_w),
    "Netzlast (kWh)":      np.minimum(pv_feed, st.session_state.grid_kw*interval_h)+ch_w+dh_w
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
