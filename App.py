import streamlit as st
import pandas as pd
import numpy as np
import pulp
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import locale

# ── 1) Page & Progress UI ───────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("UBESS-Vermarktungserlöse")

progress_bar  = st.sidebar.progress(0)
progress_text = st.sidebar.empty()

def set_progress(pct: int):
    progress_bar.progress(pct)
    progress_text.markdown(f"**Fortschritt:** {pct}%")

# ── 2) Deutsches Locale & Euro-Formatter ────────────────────────────────────
try:
    locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
    def fmt_euro(x: float) -> str:
        return locale.currency(x, symbol=True, grouping=True)
except locale.Error:
    def fmt_euro(x: float) -> str:
        s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s + " €"

# ── 3) Solver-Funktion mit Progress-Callback ────────────────────────────────
def compute_results(
    price_file,
    pv_file,
    start_soc: float,
    cap: float,
    bat_kw: float,
    grid_kw: float,
    eff_pct: float,
    max_cycles: float,
    progress_callback=None,
    baseline_no_grid: bool = False,
    end_soc_same_as_start: bool = False,
):
    """Löst zwei Fälle: mit PV (Netzlimit aktiv) und ohne PV (optional ohne Netzlimit).
    Gibt für beide Pfade u.a. Lade-/Entladeströme und SoC-Zeitreihen zurück.
    """

    # -- 3.1) Data-Loader --
    def load_price_df(upl):
        if upl.name.lower().endswith(".csv"):
            df = pd.read_csv(
                upl,
                usecols=[0, 1],
                names=["Zeitstempel", "Preis_€/MWh"],
                header=0,
                sep=";",
                decimal=",",
            )
        else:
            df = pd.read_excel(
                upl,
                usecols=[0, 1],
                names=["Zeitstempel", "Preis_€/MWh"],
                header=0,
                engine="openpyxl",
            )
        df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], dayfirst=True)
        df["Preis_€/MWh"] = pd.to_numeric(df["Preis_€/MWh"], errors="raise")
        return df

    def load_pv_df(upl):
        if upl.name.lower().endswith(".csv"):
            df = pd.read_csv(
                upl,
                usecols=[0, 1],
                names=["Zeitstempel", "PV_kWh"],
                header=0,
                sep=";",
                decimal=",",
            )
        else:
            df = pd.read_excel(
                upl,
                usecols=[0, 1],
                names=["Zeitstempel", "PV_kWh"],
                header=0,
                engine="openpyxl",
            )
        df["Zeitstempel"] = pd.to_datetime(df["Zeitstempel"], dayfirst=True)
        df["PV_kWh"] = pd.to_numeric(df["PV_kWh"], errors="raise")
        return df

    price_df = load_price_df(price_file)
    pv_df = load_pv_df(pv_file)

    timestamps = price_df["Zeitstempel"]
    prices_mwh = price_df["Preis_€/MWh"].to_numpy()
    pv_feed = pv_df["PV_kWh"].to_numpy()

    T = len(prices_mwh)
    if len(pv_feed) != T:
        st.error(f"Preisdaten ({T}) ≠ PV-Daten ({len(pv_feed)})")
        st.stop()

    # -- 3.2) Parameter & Limits --
    prices = prices_mwh / 1000.0  # €/kWh
    interval_h = 0.25             # 15-Minuten-Auflösung
    batt_max = bat_kw * interval_h
    grid_max = grid_kw * interval_h
    pv_use = np.minimum(pv_feed, grid_max)

    # -- 3.3) Optimierungsproblem (mit Fortschritt) --
    def solve(pv_vec, enforce_grid=True):
        m = pulp.LpProblem("BESS", pulp.LpMaximize)

        c = pulp.LpVariable.dicts("c", range(T), cat="Binary")
        d = pulp.LpVariable.dicts("d", range(T), cat="Binary")
        ch = pulp.LpVariable.dicts("ch", range(T), lowBound=0, upBound=batt_max)
        dh = pulp.LpVariable.dicts("dh", range(T), lowBound=0, upBound=batt_max)
        soc = pulp.LpVariable.dicts("soc", range(T), lowBound=0, upBound=cap)
        pv2b = pulp.LpVariable.dicts("pv2b", range(T), lowBound=0, upBound=batt_max)  # PV → Batterie

        # Zielfunktion: Erlöse maximieren + winziger Bonus auf PV→BESS für eindeutige Zuordnung
        tiny = 1e-7
        m += pulp.lpSum(prices[t] * dh[t] - prices[t] * ch[t] for t in range(T)) \
             + tiny * pulp.lpSum(pv2b[t] for t in range(T))

        eff = eff_pct ** 0.5  # hin/rück jeweils sqrt(RTE)

        for t in range(T):
            # Keine gleichzeitige Lade-/Entlade-Aktivität
            m += c[t] + d[t] <= 1

            # Power-Limits + sehr kleine Mindestaktivität (vermeidet infinitesimale Flüsse)
            eps = 1e-6
            m += ch[t] <= batt_max * c[t]
            m += ch[t] >= eps * c[t]
            m += dh[t] <= batt_max * d[t]
            m += dh[t] >= eps * d[t]

            # PV→BESS darf nicht mehr sein als PV oder Laden
            m += pv2b[t] <= pv_vec[t]
            m += pv2b[t] <= ch[t]

            # AC-Bus Limit (Import+Export+PV als Absolutflüsse begrenzt)
            if enforce_grid:
                m += pv_vec[t] + ch[t] + dh[t] <= grid_max

            # SoC-Dynamik
            prev = start_soc if t == 0 else soc[t - 1]
            m += soc[t] == prev + eff * ch[t] - dh[t] / eff

            # Fortschritt: während Constraint-Builds
            if progress_callback and (t % max(1, T // 50) == 0):
                progress_callback(5 + int(45 * t / T))

        # Optional: Terminal-SoC
        if end_soc_same_as_start:
            m += soc[T - 1] == start_soc

        # Zyklenbudget auf Zeitraum skaliert (Basis: Jahresbudget)
        intervals_per_year = int(round(365 * 24 / interval_h))  # ~35.040 bei 15 min
        period_fraction = T / intervals_per_year
        m += pulp.lpSum((ch[t] + dh[t]) / (2 * cap) for t in range(T)) <= max_cycles * period_fraction

        # Solve
        if progress_callback:
            progress_callback(50)
        pulp.PULP_CBC_CMD(msg=False, timeLimit=120).solve(m)
        if progress_callback:
            progress_callback(90)

        obj = pulp.value(m.objective) or 0.0
        ch_v = np.array([ch[t].value() for t in range(T)])
        dh_v = np.array([dh[t].value() for t in range(T)])
        soc_v = np.array([soc[t].value() for t in range(T)])
        pv2b_v = np.array([pv2b[t].value() for t in range(T)])
        return obj, ch_v, dh_v, soc_v, pv2b_v

    # -- 3.4) Zwei Fälle lösen und End-Progress setzen --
    obj_w, ch_w, dh_w, soc_w, pv2b_w = solve(pv_use, enforce_grid=True)
    obj_n, ch_n, dh_n, soc_n, pv2b_n = solve(np.zeros(T), enforce_grid=not baseline_no_grid)

    if progress_callback:
        progress_callback(100)

    return (
        timestamps,
        prices_mwh,
        pv_feed,
        obj_w,
        ch_w,
        dh_w,
        soc_w,
        pv2b_w,
        obj_n,
        ch_n,
        dh_n,
        soc_n,
        pv2b_n,
        interval_h,
    )

# ── 4) Datei-Uploads & Eingaben (VOR dem Button) ────────────────────────────
st.sidebar.markdown("### Datei-Uploads")
st.session_state.price_file = st.sidebar.file_uploader(
    "Strommarkt-Preise (Zeit, Preis €/MWh)", type=["csv", "xls", "xlsx"], key="price_file_upl"
)
st.session_state.pv_file = st.sidebar.file_uploader(
    "PV-Lastgang (Zeit, kWh)", type=["csv", "xls", "xlsx"], key="pv_file_upl"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Eingaben")
st.session_state.start_soc = st.sidebar.number_input("Start-SoC (kWh)", 0.0, 1e6, 0.0, step=1.0)
st.session_state.cap = st.sidebar.number_input("Kapazität (kWh)", 0.0, 1e6, 4472.0, step=1.0)
st.session_state.bat_kw = st.sidebar.number_input("Batterieleistung (kW)", 0.0, 1e6, 559.0, step=1.0)
st.session_state.grid_kw = st.sidebar.number_input("Netzanschluss (kW)", 0.0, 1e6, 757.5, step=1.0)
st.session_state.eff_pct = (
    st.sidebar.number_input("Round-Trip Eff. (%)", 0.0, 100.0, 91.0, step=0.1, format="%.1f") / 100.0
)
st.session_state.max_cycles = st.sidebar.number_input("Zyklen/Jahr (Budget)", 0.0, 1e4, 548.0, step=1.0)
st.session_state.end_soc_same_as_start = st.sidebar.checkbox("End-SoC = Start-SoC", value=False)
# EEG-Vergütung (ct/kWh) → € / kWh
st.session_state.eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung [ct/kWh]", 0.0, 100.0, 8.0, step=0.1) / 100.0

# Idealisiertes Baseline-Setup
st.session_state.baseline_no_grid = st.sidebar.checkbox(
    "Baseline ohne Netzlimit (idealisiert)", value=False,
    help="Vergleich ohne Netzanschluss-Constraint; PV-Lauf bleibt mit Netzlimit."
)

# ── 5) ▶️ Simulation starten ────────────────────────────────────────────────
if st.sidebar.button("▶️ Simulation starten"):
    if not (st.session_state.price_file and st.session_state.pv_file):
        st.sidebar.error("Bitte zuerst beide Dateien hochladen.")
    else:
        st.session_state.results = compute_results(
            price_file=st.session_state.price_file,
            pv_file=st.session_state.pv_file,
            start_soc=st.session_state.start_soc,
            cap=st.session_state.cap,
            bat_kw=st.session_state.bat_kw,
            grid_kw=st.session_state.grid_kw,
            eff_pct=st.session_state.eff_pct,
            max_cycles=st.session_state.max_cycles,
            progress_callback=set_progress,
            baseline_no_grid=st.session_state.baseline_no_grid,
            end_soc_same_as_start=st.session_state.end_soc_same_as_start,
        )

# ── 6) Wenn noch keine Ergebnisse vorhanden sind ────────────────────────────
if "results" not in st.session_state:
    st.info("Bitte auf **Simulation starten** klicken.")
    st.stop()

# ── 7) Ergebnisse entpacken (inkl. Backward-Kompatibilität) ─────────────────
res = st.session_state.results

# Alte Struktur (ohne SoC) → migrieren
if isinstance(res, (list, tuple)) and len(res) == 10:
    (
        timestamps,
        prices_mwh,
        pv_feed,
        obj_w,
        ch_w,
        dh_w,
        obj_n,
        ch_n,
        dh_n,
        interval_h,
    ) = res

    eff = st.session_state.eff_pct ** 0.5
    start_soc_local = st.session_state.start_soc

    T = len(ch_w)
    soc_w = np.zeros(T)
    soc_n = np.zeros(T)

    prev = start_soc_local
    for t in range(T):
        soc_w[t] = prev + eff * ch_w[t] - dh_w[t] / eff
        prev = soc_w[t]

    prev = start_soc_local
    for t in range(T):
        soc_n[t] = prev + eff * ch_n[t] - dh_n[t] / eff
        prev = soc_n[t]

    st.session_state.results = (
        timestamps,
        prices_mwh,
        pv_feed,
        obj_w,
        ch_w,
        dh_w,
        soc_w,
        obj_n,
        ch_n,
        dh_n,
        soc_n,
        interval_h,
    )

elif isinstance(res, (list, tuple)) and len(res) == 14:
    pass
else:
    st.warning("Inkompatible Ergebnisstruktur. Bitte Simulation erneut starten.")
    st.stop()

(
    timestamps,
    prices_mwh,
    pv_feed,
    obj_w,
    ch_w,
    dh_w,
    soc_w,
    pv2b_w,
    obj_n,
    ch_n,
    dh_n,
    soc_n,
    pv2b_n,
    interval_h,
) = st.session_state.results

# ── 7b) Datensatz-Check (Plausibilität) ─────────────────────────────────────
T_points = len(timestamps)
# Versuche Intervall aus Zeitstempeln zu schätzen
if T_points >= 2:
    inferred_interval_minutes = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() / 60.0
else:
    inferred_interval_minutes = 0

# Gesamtdauer
if T_points >= 2:
    duration_days = (timestamps.iloc[-1] - timestamps.iloc[0]).days + 1
else:
    duration_days = 0

is_15min = abs(inferred_interval_minutes - 15.0) < 1e-3
is_full_year_points = (T_points == 35040)

if is_15min and is_full_year_points:
    st.success(f"Datensatz erkannt: 15‑Minuten-Auflösung, **{T_points}** Punkte → **volles Jahr**.")
else:
    st.warning(
        f"Datensatz-Check: Intervall ≈ {inferred_interval_minutes:.2f} min, Punkte = {T_points}, "
        f"Dauer ~ {duration_days} Tage. Erwartet wären 15 min & 35040 Punkte für ein volles Jahr."
    )

# ── 8) Kennzahlen ───────────────────────────────────────────────────────────
loss_abs = obj_n - obj_w
loss_pct = (loss_abs / obj_n * 100) if obj_n else 0.0

# EEG-Verlust durch bilanzielle Selbstladung (PV→BESS) – **nur für den betrachteten Datensatz**, keine Annualisierung
pv2b_sum = float(np.nansum(pv2b_w))  # kWh im Datensatz
EEG_rate_eur_per_kwh = st.session_state.eeg_ct_per_kwh
EEG_loss_period_eur = pv2b_sum * EEG_rate_eur_per_kwh

c1, c2, c3, c4 = st.columns(4)
c1.metric("Gewinn ohne PV", fmt_euro(obj_n))
c2.metric("Gewinn mit PV", fmt_euro(obj_w))
c3.metric("Differenz (ohne − mit)", fmt_euro(loss_abs), f"{loss_pct:.2f} %")
c4.metric("PV→BESS (Datensatz)", f"{pv2b_sum:,.0f}".replace(",", ".") + " kWh")

eeg_caption = (
    f"EEG-Verlust: **{fmt_euro(EEG_loss_period_eur)}** bei {EEG_rate_eur_per_kwh*100:.1f} ct/kWh "
    + ("(volles Jahr erkannt)" if is_15min and is_full_year_points else "(Zeitraum gemäß Datensatz)")
)
st.caption(eeg_caption)

# ── 9) Monats-Erlöse (Balken mit roter Kappe) ─────────────────────────────── (Balken mit roter Kappe) ─────────────────────────────── (Balken mit roter Kappe) ───────────────────────────────
rev_w = prices_mwh / 1000 * dh_w - prices_mwh / 1000 * ch_w
rev_n = prices_mwh / 1000 * dh_n - prices_mwh / 1000 * ch_n

# Aggregation: Monats-Summen
_df_month = (
    pd.DataFrame({"Datum": timestamps, "ohne PV": rev_n, "mit PV": rev_w})
    .set_index("Datum")
    .resample("M").sum()
)

st.subheader("Erlöse (monatsweise)")

pos = np.arange(len(_df_month))
width = 0.7
months = [d.strftime("%b") for d in _df_month.index]

loss = (_df_month["ohne PV"] - _df_month["mit PV"]).clip(lower=0)

fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.bar(pos, _df_month["mit PV"], width=width, label="mit PV", zorder=3)
ax1.bar(pos, loss, width=width, bottom=_df_month["mit PV"], label="Differenz (ohne−mit)", zorder=3)
ax1.legend(loc="upper left")
ax1.set_xticks(pos)
ax1.set_xticklabels(months)

# Dynamische Y-Ticks
if len(_df_month):
    ymax = float(_df_month["ohne PV"].max())
else:
    ymax = 1.0
step = 10 ** np.floor(np.log10(max(1.0, ymax / 5)))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(step))
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,d}".replace(",", ".") + " €"))
ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

st.pyplot(fig1, use_container_width=True)

# ── 10) Kumulierte Erlöse (Linien) ──────────────────────────────────────────
cum = (
    pd.DataFrame({"Datum": timestamps, "ohne PV": rev_n, "mit PV": rev_w})
    .set_index("Datum")
    .resample("D").sum()
    .cumsum()
)

st.subheader("Kumulierte Erlöse")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(cum.index, cum["ohne PV"], label="ohne PV")
ax2.plot(cum.index, cum["mit PV"], label="mit PV")

# Dynamische Y-Achse
if len(cum):
    ymax2 = float(max(cum["ohne PV"].max(), cum["mit PV"].max()))
else:
    ymax2 = 1.0
step2 = 10 ** np.floor(np.log10(max(1.0, ymax2 / 5)))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(step2))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,d}".replace(",", ".") + " €"))
ax2.grid(axis="y", linestyle="--", alpha=0.3)

ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig2.autofmt_xdate()
ax2.legend(loc="upper left")

st.pyplot(fig2, use_container_width=True)

# ── 11) Ergebnis-Tabelle & Download ─────────────────────────────────────────
loss_factor = 1 - (st.session_state.eff_pct ** 0.5)
cycles_w = (ch_w + dh_w) / (2 * st.session_state.cap)

out = pd.DataFrame({
    "Zeitstempel": timestamps,
    "Preis (€/MWh)": prices_mwh,
    "PV-Einspeisung (kWh)": pv_feed,
    "PV→BESS (kWh)": pv2b_w,
    "PV-genutzt (kWh)": np.minimum(pv_feed, st.session_state.grid_kw * interval_h),
    "Ladeaktiv": (ch_w > 0).astype(int),
    "Entladeaktiv": (dh_w > 0).astype(int),
    "Lade-kWh": ch_w,
    "Entlade-kWh": dh_w,
    "SoC (kWh)": soc_w,
    "Verlust (kWh)": -(ch_w + dh_w) * loss_factor,
    "Kum. Zyklen": np.cumsum(cycles_w),
    "Netzlast (kWh)": np.minimum(pv_feed, st.session_state.grid_kw * interval_h) - ch_w + dh_w,
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
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
