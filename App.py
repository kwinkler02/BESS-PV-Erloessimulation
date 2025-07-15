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

# 2) Input-Dateien & Parameter wie gehabt …
#    (Omitted for brevity; bleibt unverändert)

# 4) Abbruch wenn nicht gestartet
if not st.session_state.run:
    st.info("Bitte auf **Simulation starten** klicken, um die Optimierung auszuführen.")
    st.stop()

# 5) Fortschritts-Container im Hauptbereich
progress_bar = st.empty()      # Platz für die Balken-UI
progress_text = st.empty()     # Platz für die Prozent-Anzeige

def set_progress(pct:int):
    """Hilfsfunktion, um Fortschritt in beiden Containern zu setzen."""
    progress_bar.progress(pct)
    progress_text.markdown(f"**Fortschritt:** {pct}%")

set_progress(0)                # initial auf 0 %

# 6) Beispielschritt: Dateien einlesen
set_progress(10)
# load_df, df_price, df_pv …

# 7) Daten prüfen und Vorverarbeitung
set_progress(20)
# timestamps, prices, pv_feed …

# 8) PV-Nutzung berechnen
set_progress(30)
# pv_use …

# 9) Optimierungsfunktion mit groben Updates
def solve_arbitrage(pv_vec, prog_target):
    # … Modell aufbauen und lösen …
    # nach dem solve:
    set_progress(prog_target)
    return obj, ch, dh, soc

with st.spinner("⏳ Berechne mit PV…"):
    obj_w, ch_w, dh_w, soc_w = solve_arbitrage(pv_use, prog_target=60)

with st.spinner("⏳ Berechne ohne PV…"):
    obj_n, ch_n, dh_n, soc_n = solve_arbitrage(np.zeros(len(pv_use)), prog_target=90)

set_progress(100)              # fertig

# 10) Ergebnisse anzeigen …
#    (Kennzahlen, DataFrame, Download-Button wie gehabt)
