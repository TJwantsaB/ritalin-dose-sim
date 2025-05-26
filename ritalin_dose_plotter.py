"""
Ritalin Concentration Simulator
--------------------------------
Run with Streamlit during development:
    streamlit run ritalin_dose_plotter.py

Packaging quick‑start
---------------------
* **PyInstaller builds are OS‑specific.** Build on Windows for a .exe, build on Linux for an ELF binary – cross‑compiling isn’t supported out‑of‑the‑box.
* **Linux tip:** if you get “Failed to load Python shared library …libpythonX.Y.so”, make sure the matching *-dev* package (or a Python built with `--enable-shared`) is available. Example on Ubuntu/Debian:
      sudo apt install python3.10-dev
* **Stable Python versions:** PyInstaller is rock‑solid with 3.8‑3.11. Python 3.12 works with the latest PyInstaller betas; 3.13 is still experimental. If you see linker errors, try a 3.10 venv.

Build commands
--------------
Windows (PowerShell):
    pip install streamlit plotly numpy pyinstaller
    pyinstaller --noconfirm --onefile ritalin_dose_plotter.py

Linux/macOS (bash):
    python -m venv venv && source venv/bin/activate
    pip install streamlit plotly numpy pyinstaller
    pyinstaller --noconfirm --onefile ritalin_dose_plotter.py

The executable will appear under **dist/**. On Linux it has no extension; on Windows it ends with .exe.

Double‑click (or `./dist/ritalin_dose_plotter`) and note the local URL, e.g. http://localhost:8501 – open it in your browser to use the app.


Run with Streamlit during development:
    streamlit run ritalin_dose_plotter.py

Build a standalone Windows EXE (bundles Python + dependencies):
    # First install required libs in a clean venv
    pip install streamlit plotly numpy pyinstaller

    # Build the executable (one‑file)
    pyinstaller --noconfirm --onefile ritalin_dose_plotter.py

The EXE will appear in the dist/ folder. Double‑click it and a
terminal window will open telling you which local URL to visit, e.g.:
    http://localhost:8501
Note: the first run may take a moment while Streamlit spins up.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------------------
# Pharmacokinetic core routine
# ----------------------------

def concentration_profile(dose_times, dose_amounts, half_life, t_end=48.0, dt=0.1):
    """Return (t_array, concentration_array) for a 1‑compartment i.v. bolus model.

    Parameters
    ----------
    dose_times : list[float]
        Times (h) since t=0 when each dose is taken.
    dose_amounts : list[float]
        Amount (mg) of each corresponding dose.
    half_life : float
        Elimination half‑life in hours.
    t_end : float, optional
        Simulation span in hours (default 48).
    dt : float, optional
        Time‑step in hours (default 0.1 h = 6 min).
    """
    k = np.log(2) / half_life  # elimination rate constant (h⁻¹)
    t = np.arange(0.0, t_end + dt, dt)
    conc = np.zeros_like(t)
    for t_d, dose in zip(dose_times, dose_amounts):
        conc += dose * np.where(t >= t_d, np.exp(-k * (t - t_d)), 0.0)
    return t, conc

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Ritalin PK Simulator", layout="wide")
st.title("💊 Ritalin (Methylphenidate) Concentration Simulator")

with st.sidebar:
    st.header("Simulation settings")
    half_life = st.number_input("Half‑life (hours)", value=3.0, min_value=0.5, step=0.5)
    t_end = st.number_input("Simulation length (hours)", value=48.0, min_value=6.0, step=6.0)
    dt = st.number_input("Time step (hours)", value=0.1, min_value=0.01, step=0.05, format="%0.2f")
    st.markdown("---")
    st.caption("Add your doses below →")

# Dose table
n_doses = st.number_input("Number of doses", min_value=1, max_value=24, value=4, step=1)

# Build a table‑like interface with columns
st.subheader("Dose schedule (time in hours from t0 and amount in mg)")
dose_times = []
dose_amounts = []
for i in range(int(n_doses)):
    c1, c2 = st.columns(2)
    with c1:
        t_h = st.number_input(f"Time of dose {i+1} (h)", key=f"time_{i}", value=12.0 + 4*i)
    with c2:
        amt = st.number_input(f"Dose {i+1} amount (mg)", key=f"amt_{i}", value=5.0)
    dose_times.append(float(t_h))
    dose_amounts.append(float(amt))

if st.button("Run simulation"):  # ------------------- RUN -------------------
    t, conc = concentration_profile(dose_times, dose_amounts, half_life, t_end, dt)

    # Convert t‑axis to datetime labels starting from 00:00 for nicer display
    start_dt = datetime.strptime("00:00", "%H:%M")
    time_labels = [start_dt + timedelta(hours=float(h)) for h in t]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time_labels, y=conc, mode="lines", name="Total drug", hovertemplate="%{y:.2f} mg at %{x|%H:%M}<extra></extra>")
    )
    fig.update_layout(
        xaxis_title="Time of day",
        yaxis_title="Amount in body (mg)",
        xaxis=dict(tickformat="%H:%M", dtick=3600000*4),  # 4‑hour ticks
        template="plotly_white",
        margin=dict(l=40, r=30, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show summary metrics
    st.markdown("### Quick metrics")
    st.write(f"**Peak amount:** {conc.max():.2f} mg at t = {t[conc.argmax()]:.1f} h")
    auc = np.trapz(conc, t)
    st.write(f"**AUC (0‑{t_end:.0f} h):** {auc:.1f} mg·h")
