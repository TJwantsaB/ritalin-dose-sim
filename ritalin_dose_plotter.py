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

Cross‑platform build – GitHub Actions CI
---------------------------------------
Don’t have Windows at hand? Push this repo to GitHub and add **.github/workflows/build.yml**:

```yaml
name: Build Windows EXE

on:
  workflow_dispatch:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: |
          python -m pip install --upgrade pip
          pip install streamlit plotly numpy pyinstaller
          pyinstaller --noconfirm --onefile ritalin_dose_plotter.py
      - uses: actions/upload-artifact@v4
        with:
          name: ritalin_dose_plotter.exe
          path: dist/ritalin_dose_plotter.exe
```

Click **Actions → Build Windows EXE** after pushing – GitHub will hand back a downloadable `.exe` artifact, no Windows PC required.

The executable will appear under **dist/**. On Linux it has no extension; on Windows it ends with .exe.

Double‑click (or `./dist/ritalin_dose_plotter`) and note the local URL, e.g. http://localhost:8501 – open it in your browser to use the app.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import root_scalar

# ----------------------------
# Pharmacokinetic core routine
# ----------------------------

def concentration_profile(dose_times, dose_amounts, half_life, k_absorb, t_end=48.0, dt=0.1):
    """Return (t_array, concentration_array) for a 1-compartment oral model with first-order absorption."""
    k = np.log(2) / half_life  # elimination rate constant (h⁻¹)
    t = np.arange(0.0, t_end + dt, dt)
    conc = np.zeros_like(t)
    for t_d, dose in zip(dose_times, dose_amounts):
        delta_t = np.maximum(t - t_d, 0)
        term = dose * k_absorb / (k_absorb - k)
        conc += term * (np.exp(-k * delta_t) - np.exp(-k_absorb * delta_t))
    return t, conc

def estimate_ka(tmax, half_life):
    ke = np.log(2) / half_life
    def f(ka):
        if ka <= ke:
            return np.inf
        return (np.log(ka) - np.log(ke)) / (ka - ke) - tmax
    result = root_scalar(f, bracket=[ke + 1e-6, 10], method='brentq')
    return result.root if result.converged else None

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Ritalin PK Simulator", layout="wide")
st.title("💊 Ritalin (Methylphenidate) Concentration Simulator")

with st.sidebar:
    st.header("Simulation settings")
    half_life = st.number_input("Half-life (hours)", value=3.0, min_value=0.5, step=0.5)
    use_tmax = st.checkbox("Estimate absorption from Tmax?", value=False)

    if use_tmax:
        tmax = st.number_input("Tmax (hours)", value=2.0, min_value=0.1, step=0.1)
        k_absorb = estimate_ka(tmax, half_life)
        if k_absorb is not None:
            st.markdown(f"Estimated ka: **{k_absorb:.2f} 1/h**")
        else:
            st.error("Could not estimate ka. Try a different Tmax or half-life.")
    else:
        k_absorb = st.number_input(
            "Absorption rate constant (1/h)", value=1.5, min_value=0.1, step=0.1,
            help="Typical range: 1.0–2.0 for immediate-release; 0.3–1.0 for extended-release"
        )

    t_end = st.number_input("Simulation length (hours)", value=48.0, min_value=6.0, step=6.0)
    dt = st.number_input("Time step (hours)", value=0.1, min_value=0.01, step=0.05, format="%0.2f")
    st.markdown("---")
    st.caption("Add your doses below →")

# Dose table
n_doses = st.number_input("Number of doses", min_value=1, max_value=24, value=4, step=1)

# Build a table-like interface with columns
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

if st.button("Run simulation"):
    t, conc = concentration_profile(dose_times, dose_amounts, half_life, k_absorb, t_end, dt)

    # Convert t-axis to datetime labels starting from 00:00 for nicer display
    start_dt = datetime.strptime("00:00", "%H:%M")
    time_labels = [start_dt + timedelta(hours=float(h)) for h in t]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time_labels, y=conc, mode="lines", name="Total drug", hovertemplate="%{y:.2f} mg at %{x|%H:%M}<extra></extra>")
    )
    fig.update_layout(
        xaxis_title="Time of day",
        yaxis_title="Amount in body (mg)",
        xaxis=dict(tickformat="%H:%M", dtick=3600000*4),
        template="plotly_white",
        margin=dict(l=40, r=30, t=40, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show summary metrics
    st.markdown("### Quick metrics")
    st.write(f"**Peak amount:** {conc.max():.2f} mg at t = {t[conc.argmax()]:.1f} h")
    auc = np.trapz(conc, t)
    st.write(f"**AUC (0-{t_end:.0f} h):** {auc:.1f} mg·h")
