import sys
from pathlib import Path
import streamlit as st

# Ensure repo root is on PYTHONPATH when running "streamlit run gui/app.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(page_title="Hybrid Simulation GUI", layout="wide")

st.title("Hybrid Simulation GUI")
st.write("Use the sidebar to run a model or view an existing `.npz` result.")

st.info(
    "MVP: Run from a JSON spec (mass-action reaction terms), or drag & drop `.npz` to view plots."
)