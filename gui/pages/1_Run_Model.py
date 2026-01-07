import sys
from pathlib import Path
from datetime import datetime
import os
import re
import json
import streamlit as st
import numpy as np
import time

# Ensure repo root on path (so srcm_engine + gui imports work when running from repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.backend.builder import build_model_from_spec
from srcm_engine.results.io import save_npz  # save directly, no temp

st.title("Run a Model (JSON spec MVP)")

default_spec = {
    "species": ["A", "B"],
    "domain": {"L": 10.0, "K": 40, "pde_multiple": 8, "boundary": "zero-flux"},
    "diffusion": {"A": 0.1, "B": 0.1},
    "conversion": {"threshold": 4, "rate": 1.0},
    "reactions": [
        {"reactants": {"A": 1}, "products": {"B": 1}, "rate_name": "alpha"},
        {"reactants": {"B": 1}, "products": {"A": 1}, "rate_name": "beta"},
    ],
    "rates": {"alpha": 0.01, "beta": 0.01},
    "run": {
        "total_time": 30.0,
        "dt": 0.006,
        "repeats": 10,
        "seed": 1,
        "parallel": True,
        "n_jobs": -1,
    },
    "ic": {"mode": "split", "left_species": "A", "right_species": "B", "count": 10},
}

# --- Save settings (absolute OR relative, user's choice) ---
st.subheader("Save output")

default_out_dir = str(Path.home() / "sim_runs")
save_dir_text = st.text_input(
    "Save folder path (absolute or relative). Examples: /Users/you/runs  or  ~/runs  or  runs/",
    value=default_out_dir,
)

run_name = st.text_input("Run name (used in filename)", value="ab_switch")
run_name_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_") or "run"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Expand ~ and env vars, resolve absolute path (relative paths resolve from current working dir)
save_dir = Path(os.path.expandvars(os.path.expanduser(save_dir_text))).resolve()

try:
    save_dir.mkdir(parents=True, exist_ok=True)
except Exception as e:
    st.error(f"Could not create/access folder: {save_dir}")
    st.caption(f"Error: {e}")
    st.stop()

out_npz = save_dir / f"{run_name_safe}_{timestamp}.npz"
out_json = save_dir / f"{run_name_safe}_{timestamp}.json"

st.caption(f"Will save to:\n- `{out_npz}`\n- `{out_json}`")

# --- Spec editor ---
spec_text = st.text_area(
    "Model spec (JSON)",
    value=json.dumps(default_spec, indent=2),
    height=420,
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run_clicked = st.button("Run simulation", type="primary")
with col2:
    quick = st.button("Quick test (repeats=2)")
with col3:
    st.caption("Tip: start with repeats=2–10 while debugging.")

if not (run_clicked or quick):
    st.stop()

# Parse JSON
try:
    spec = json.loads(spec_text)
except Exception as e:
    st.error(f"Invalid JSON: {e}")
    st.stop()

if quick:
    spec["run"]["repeats"] = 2
    spec["run"]["total_time"] = min(float(spec["run"]["total_time"]), 10.0)

# Build model
try:
    m = build_model_from_spec(spec)
except Exception as e:
    st.error("Failed to build model from spec.")
    st.exception(e)
    st.stop()

# Initial conditions (MVP)
K = int(spec["domain"]["K"])
pde_multiple = int(spec["domain"]["pde_multiple"])
n_species = len(spec["species"])

init_ssa = np.zeros((n_species, K), dtype=int)
init_pde = np.zeros((n_species, K * pde_multiple), dtype=float)

ic = spec.get("ic", {"mode": "zeros"})
if ic.get("mode") == "split" and n_species >= 2:
    left = ic.get("left_species", spec["species"][0])
    right = ic.get("right_species", spec["species"][1])
    count = int(ic.get("count", 10))

    sidx = {s: i for i, s in enumerate(spec["species"])}
    init_ssa[sidx[left], : K // 4] = count
    init_ssa[sidx[right], 3 * K // 4 :] = count

# --- "Indeterminate" progress UI (since tqdm is in terminal) ---
status = st.empty()
bar = st.progress(0, text="Starting…")


def pulse_progress(seconds: float = 0.6, start: int = 10, end: int = 30):
    """Small visual heartbeat before the blocking call."""
    steps = 10
    for i in range(steps):
        p = start + int((end - start) * (i / (steps - 1)))
        bar.progress(p, text="Running… (tqdm in terminal)")
        time.sleep(seconds / steps)


# Run
r = spec["run"]
try:
    status.info("Running simulation… (tqdm will appear in the terminal)")
    bar.progress(5, text="Preparing run…")
    pulse_progress()

    with st.spinner("Running simulation..."):
        res = m.run_repeats(
            init_ssa,
            init_pde,
            time=float(r["total_time"]),
            dt=float(r["dt"]),
            repeats=int(r["repeats"]),
            seed=int(r["seed"]),
            parallel=bool(r.get("parallel", False)),
            n_jobs=int(r.get("n_jobs", 1)),
            progress=True,
        )

    bar.progress(85, text="Saving results…")

except Exception as e:
    st.error("Simulation failed.")
    st.exception(e)
    st.stop()

# Metadata (+ spec for reproducibility)
meta = m.metadata()
meta.update({"spec": spec})

# Save outputs (no loading back)
try:
    save_npz(res, str(out_npz), meta=meta)
    out_json.write_text(json.dumps(spec, indent=2))
except Exception as e:
    st.error("Failed to save outputs.")
    st.exception(e)
    st.stop()

bar.progress(100, text="Done ✅")
status.success("Done.")

st.success(f"Saved results: {out_npz}")
st.info(f"Saved spec: {out_json}")

# Handy copy-paste command for your local viewer script (quotes handle spaces in paths)
st.code(f'python view_npz.py "{out_npz}" --plot --anim')
