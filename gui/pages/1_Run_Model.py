import sys
from pathlib import Path
from datetime import datetime
import os
import json
import streamlit as st
import numpy as np
import pandas as pd

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from srcm_engine.core import HybridModel
from srcm_engine.results.io import save_npz

st.set_page_config(
    page_title="SRCM Pro Runner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stTextArea textarea { font-family: 'Courier New', Courier, monospace; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔬 Hybrid Model Studio")
st.caption("Configure, Simulate, and Export SRCM Hybrid Models")

# --- 1. SIDEBAR: EXECUTION & STORAGE ---
with st.sidebar:
    st.header("⚙️ Global Settings")

    with st.expander("Output & Pathing", expanded=True):
        default_dir = str(Path.home() / "Documents/srcm_runs")
        save_dir_str = st.text_input("Save Directory", value=default_dir)
        run_name = st.text_input("Run Name", value="hybrid_experiment")

    with st.expander("Simulation Parameters", expanded=True):
        total_time = st.number_input("Total Time (s)", value=30.0, step=1.0)
        dt = st.number_input("Time Step (dt)", value=0.006, format="%.4f")
        repeats = st.number_input("Repeats", value=10, min_value=1)
        parallel = st.checkbox("Parallel Execution", value=True)

    
    with st.expander("🔁 Hybrid Conversion", expanded=True):
        conv_threshold = st.number_input(
            "Conversion Threshold",
            value=4,
            min_value=0,
            step=1,
            help="Typically: particle count threshold for converting SSA <-> PDE (engine-defined)."
        )
        conv_rate = st.number_input(
            "Conversion Rate",
            value=1.0,
            min_value=0.0,
            step=0.1,
            help="Rate/strength of conversion (engine-defined)."
        )

# --- 2. MODEL ARCHITECTURE ---
c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.subheader("🧬 Physical Domain")
    with st.container(border=True):
        species_raw = st.text_input("Species", value="A, B", help="Comma separated names like: A, B, C")
        species = [s.strip() for s in species_raw.split(",") if s.strip()]

        if len(species) == 0:
            st.error("You must provide at least one species name.")
            st.stop()

        s_map = {s: i for i, s in enumerate(species)}

        col_d1, col_d2, col_d3 = st.columns(3)
        L = col_d1.number_input("Length (L)", value=10.0)
        K = int(col_d2.number_input("SSA Bins (K)", value=40, min_value=1, step=1))
        pde_mult = int(col_d3.number_input("PDE Factor", value=8, min_value=1, step=1))

        st.write("**Diffusion Coefficients**")
        diff_values = {}
        d_cols = st.columns(len(species))
        for i, s in enumerate(species):
            diff_values[s] = d_cols[i].number_input(f"D[{s}]", value=0.1)

    st.subheader("🔢 Reaction Rates")
    rates_raw = st.text_area("Constants (JSON)", value='{"alpha": 0.01, "beta": 0.01}', height=110)
    try:
        rates = json.loads(rates_raw) if rates_raw.strip() else {}
        if not isinstance(rates, dict):
            raise ValueError("Rates JSON must be an object/dict.")
    except Exception as e:
        st.warning(f"Invalid JSON in Rates: {e}")
        rates = {}

with c2:
    st.subheader("⚗️ Reaction Logic")
    with st.container(border=True):
        st.caption("🧪 PDE Drift/Reaction (Python tuple-like expressions)")
        st.caption("Write ONE expression per species, separated by commas (newlines are okay).")
        pde_code = st.text_area(
            "Reaction Vector",
            value="r['beta'] * B - r['alpha'] * A,\n r['alpha'] * A - r['beta'] * B",
            height=140,
            help="Example for species [A,B]: expr_for_A, expr_for_B"
        )

        st.caption("🎲 SSA Discrete Events (LHS -> RHS ; RateName)")
        ssa_raw = st.text_area(
            "Jump Processes",
            value="A -> B ; alpha\nB -> A ; beta",
            height=140,
            help="Example: A + B -> C ; k1"
        )

# --- 3. INITIAL CONDITIONS (MASS & BINS) ---
st.divider()
st.subheader("📐 Initial Distribution")

# Storage arrays (always created)
init_ssa = np.zeros((len(species), K), dtype=int)
init_pde = np.zeros((len(species), K * pde_mult), dtype=float)

with st.container(border=True):
    ic_type = st.radio("Target Layer", ["SSA (Particles)", "PDE (Mass/Concentration)"], horizontal=True)

    ic_mode = st.selectbox(
        "Spatial Preset",
        ["Manual (Zeros)", "Split (Half/Half)", "Uniform Fill", "Manual (Ranges)", "Manual (Table)"]
    )

    # Helpers
    def parse_side(side: str):
        """
        Parse 'A + A + B' into {'A':2,'B':1}.
        """
        parts = [p.strip() for p in side.split("+") if p.strip()]
        out = {}
        for p in parts:
            out[p] = out.get(p, 0) + 1
        return out

    def apply_ranges_to_array(layer: str, ranges_df: pd.DataFrame):
        """
        ranges_df columns: species, start, end, value
        start/end inclusive, 0-indexed bins
        """
        nonlocal_init_ssa = init_ssa
        nonlocal_init_pde = init_pde

        n_bins = K if layer == "SSA (Particles)" else (K * pde_mult)

        for _, row in ranges_df.iterrows():
            sp = str(row["species"]).strip()
            if sp not in s_map:
                continue

            start = int(row["start"])
            end = int(row["end"])
            start = max(0, min(start, n_bins - 1))
            end = max(0, min(end, n_bins - 1))
            if end < start:
                start, end = end, start

            val = row["value"]
            idx = s_map[sp]

            if layer == "SSA (Particles)":
                nonlocal_init_ssa[idx, start : end + 1] = int(val)
            else:
                nonlocal_init_pde[idx, start : end + 1] = float(val)

        # write back (keeps intent obvious)
        init_ssa[:, :] = nonlocal_init_ssa
        init_pde[:, :] = nonlocal_init_pde

    # Presets
    if ic_mode == "Split (Half/Half)":
        ic_col1, ic_col2, ic_col3 = st.columns(3)
        s_left = ic_col1.selectbox("Left Side Species", species, index=0)
        s_right = ic_col1.selectbox("Right Side Species", species, index=min(1, len(species) - 1))

        default_val = 10.0 if ic_type == "PDE (Mass/Concentration)" else 10
        val_left = ic_col2.number_input("Left Mass/Count", value=default_val)
        val_right = ic_col3.number_input("Right Mass/Count", value=default_val)

        if ic_type == "SSA (Particles)":
            init_ssa[s_map[s_left], : K // 2] = int(val_left)
            init_ssa[s_map[s_right], K // 2 :] = int(val_right)
        else:
            n = K * pde_mult
            mid = n // 2
            init_pde[s_map[s_left], :mid] = float(val_left)
            init_pde[s_map[s_right], mid:] = float(val_right)

    elif ic_mode == "Uniform Fill":
        ic_col1, ic_col2 = st.columns(2)
        s_target = ic_col1.selectbox("Target Species", species)
        val_all = ic_col2.number_input("Mass/Count per Compartment", value=5.0)

        if ic_type == "SSA (Particles)":
            init_ssa[s_map[s_target], :] = int(val_all)
        else:
            init_pde[s_map[s_target], :] = float(val_all)

    elif ic_mode == "Manual (Ranges)":
        layer_bins = K if ic_type == "SSA (Particles)" else (K * pde_mult)
        st.caption(
            f"Define ranges of compartments (0-indexed) and set values. "
            f"Current layer has **{layer_bins}** compartments."
        )

        default_ranges = pd.DataFrame(
            [
                {"species": species[0], "start": 0, "end": max(0, layer_bins // 2 - 1), "value": 10},
                {"species": species[min(1, len(species) - 1)], "start": layer_bins // 2, "end": layer_bins - 1, "value": 10},
            ]
        )

        ranges_df = st.data_editor(
            default_ranges,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "species": st.column_config.SelectboxColumn("species", options=species, required=True),
                "start": st.column_config.NumberColumn("start", min_value=0, step=1, required=True),
                "end": st.column_config.NumberColumn("end", min_value=0, step=1, required=True),
                "value": st.column_config.NumberColumn("value", required=True),
            },
        )

        apply_ranges_to_array(ic_type, ranges_df)

    elif ic_mode == "Manual (Table)":
        st.caption("Edit every compartment directly (fine for smaller K / PDE grids).")

        if ic_type == "SSA (Particles)":
            df = pd.DataFrame(init_ssa, index=species, columns=[f"bin_{i}" for i in range(K)])
            edited = st.data_editor(df, use_container_width=True)
            # sanitize -> ints
            init_ssa[:, :] = np.array(edited, dtype=float).round().astype(int)

        else:
            n = K * pde_mult
            df = pd.DataFrame(init_pde, index=species, columns=[f"cell_{i}" for i in range(n)])
            edited = st.data_editor(df, use_container_width=True)
            init_pde[:, :] = np.array(edited, dtype=float)

    # Preview a quick summary
    st.markdown("**Initial condition summary**")
    if ic_type == "SSA (Particles)":
        totals = init_ssa.sum(axis=1)
        st.write({species[i]: int(totals[i]) for i in range(len(species))})
    else:
        totals = init_pde.sum(axis=1)
        st.write({species[i]: float(totals[i]) for i in range(len(species))})

# --- 4. EXECUTION ---
st.write("")
if st.button("🚀 RUN SIMULATION"):
    try:
        # Resolve Path
        save_path = Path(os.path.expanduser(save_dir_str)).resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        # Build Model
        m = HybridModel(species=species)
        m.domain(L=L, K=K, pde_multiple=pde_mult, boundary="zero-flux")
        m.diffusion(**diff_values)
        m.conversion(threshold=4, rate=1.0)  # Standard PhD threshold

        # PDE logic
        # NOTE: eval is powerful but risky. If you're the only user, fine.
        # If sharing publicly, replace this with a safer parser.
        pde_func = eval(
            f"lambda {', '.join(species)}, r: ({pde_code})",
            {"np": np},
            {}
        )
        m.reaction_terms(pde_func)

        # SSA logic
        for line in ssa_raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "->" in line and ";" in line:
                eq, rname = line.split(";", 1)
                lhs, rhs = eq.split("->", 1)
                m.add_reaction(parse_side(lhs), parse_side(rhs), rate_name=rname.strip())

        m.build(rates=rates)

        with st.status("Simulating Model...", expanded=True) as status:
            st.write("Initializing arrays...")
            res = m.run_repeats(
                init_ssa,
                init_pde,
                time=total_time,
                dt=dt,
                repeats=int(repeats),
                seed=1,
                parallel=parallel,
                n_jobs=-1,
                progress=True
            )
            status.update(label="Simulation Complete!", state="complete")

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = save_path / f"{run_name}_{timestamp}.npz"

        meta = m.metadata()
        meta.update({"total_time": total_time, "dt": dt, "repeats": int(repeats), "ic_mode": ic_mode, "ic_type": ic_type})
        save_npz(res, str(out_file), meta=meta)

        st.balloons()
        st.success(f"**Success!** Data written to: `{out_file}`")
        st.info("💡 Copy the command below to visualize your results:")
        st.code(f"python view_npz.py \"{out_file}\" --plot --anim")

    except Exception as e:
        st.error(f"Critical Failure: {e}")
        st.exception(e)
