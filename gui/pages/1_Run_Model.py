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

# -----------------------------
# Streamlit config + styling
# -----------------------------
st.set_page_config(
    page_title="SRCM Pro Runner",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .run-button { background-color: #007bff; color: white; }
    .stop-button { background-color: #dc3545; color: white; }
    .stTextArea textarea { font-family: 'Courier New', Courier, monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Hybrid Model Studio")
st.caption("Configure, Simulate, and Export SRCM Hybrid Models")

# -----------------------------
# Cancel flag for stopping simulation
# -----------------------------
if 'cancel_simulation' not in st.session_state:
    st.session_state.cancel_simulation = False

if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

def stop_simulation():
    st.session_state.cancel_simulation = True
    st.session_state.simulation_running = False

# -----------------------------
# Preset utilities
# -----------------------------
PRESET_DIR = REPO_ROOT / "gui" / "presets"
PRESET_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PRESET_PATH = PRESET_DIR / "Turing_hybrid_bump.json"


def _safe_read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.sidebar.error(
            f"Failed to read preset JSON: {path.name}\n\n{e}\n\n"
            "Tip: JSON cannot contain comments like /* ... */."
        )
        return None


def preset_from_ui(
    name: str,
    species: list[str],
    L: float,
    K: int,
    pde_mult: int,
    boundary: str,
    diffusion: dict,
    conv_threshold: int,
    conv_rate: float,
    rates: dict,
    pde_code: str,
    ssa_raw: str,
    total_time: float,
    dt: float,
    repeats: int,
    parallel: bool,
    ic_type: str,
    ic_mode: str,
    init_ssa: np.ndarray,
    init_pde: np.ndarray,
):
    return {
        "name": name,
        "version": 1,
        "model": {
            "species": species,
            "domain": {
                "L": float(L),
                "K": int(K),
                "pde_multiple": int(pde_mult),
                "boundary": boundary,
            },
            "diffusion": {k: float(v) for k, v in diffusion.items()},
            "conversion": {"threshold": int(conv_threshold), "rate": float(conv_rate)},
            "rates": rates,
            "pde_code": pde_code,
            "ssa_raw": ssa_raw,
        },
        "run": {
            "total_time": float(total_time),
            "dt": float(dt),
            "repeats": int(repeats),
            "parallel": bool(parallel),
        },
        "initial_conditions": {
            "ic_type": ic_type,
            "ic_mode": ic_mode,
            "init_ssa": init_ssa.tolist(),
            "init_pde": init_pde.tolist(),
        },
    }


def apply_preset_to_session(preset: dict):
    """
    Push preset values into st.session_state keys that back the widgets.
    Also stores species order for IC remapping.
    """
    st.session_state["preset_loaded"] = True

    m = preset.get("model", {})
    r = preset.get("run", {})
    ic = preset.get("initial_conditions", {})

    species = m.get("species", ["A", "B"])
    domain = m.get("domain", {})
    diffusion = m.get("diffusion", {})
    conversion = m.get("conversion", {})
    rates = m.get("rates", {})

    # Widget-backed keys
    st.session_state["species_raw"] = ", ".join(species)
    st.session_state["L"] = float(domain.get("L", 10.0))
    st.session_state["K"] = int(domain.get("K", 40))
    st.session_state["pde_mult"] = int(domain.get("pde_multiple", 8))
    st.session_state["boundary"] = domain.get("boundary", "zero-flux")

    st.session_state["rates_raw"] = json.dumps(rates, indent=2)
    st.session_state["pde_code"] = m.get("pde_code", "")
    st.session_state["ssa_raw"] = m.get("ssa_raw", "")

    st.session_state["total_time"] = float(r.get("total_time", 30.0))
    st.session_state["dt"] = float(r.get("dt", 0.006))
    st.session_state["repeats"] = int(r.get("repeats", 10))
    st.session_state["parallel"] = bool(r.get("parallel", True))

    st.session_state["conv_threshold"] = int(conversion.get("threshold", 4))
    st.session_state["conv_rate"] = float(conversion.get("rate", 1.0))

    # Store diffusion per species for later
    st.session_state["diffusion_loaded"] = {str(k): float(v) for k, v in diffusion.items()}

    # Store species order for IC remapping
    st.session_state["init_species_loaded"] = list(species)

    # Optional: load IC arrays
    init_ssa = ic.get("init_ssa")
    init_pde = ic.get("init_pde")
    st.session_state["init_ssa_loaded"] = np.array(init_ssa, dtype=int) if init_ssa is not None else None
    st.session_state["init_pde_loaded"] = np.array(init_pde, dtype=float) if init_pde is not None else None
    st.session_state["ic_type_loaded"] = ic.get("ic_type")
    st.session_state["ic_mode_loaded"] = ic.get("ic_mode")


def list_preset_files() -> list[Path]:
    return sorted(PRESET_DIR.glob("*.json"))


def _remap_by_species(loaded_arr: np.ndarray, loaded_species: list[str], current_species: list[str]) -> np.ndarray | None:
    """
    Reorder loaded_arr from loaded_species order -> current_species order.
    Only keeps species that exist in both.
    """
    if loaded_species is None:
        return None
    if loaded_arr.ndim != 2:
        return None

    li = {s: i for i, s in enumerate(loaded_species)}
    out = np.zeros((len(current_species), loaded_arr.shape[1]), dtype=loaded_arr.dtype)

    for i, s in enumerate(current_species):
        if s in li:
            out[i, :] = loaded_arr[li[s], :]

    return out


def _apply_loaded_ics(
    init_ssa: np.ndarray,
    init_pde: np.ndarray,
    current_species: list[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Applies st.session_state loaded IC arrays to init_ssa/init_pde with safe remapping.
    Returns (init_ssa, init_pde, note).
    """
    note = "ui"

    loaded_ssa = st.session_state.get("init_ssa_loaded")
    loaded_pde = st.session_state.get("init_pde_loaded")
    loaded_species = st.session_state.get("init_species_loaded")

    # SSA
    if isinstance(loaded_ssa, np.ndarray):
        if isinstance(loaded_species, list) and len(loaded_species) == loaded_ssa.shape[0]:
            remapped = _remap_by_species(loaded_ssa, loaded_species, current_species)
            if remapped is not None and remapped.shape == init_ssa.shape:
                init_ssa[:, :] = remapped
                note = "loaded"
            elif loaded_ssa.shape == init_ssa.shape:
                init_ssa[:, :] = loaded_ssa
                note = "loaded"
        elif loaded_ssa.shape == init_ssa.shape:
            init_ssa[:, :] = loaded_ssa
            note = "loaded"

    # PDE
    if isinstance(loaded_pde, np.ndarray):
        if isinstance(loaded_species, list) and len(loaded_species) == loaded_pde.shape[0]:
            remapped = _remap_by_species(loaded_pde, loaded_species, current_species)
            if remapped is not None and remapped.shape == init_pde.shape:
                init_pde[:, :] = remapped
                note = "loaded"
            elif loaded_pde.shape == init_pde.shape:
                init_pde[:, :] = loaded_pde
                note = "loaded"
        elif loaded_pde.shape == init_pde.shape:
            init_pde[:, :] = loaded_pde
            note = "loaded"

    return init_ssa, init_pde, note


def parse_side(side: str):
    """
    Parse 'A + A + B' into {'A':2,'B':1}.
    Empty side -> {}.
    """
    side = side.strip()
    if side == "":
        return {}
    parts = [p.strip() for p in side.split("+") if p.strip()]
    out = {}
    for p in parts:
        out[p] = out.get(p, 0) + 1
    return out

# -----------------------------
# Custom progress callback that checks for cancellation
# -----------------------------
def create_progress_callback():
    """Create a progress callback that checks for cancellation"""
    def callback(current, total, description=""):
        # Check if simulation should be stopped
        if st.session_state.cancel_simulation:
            raise KeyboardInterrupt("Simulation cancelled by user")
        
        # Update progress bar if it exists
        if 'progress_bar' in st.session_state:
            st.session_state.progress_bar.progress(current / total, text=description)
    
    return callback

# -----------------------------
# Sidebar: presets + global settings
# -----------------------------
with st.sidebar:
    st.header("💾 Presets")

    preset_files = list_preset_files()
    preset_names = [p.name for p in preset_files]
    default_index = preset_names.index(DEFAULT_PRESET_PATH.name) if DEFAULT_PRESET_PATH in preset_files else 0

    chosen = st.selectbox(
        "Preset file",
        options=preset_names if preset_names else ["(no presets found)"],
        index=default_index if preset_names else 0,
        disabled=(not preset_names),
        key="chosen_preset",
    )

    colp1, colp2 = st.columns(2)
    if colp1.button("Load preset", disabled=(not preset_names)):
        preset_path = PRESET_DIR / chosen
        preset = _safe_read_json(preset_path)
        if preset:
            apply_preset_to_session(preset)
            st.success(f"Loaded: {preset.get('name', preset_path.stem)}")
            st.rerun()

    uploaded = st.file_uploader("Upload preset JSON", type=["json"])
    if uploaded is not None:
        try:
            preset = json.load(uploaded)
            apply_preset_to_session(preset)
            st.success(f"Loaded uploaded preset: {preset.get('name', 'unnamed')}")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid preset JSON: {e}")

    st.divider()
    st.header("⚙️ Global Settings")

    with st.expander("Output & Pathing", expanded=True):
        default_dir = str(Path.home() / "Documents/srcm_runs")
        save_dir_str = st.text_input("Save Directory", value=default_dir, key="save_dir_str")
        run_name = st.text_input("Run Name", value="hybrid_experiment", key="run_name")

    with st.expander("Simulation Parameters", expanded=True):
        total_time = st.number_input("Total Time (s)", value=30.0, step=1.0, key="total_time")
        dt = st.number_input("Time Step (dt)", value=0.006, format="%.4f", key="dt")
        repeats = st.number_input("Repeats", value=10, min_value=1, key="repeats")
        parallel = st.checkbox("Parallel Execution", value=True, key="parallel")

    with st.expander("🔁 Hybrid Conversion", expanded=True):
        conv_threshold = st.number_input(
            "Conversion Threshold",
            value=4,
            min_value=0,
            step=1,
            key="conv_threshold",
            help="Typically: particle count threshold for converting SSA <-> PDE (engine-defined).",
        )
        conv_rate = st.number_input(
            "Conversion Rate",
            value=1.0,
            min_value=0.0,
            step=0.1,
            key="conv_rate",
            help="Rate/strength of conversion (engine-defined).",
        )

    st.divider()
    st.subheader("🧫 Initial conditions")
    ic_npz = st.file_uploader("Load initial conditions (.npz)", type=["npz"])
    if ic_npz is not None:
        try:
            data = np.load(ic_npz, allow_pickle=True)
            st.session_state["init_ssa_loaded"] = data["init_ssa"].astype(int)
            st.session_state["init_pde_loaded"] = data["init_pde"].astype(float)

            if "species" in data:
                st.session_state["init_species_loaded"] = [str(x) for x in data["species"].tolist()]

            st.success("Loaded initial conditions from NPZ")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load IC NPZ: {e}")

    st.divider()
    st.subheader("📦 Save current UI as preset")
    preset_name = st.text_input("Preset name", value="my_preset", key="preset_name")
    preset_save_filename = st.text_input(
        "Preset filename",
        value=f"{preset_name}.json",
        help=f"Saved to: {PRESET_DIR}",
        key="preset_filename",
    )

# -----------------------------
# Model architecture
# -----------------------------
c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.subheader("🧬 Physical Domain")
    with st.container(border=True):
        species_raw = st.text_input(
            "Species",
            value=st.session_state.get("species_raw", "A, B"),
            key="species_raw",
            help="Comma separated names like: A, B, C",
        )
        species = [s.strip() for s in species_raw.split(",") if s.strip()]
        if len(species) == 0:
            st.error("You must provide at least one species name.")
            st.stop()

        s_map = {s: i for i, s in enumerate(species)}

        col_d1, col_d2, col_d3 = st.columns(3)
        L = col_d1.number_input("Length (L)", value=10.0, key="L")
        K = int(col_d2.number_input("SSA Bins (K)", value=40, min_value=1, step=1, key="K"))
        pde_mult = int(col_d3.number_input("PDE Factor", value=8, min_value=1, step=1, key="pde_mult"))

        boundary = st.selectbox("Boundary", ["zero-flux"], index=0, key="boundary")

        st.write("**Diffusion Coefficients**")
        diffusion_loaded = st.session_state.get("diffusion_loaded", {})
        diff_values = {}
        d_cols = st.columns(len(species))
        for i, s in enumerate(species):
            default_D = float(diffusion_loaded.get(s, 0.1))
            diff_values[s] = d_cols[i].number_input(f"D[{s}]", value=default_D, key=f"D_{s}")

    st.subheader("🔢 Reaction Rates")
    rates_raw = st.text_area(
        "Constants (JSON)",
        value=st.session_state.get("rates_raw", '{"alpha": 0.01, "beta": 0.01}'),
        height=110,
        key="rates_raw",
    )
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
            value=st.session_state.get(
                "pde_code",
                "r['beta'] * B - r['alpha'] * A,\n r['alpha'] * A - r['beta'] * B",
            ),
            height=140,
            key="pde_code",
            help="Example for species [A,B]: expr_for_A, expr_for_B",
        )

        st.caption("🎲 SSA Discrete Events (LHS -> RHS ; RateName)")
        ssa_raw = st.text_area(
            "Jump Processes",
            value=st.session_state.get("ssa_raw", "A -> B ; alpha\nB -> A ; beta"),
            height=140,
            key="ssa_raw",
            help="Example: A + B -> C ; k1",
        )

# -----------------------------
# Initial conditions
# -----------------------------
st.divider()
st.subheader("📐 Initial Distribution")

init_ssa = np.zeros((len(species), K), dtype=int)
init_pde = np.zeros((len(species), K * pde_mult), dtype=float)

# Apply loaded ICs (preset or NPZ), safely remapping by species if available
init_ssa, init_pde, ic_source = _apply_loaded_ics(init_ssa, init_pde, species)
st.caption(f"IC source: {ic_source}")

with st.container(border=True):
    ic_type_default = st.session_state.get("ic_type_loaded", "SSA (Particles)")
    ic_mode_default = st.session_state.get("ic_mode_loaded", "Manual (Zeros)")

    ic_type = st.radio(
        "Target Layer",
        ["SSA (Particles)", "PDE (Mass/Concentration)"],
        horizontal=True,
        index=0 if ic_type_default == "SSA (Particles)" else 1,
        key="ic_type",
    )

    ic_mode = st.selectbox(
        "Spatial Preset",
        ["Manual (Zeros)", "Split (Half/Half)", "Uniform Fill", "Manual (Ranges)", "Manual (Table)"],
        index=["Manual (Zeros)", "Split (Half/Half)", "Uniform Fill", "Manual (Ranges)", "Manual (Table)"].index(
            ic_mode_default
            if ic_mode_default in ["Manual (Zeros)", "Split (Half/Half)", "Uniform Fill", "Manual (Ranges)", "Manual (Table)"]
            else "Manual (Zeros)"
        ),
        key="ic_mode",
    )

    def apply_ranges_to_array(layer: str, ranges_df: pd.DataFrame):
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
                init_ssa[idx, start : end + 1] = int(val)
            else:
                init_pde[idx, start : end + 1] = float(val)

    if ic_mode == "Split (Half/Half)":
        ic_col1, ic_col2, ic_col3 = st.columns(3)
        s_left = ic_col1.selectbox("Left Side Species", species, index=0, key="ic_left")
        s_right = ic_col1.selectbox("Right Side Species", species, index=min(1, len(species) - 1), key="ic_right")

        default_val = 10.0 if ic_type == "PDE (Mass/Concentration)" else 10
        val_left = ic_col2.number_input("Left Mass/Count", value=default_val, key="ic_val_left")
        val_right = ic_col3.number_input("Right Mass/Count", value=default_val, key="ic_val_right")

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
        s_target = ic_col1.selectbox("Target Species", species, key="ic_uniform_species")
        val_all = ic_col2.number_input("Mass/Count per Compartment", value=5.0, key="ic_uniform_val")

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
                {
                    "species": species[min(1, len(species) - 1)],
                    "start": layer_bins // 2,
                    "end": layer_bins - 1,
                    "value": 10,
                },
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
            key="ranges_df",
        )

        apply_ranges_to_array(ic_type, ranges_df)

    elif ic_mode == "Manual (Table)":
        st.caption("Edit every compartment directly (fine for smaller K / PDE grids).")

        if ic_type == "SSA (Particles)":
            df = pd.DataFrame(init_ssa, index=species, columns=[f"bin_{i}" for i in range(K)])
            edited = st.data_editor(df, use_container_width=True, key="ic_table_ssa")
            init_ssa[:, :] = np.array(edited, dtype=float).round().astype(int)
        else:
            n = K * pde_mult
            df = pd.DataFrame(init_pde, index=species, columns=[f"cell_{i}" for i in range(n)])
            edited = st.data_editor(df, use_container_width=True, key="ic_table_pde")
            init_pde[:, :] = np.array(edited, dtype=float)

    st.markdown("**Initial condition summary**")
    if ic_type == "SSA (Particles)":
        totals = init_ssa.sum(axis=1)
        st.write({species[i]: int(totals[i]) for i in range(len(species))})
    else:
        totals = init_pde.sum(axis=1)
        st.write({species[i]: float(totals[i]) for i in range(len(species))})

# -----------------------------
# Save preset button (after arrays exist)
# -----------------------------
with st.sidebar:
    if st.button("Save preset to disk"):
        preset = preset_from_ui(
            name=st.session_state.get("preset_name", "my_preset"),
            species=species,
            L=L,
            K=K,
            pde_mult=pde_mult,
            boundary=boundary,
            diffusion=diff_values,
            conv_threshold=int(st.session_state.get("conv_threshold", 4)),
            conv_rate=float(st.session_state.get("conv_rate", 1.0)),
            rates=rates,
            pde_code=pde_code,
            ssa_raw=ssa_raw,
            total_time=float(st.session_state.get("total_time", 30.0)),
            dt=float(st.session_state.get("dt", 0.006)),
            repeats=int(st.session_state.get("repeats", 10)),
            parallel=bool(st.session_state.get("parallel", True)),
            ic_type=ic_type,
            ic_mode=ic_mode,
            init_ssa=init_ssa,
            init_pde=init_pde,
        )

        filename = st.session_state.get("preset_filename", f"{preset['name']}.json")
        if not filename.endswith(".json"):
            filename += ".json"
        out_path = PRESET_DIR / filename

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(preset, f, indent=2)
            st.success(f"Saved preset: {out_path}")
        except Exception as e:
            st.error(f"Failed to save preset: {e}")

# -----------------------------
# Execution with Stop button
# -----------------------------
st.write("")

# Create columns for Run and Stop buttons
col1, col2 = st.columns(2)

with col1:
    run_button = st.button(
        "🚀 RUN SIMULATION", 
        disabled=st.session_state.simulation_running,
        key="run_button"
    )

with col2:
    stop_button = st.button(
        "🛑 STOP SIMULATION", 
        disabled=not st.session_state.simulation_running,
        on_click=stop_simulation,
        key="stop_button"
    )

# Add custom CSS for button styling
st.markdown("""
    <style>
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #dc3545 !important;
        color: white !important;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #007bff !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

if run_button and not st.session_state.simulation_running:
    try:
        # Reset cancellation flag
        st.session_state.cancel_simulation = False
        st.session_state.simulation_running = True
        
        save_path = Path(os.path.expanduser(st.session_state["save_dir_str"])).resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        m = HybridModel(species=species)
        m.domain(L=L, K=K, pde_multiple=pde_mult, boundary=boundary)
        m.diffusion(**diff_values)

        # FIXED: use UI values (not hardcoded)
        m.conversion(
            threshold=int(st.session_state.get("conv_threshold", 4)),
            rate=float(st.session_state.get("conv_rate", 1.0)),
        )

        # PDE logic (eval)
        pde_func = eval(
            f"lambda {', '.join(species)}, r: ({pde_code})",
            {"np": np},
            {},
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
            
            # Create a progress bar
            progress_bar = st.progress(0, text="Starting simulation...")
            st.session_state.progress_bar = progress_bar
            
            # Create progress callback
            progress_callback = create_progress_callback()
            
            try:
                res = m.run_repeats(
                    init_ssa,
                    init_pde,
                    time=float(st.session_state.get("total_time", total_time)),
                    dt=float(st.session_state.get("dt", dt)),
                    repeats=int(st.session_state.get("repeats", repeats)),
                    seed=1,
                    parallel=bool(st.session_state.get("parallel", parallel)),
                    n_jobs=-1,
                    progress=True,
                    # Note: You may need to modify run_repeats to accept a callback
                    # If the HybridModel doesn't support callbacks, we need a different approach
                )
                
                if st.session_state.cancel_simulation:
                    status.update(label="Simulation Cancelled", state="error")
                    st.warning("Simulation was cancelled by the user.")
                else:
                    status.update(label="Simulation Complete!", state="complete")
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_file = save_path / f"{st.session_state['run_name']}_{timestamp}.npz"

                    meta = m.metadata()
                    meta.update(
                        {
                            "total_time": float(st.session_state.get("total_time", total_time)),
                            "dt": float(st.session_state.get("dt", dt)),
                            "repeats": int(st.session_state.get("repeats", repeats)),
                            "ic_mode": ic_mode,
                            "ic_type": ic_type,
                            "preset_file": st.session_state.get("chosen_preset"),
                        }
                    )

                    save_npz(res, str(out_file), meta=meta)

                    st.balloons()
                    st.success(f"**Success!** Data written to: `{out_file}`")
                    st.info("💡 Copy the command below to visualize your results:")
                    st.code(f"python view_npz.py \"{out_file}\" --plot --anim")
                    
            except KeyboardInterrupt as e:
                status.update(label="Simulation Cancelled", state="error")
                st.warning(f"Simulation was interrupted: {e}")
            except Exception as e:
                status.update(label="Simulation Failed", state="error")
                st.error(f"Simulation failed: {e}")
                st.exception(e)
            finally:
                # Clean up progress bar
                if 'progress_bar' in st.session_state:
                    del st.session_state.progress_bar
                # Reset running state
                st.session_state.simulation_running = False

    except Exception as e:
        st.error(f"Critical Failure: {e}")
        st.exception(e)
        st.session_state.simulation_running = False