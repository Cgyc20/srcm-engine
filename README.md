# SRCM Engine

**SRCM Engine** is a Python package for simulating **spatial reaction–diffusion systems**
using a **hybrid stochastic–continuum (SSA–PDE)** framework. This is called the **Spatial regime conversion method**, a method co-authored by Myself (Charles Cameron), Professor Kit Yates and Dr Cameron Smith. This has been the major component of my PhD research at the University of Bath.

This is a Hybrid framework which combined PDE's with stochastic compartment method. This allows the system to switch between either framework dynamically across the spatial domain depending on local particle counts. Please refer to the paper **Cameron, C. G., Smith, C. A., & Yates, C. A. (2025). The Spatial Regime Conversion Method. Mathematics, 13(21), 3406. https://doi.org/10.3390/math13213406Z** for more reading.

It is designed for systems in which:
- particle numbers may be low in some regions (requiring stochastic simulation),
- but high in others (where a continuum PDE description is more appropriate),
- and where spatial diffusion is essential.

SRCM Engine automatically couples **discrete stochastic reactions** with **continuous diffusion**
on a spatial domain, without requiring users to manually write hybrid reaction channels.

![Schematic](figures/schematic.png)
---

## 1. Motivation

Many spatial reaction systems sit awkwardly between two classical modelling approaches:

### Pure SSA
Accurate at low copy numbers, but expensive and noisy at large scales.

### Pure PDE models
Efficient at large scales, but invalid when particle numbers are small.

Hybrid SSA–PDE methods address this by treating some mass discretely and some continuously.
However, existing approaches often require:
- hand-written hybrid propensities,
- system-specific derivations,
- or deep knowledge of the numerical method.

**SRCM Engine removes this barrier** by allowing users to:
1. Write reactions at the *macroscopic* level
2. Define PDE reaction terms in a natural, mathematical form  
3. Automatically obtain a consistent hybrid SSA–PDE simulation  

The goal is to make **hybrid spatial modelling accessible, reproducible, and safe**, without sacrificing mathematical correctness or performance.

---

## 2. Core Ideas

SRCM Engine is built around the following principles:

### Hybrid Representation
- Each species exists in **both discrete (SSA)** and **continuous (PDE)** forms.
- A **conversion mechanism** dynamically moves mass between the two representations
  based on local particle numbers.

### Spatial Structure
- Space is discretised into compartments for SSA.
- Each compartment is internally resolved by a finer PDE grid.
- Diffusion is handled consistently across both representations.

### Automatic Hybridisation
- Users specify **macroscopic reactions** (e.g. `A + B → C`).
- SRCM Engine automatically decomposes these into the correct hybrid reaction channels:
  - discrete–discrete
  - discrete–continuous
  - continuous–discrete
- This ensures correctness without user intervention.

---

## 3. Package Structure

At a high level, SRCM Engine consists of:

- `HybridModel` — a **user-facing API** for building models
- `SRCMEngine` — the core simulation engine
- `HybridReactionSystem` — reaction bookkeeping and decomposition
- `Domain` — spatial domain definition
- `ConversionParams` — rules for SSA ↔ PDE conversion
- `SimulationResults` — structured output and analysis helpers

Most users will only interact with **`HybridModel`**.

---

## 4. Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-org/srcm-engine.git
cd srcm-engine
pip install -e .
````

Requirements:

* Python ≥ 3.9
* NumPy
* Matplotlib
* Joblib (for parallel execution)

---

## 5. Quick Start Example

Below is a complete example of a **two-species switching system**:

$$
A \rightleftharpoons B
$$

with spatial diffusion and hybrid dynamics.

### 5.1 Build the Model

```python
import numpy as np
from srcm_engine.core import HybridModel

m = HybridModel(species=["A", "B"])

m.domain(
    L=10.0,
    K=40,
    pde_multiple=8,
    boundary="zero-flux",
)

m.diffusion(A=0.1, B=0.1)

m.conversion(
    threshold=4,
    rate=1.0,
)

m.reaction_terms(
    lambda A, B, r: (
        r["beta"] * B - r["alpha"] * A,
        r["alpha"] * A - r["beta"] * B,
    )
)

m.add_reaction({"A": 1}, {"B": 1}, rate_name="alpha")
m.add_reaction({"B": 1}, {"A": 1}, rate_name="beta")

m.build(rates={"alpha": 0.01, "beta": 0.01})
```

---

### 5.2 Initial Conditions

Users provide **raw NumPy arrays**:

```python
K = m.domain_obj.K
n_pde = m.domain_obj.n_pde

init_ssa = np.zeros((2, K), dtype=int)
init_pde = np.zeros((2, n_pde), dtype=float)

init_ssa[0, :K//4] = 10
init_ssa[1, 3*K//4:] = 10
```

---

### 5.3 Run the Simulation

```python
res = m.run_repeats(
    init_ssa,
    init_pde,
    time=30.0,
    dt=0.006,
    repeats=100,
    parallel=True,
)
```

---

## 6. Parallel Execution

SRCM Engine supports **parallel execution** of ensemble simulations.

When `parallel=True`, repeated simulations are distributed across CPU cores using
multi-processing.

```python
res = m.run_repeats(
    init_ssa,
    init_pde,
    time=30.0,
    dt=0.006,
    repeats=100,
    parallel=True,
    n_jobs=-1,
)
```

### Notes

* Parallelism is optional

## 7. Saving Results and Metadata

Simulation results can be saved in a portable `.npz` format:

```python
from srcm_engine.results.io import save_npz

meta = m.metadata()
meta.update({
    "total_time": 30.0,
    "dt": 0.006,
    "repeats": 100,
})

save_npz(res, "ab_switch_mean.npz", meta=meta)
```

Metadata includes:

* domain parameters
* diffusion coefficients
* conversion settings
* reaction rates
* hybrid reaction labels

This ensures simulations are **fully reproducible**.

---

## 8. Visualisation

### 8.1 Inline Animation (Jupyter)

```python
from IPython.display import HTML, display
from srcm_engine.animation_util import AnimationConfig, animate_results

cfg = AnimationConfig(
    stride=20,
    interval_ms=25,
    threshold_particles=meta["threshold_particles"],
    title="Hybrid Simulation: A ⇌ B",
)

anim = animate_results(res, cfg=cfg, return_animation=True)
display(HTML(anim.to_jshtml()))
```

---

### 8.2 Time Series Plots

```python
from srcm_engine.animation_util import plot_mass_time_series
plot_mass_time_series(res)
```

---

## 9. Reaction System Introspection

You can inspect how macroscopic reactions were decomposed:

```python
m.describe_reactions()
```

This prints:

* macroscopic reactions
* corresponding hybrid reaction channels
* propensities and state changes

This is useful for:

* debugging
* teaching
* verification

---

## 10. When Should I Use SRCM Engine?

SRCM Engine is well suited for:

* pattern formation (e.g. Turing systems)
* ecological or biochemical spatial models
* systems with sharp gradients in particle number
* models requiring both stochasticity and efficiency

---

## 11. Limitations

Current limitations include:

* reactions of order > 2 are not supported
* 1D spatial domains only
* explicit time stepping for PDEs

These are active areas of development.

---

## 12. Citation and Attribution

If you use SRCM Engine in academic work, please cite:

> Cameron, C. (2026).
> *SRCM Engine: A hybrid stochastic–continuum framework for spatial reaction–diffusion systems.*

> *Cameron, C. G., Smith, C. A., & Yates, C. A. (2025). The Spatial Regime Conversion Method. Mathematics, 13(21), 3406. https://doi.org/10.3390/math13213406*
---

## 13. Contributing

Contributions are welcome.

Suggested areas:

* higher-order reactions
* adaptive domains
* GPU acceleration
* improved visualisation

Please open an issue or pull request.

---


