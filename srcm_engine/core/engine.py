from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import os

import numpy as np
from srcm_engine.domain import Domain
from srcm_engine.conversion import ConversionParams, combined_mass
from srcm_engine.reactions import HybridReactionSystem
from srcm_engine.state import HybridState
from srcm_engine.pde import laplacian_1d, rk4_step
from srcm_engine.diffusion import SSADiffusion
from srcm_engine.core.gillespie_loop import gillespie_draw
from srcm_engine.conversion.regime_utils import sufficient_pde_concentration_mask
from contextlib import contextmanager
from srcm_engine.results import SimulationResults

PDETermsFn = Callable[[np.ndarray, Dict[str, float]], np.ndarray]
# signature: pde_terms(C, rates) -> reaction-only dC/dt, shape (n_species, Npde)



@dataclass
class SRCMEngine:
    reactions: HybridReactionSystem
    pde_reaction_terms: PDETermsFn
    diffusion_rates: Dict[str, float]         # macroscopic diffusion D per species
    domain: Domain
    conversion: ConversionParams
    reaction_rates: Dict[str, float]          # dict passed into propensity lambdas + pde_terms

    def __post_init__(self):
        # basic validations
        if set(self.reactions.species) != set(self.diffusion_rates.keys()):
            raise ValueError("diffusion_rates keys must match reactions.species")
        if self.domain.K <= 1 and self.domain.boundary == "zero-flux":
            raise ValueError("zero-flux with K<=1 is degenerate (needs at least 2 compartments)")

        # Precompute PDE Laplacian once
        self._L = laplacian_1d(self.domain)

        # Precompute SSA jump rates per species: D / h^2
        jump_rates = {}
        for sp in self.reactions.species:
            D = float(self.diffusion_rates[sp])
            if D < 0:
                raise ValueError("diffusion rates must be >= 0")
            jump_rates[sp] = D / (self.domain.h ** 2)

        self._diffusion = SSADiffusion(species=self.reactions.species, jump_rates=jump_rates)

        # Useful mappings
        self._sp_to_idx = {sp: i for i, sp in enumerate(self.reactions.species)}

        # Precompute more values
        self._gamma = float(self.conversion.rate)
        self._h_squared = self.domain.h ** 2
        self._dx_squared = self.domain.dx ** 2
        
        # Precompute diffusion coefficients array
        self._D_array = np.array([self.diffusion_rates[sp] 
                                for sp in self.reactions.species])
        
        # Precompute jump rates array
        self._jump_rates_array = self._D_array / self._h_squared

    # ------------------------------------------------------------------
    # PDE RHS: diffusion + reaction terms
    # ------------------------------------------------------------------
    # def pde_rhs(self, C: np.ndarray, t: float) -> np.ndarray:
    #     """
    #     C shape: (n_species, Npde)
    #     returns dC/dt shape: (n_species, Npde)
    #     """
    #     n_species, Npde = C.shape
    #     out = np.zeros_like(C, dtype=float)

    #     # Diffusion part: (D/dx^2) * L @ C_s
    #     dx2 = self.domain.dx ** 2
    #     for sp, s_idx in self._sp_to_idx.items():
    #         D = float(self.diffusion_rates[sp])
    #         out[s_idx, :] = (D / dx2) * (self._L @ C[s_idx, :])

    #     # Reaction part (macroscopic)
    #     out += self.pde_reaction_terms(C, self.reaction_rates)
    #     return out

    def pde_rhs(self, C: np.ndarray, t: float) -> np.ndarray:
        """Vectorized PDE RHS"""
        n_species, Npde = C.shape
        out = np.zeros_like(C, dtype=float)
        
        # Vectorized diffusion using einsum
        # (D/dx^2) * L @ C_s for each species
        diff_coeff = self._D_array[:, np.newaxis] / self._dx_squared
        # Use @ for matrix multiplication (fast in numpy)
        diffusion_terms = diff_coeff * (self._L @ C.T).T
        
        out = diffusion_terms + self.pde_reaction_terms(C, self.reaction_rates)
        return out

    # ------------------------------------------------------------------
    # Propensity building (THIS STEP)
    # ------------------------------------------------------------------
#     def build_propensity_vector(
#     self,
#     state: HybridState,
#     pde_mass: np.ndarray,
#     exceeds_mask: np.ndarray,
#     sufficient_mask: np.ndarray,
#     out: Optional[np.ndarray] = None,
# ) -> np.ndarray:
#         """
#         Build propensity vector for:
#         - diffusion blocks: n_species * K
#         - CD blocks:        n_species * K
#         - DC blocks:        n_species * K
#         - hybrid blocks:    n_hybrid * K

#         IMPORTANT (SRCM admissibility):
#         Any event that REMOVES continuous mass (CD or hybrid with C_<sp> < 0)
#         is only allowed if sufficient_mask[sp, i] == 1 in that compartment.
#         """
#         state.assert_consistent(self.domain)

#         n_species = self.reactions.n_species
#         K = self.domain.K
#         n_hybrid = len(self.reactions.hybrid_reactions)
#         # At the start of build_propensity_vector, add:
#         sp_to_idx = self._sp_to_idx  # Local reference is faster than self. lookups



#         n_blocks = 3 * n_species + n_hybrid
#         n_total = n_blocks * K

#         if out is None:
#             a = np.zeros(n_total, dtype=float)
#         else:
#             if out.shape != (n_total,):
#                 raise ValueError(f"out must have shape {(n_total,)}")
#             a = out
#             a.fill(0.0)

#         gamma = float(self.conversion.rate)

#         # ------------------------------------------------------------------
#         # 0..(n_species-1): SSA diffusion
#         # ------------------------------------------------------------------
#         self._diffusion.fill_propensities(a, state.ssa, K)

#         # ------------------------------------------------------------------
#         # n_species..(2*n_species-1): CD (PDE -> SSA)
#         # Allowed iff:
#         #   exceeds_mask == 0  AND  sufficient_mask == 1
#         # propensity = gamma * (PDE mass in compartment)
#         #
#         # NOTE: We do NOT clamp PDE itself. We only forbid negative "available mass"
#         # from creating negative propensities.
#         # ------------------------------------------------------------------
#         for s_idx in range(n_species):
#             block = n_species + s_idx
#             start = block * K
#             end = start + K

#             cd_allowed = (exceeds_mask[s_idx, :] == 0) & (sufficient_mask[s_idx, :] == 1)

#             # available PDE mass cannot be negative (if PDE has gone negative, CD must not fire)
#             available_mass = np.maximum(pde_mass[s_idx, :], 0.0)

#             a[start:end] = gamma * available_mass * cd_allowed

#         # ------------------------------------------------------------------
#         # (2*n_species)..(3*n_species-1): DC (SSA -> PDE)
#         # Allowed iff exceeds_mask == 1
#         # propensity = gamma * SSA count
#         # ------------------------------------------------------------------
#         for s_idx in range(n_species):
#             block = 2 * n_species + s_idx
#             start = block * K
#             end = start + K

#             dc_allowed = (exceeds_mask[s_idx, :] == 1)
#             a[start:end] = gamma * state.ssa[s_idx, :] * dc_allowed

#         # ------------------------------------------------------------------
#         # Hybrid reactions
#         #
#         # SRCM admissibility:
#         #   If hr consumes continuous mass (any C_<sp> delta < 0), only allow it
#         #   when sufficient_mask[sp, i] == 1 for all such species sp.
#         #
#         # Also: if PDE mass is negative in that compartment for a consumed species,
#         # the reaction should be disallowed (available_mass == 0 idea).
#         # ------------------------------------------------------------------
#         base_block = 3 * n_species
#         h = float(self.domain.h)
#         species = self.reactions.species

#         for rxn_idx, hr in enumerate(self.reactions.hybrid_reactions):
#             block = base_block + rxn_idx
#             start = block * K

#             # consumes_C = getattr(hr, "consumes_continuous", False)
#             # consumed_species = getattr(hr, "consumed_species", ())

#             # Infer continuous consumption from state_change (source of truth), added 7th jan 2026. (with new reaction system)
#             consumed_species = [
#                 key.split("_", 1)[1]
#                 for key, delta in hr.state_change.items()
#                 if key.startswith("C_") and delta < 0
#             ]
#             consumes_C = (len(consumed_species) > 0)

    

#             for i in range(K):

#                 # Gate reactions that remove PDE particle mass
#                 if consumes_C:
#                     ok = True
#                     for sp in consumed_species:
#                         s_idx = self._sp_to_idx[sp]

#                         # must pass the "every fine cell >= 1/h" test
#                         if sufficient_mask[s_idx, i] == 0:
#                             ok = False
#                             break

#                         # additionally, negative integrated mass is not "available"
#                         if pde_mass[s_idx, i] < 0.0:
#                             ok = False
#                             break

#                     if not ok:
#                         a[start + i] = 0.0
#                         continue

#                 D_local = {sp: int(state.ssa[self._sp_to_idx[sp], i]) for sp in species}
#                 C_local = {sp: float(pde_mass[self._sp_to_idx[sp], i]) for sp in species}

#                 val = float(hr.propensity(D_local, C_local, self.reaction_rates, h))

#                 # A propensity must never be negative; if it is, treat as disallowed.
#                 # This is NOT clamping PDE; it is enforcing the mathematical constraint a>=0.
#                 a[start + i] = val if val > 0.0 else 0.0

#         return a

    def build_propensity_vector(
        self,
        state: HybridState,
        pde_mass: np.ndarray,
        exceeds_mask: np.ndarray,
        sufficient_mask: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        state.assert_consistent(self.domain)
        
        n_species = self.reactions.n_species
        K = self.domain.K
        n_hybrid = len(self.reactions.hybrid_reactions)
        n_total = (3 * n_species + n_hybrid) * K
        
        if out is None:
            a = np.zeros(n_total, dtype=float)
        else:
            if out.shape != (n_total,):
                raise ValueError(f"out must have shape {(n_total,)}")
            a = out
            a.fill(0.0)
        
        gamma = float(self.conversion.rate)
        
        # 1. Diffusion - already vectorized
        
        # 2. CD blocks - vectorize across compartments
        cd_start = n_species * K
        cd_end = 2 * n_species * K
        
        # Pre-compute available mass and masks
        available_mass = np.maximum(pde_mass, 0.0)
        cd_allowed = (~exceeds_mask.astype(bool)) & (sufficient_mask.astype(bool))
        
        # Fill CD blocks efficiently
        for s_idx in range(n_species):
            start = cd_start + s_idx * K
            end = start + K
            a[start:end] = gamma * available_mass[s_idx] * cd_allowed[s_idx]
        
        # 3. DC blocks - vectorize
        dc_start = 2 * n_species * K
        dc_end = 3 * n_species * K
        
        dc_allowed = exceeds_mask.astype(bool)
        for s_idx in range(n_species):
            start = dc_start + s_idx * K
            end = start + K
            a[start:end] = gamma * state.ssa[s_idx] * dc_allowed[s_idx]
        
        # 4. Hybrid reactions - try to vectorize if possible
        base_block = 3 * n_species
        h = float(self.domain.h)
        species = self.reactions.species
        
        # Pre-compute local D and C arrays for all compartments
        # This can be expensive but might be worth it
        all_D_local = np.stack([state.ssa[self._sp_to_idx[sp]] for sp in species], axis=0)
        all_C_local = np.stack([pde_mass[self._sp_to_idx[sp]] for sp in species], axis=0)
        
        for rxn_idx, hr in enumerate(self.reactions.hybrid_reactions):
            block = base_block + rxn_idx
            start = block * K
            
            # Identify continuous consumers
            consumed_species = [
                key.split("_", 1)[1]
                for key, delta in hr.state_change.items()
                if key.startswith("C_") and delta < 0
            ]
            consumes_C = (len(consumed_species) > 0)
            
            if consumes_C:
                # Check all consumed species have sufficient mask
                mask_valid = np.ones(K, dtype=bool)
                for sp in consumed_species:
                    s_idx = self._sp_to_idx[sp]
                    mask_valid &= (sufficient_mask[s_idx] == 1)
                    mask_valid &= (pde_mass[s_idx] >= 0.0)
            else:
                mask_valid = np.ones(K, dtype=bool)
            
            # Vectorized propensity calculation if hr.propensity supports it
            # If not, keep loop but use pre-computed arrays
            for i in range(K):
                if not mask_valid[i]:
                    a[start + i] = 0.0
                    continue
                    
                # Use pre-computed arrays
                D_local = {sp: all_D_local[j, i] for j, sp in enumerate(species)}
                C_local = {sp: all_C_local[j, i] for j, sp in enumerate(species)}
                
                val = float(hr.propensity(D_local, C_local, self.reaction_rates, h))
                a[start + i] = max(val, 0.0)
        
        return a


    def apply_event(
        self,
        idx: int,
        state: HybridState,
        rng: np.random.Generator,
        pde_mass: np.ndarray,
    ) -> None:
        """
        Apply the event at flat propensity index `idx` to `state` in-place.

        Block layout (per compartment, length K):
          diffusion blocks:  [0 .. n_species-1]
          CD blocks:         [n_species .. 2*n_species-1]
          DC blocks:         [2*n_species .. 3*n_species-1]
          hybrid blocks:     [3*n_species .. 3*n_species+n_hybrid-1]

        Notes
        -----
        - For diffusion we use a third uniform u3 to pick direction.
        - For CD/DC we add/remove exactly 1 particle, and adjust PDE slice by ±(1/h).
        - For hybrid reactions we use hr.state_change:
            'D_<sp>' updates SSA
            'C_<sp>' updates PDE slice by ±1 particle mass
        """
        state.assert_consistent(self.domain)

        n_species = self.reactions.n_species
        K = self.domain.K
        n_hybrid = len(self.reactions.hybrid_reactions)

        if idx < 0:
            return  # no event
        if idx >= (3 * n_species + n_hybrid) * K:
            raise ValueError("idx out of range for propensity vector")

        block = idx // K
        comp = idx % K

        # -------------------- Diffusion --------------------
        if block < n_species:
            species_idx = block
            u3 = float(rng.random())
            self._diffusion.apply_move(state.ssa, self.domain, species_idx, comp, u=u3)
            return

        # -------------------- CD: PDE -> SSA --------------------
        if block < 2 * n_species:
            species_idx = block - n_species

            # Add one discrete particle
            state.add_discrete(species_idx, comp, +1)
            # Remove one particle mass from PDE slice: - (1/h)
            state.add_continuous_particle_mass(self.domain, species_idx, comp, -1)
            return

        # -------------------- DC: SSA -> PDE --------------------
        if block < 3 * n_species:
            species_idx = block - 2 * n_species
         
            state.add_discrete(species_idx, comp, -1)
            # Add one particle mass to PDE slice: + (1/h)
            state.add_continuous_particle_mass(self.domain, species_idx, comp, +1)
            return

        # -------------------- Hybrid reactions --------------------
        rxn_idx = block - 3 * n_species
        hr = self.reactions.hybrid_reactions[rxn_idx]

        # Apply each delta in state_change
        # Convention:
        #   'D_U' means SSA species U change
        #   'C_U' means PDE mass in that compartment changes by ±1 particle
        for key, delta in hr.state_change.items():
            prefix, sp = key.split("_", 1)
            species_idx = self._sp_to_idx[sp]

            if prefix == "D":
                state.add_discrete(species_idx, comp, int(delta))
            elif prefix == "C":
                # delta is in particles
                state.add_continuous_particle_mass(self.domain, species_idx, comp, int(delta))
            else:
                raise ValueError(f"Unknown state_change key prefix '{prefix}'")

    
    
    def _simulate_interval(
        self,
        state: HybridState,
        t0: float,
        t1: float,
        rng: np.random.Generator,
        propensity: np.ndarray,
        cumulative: np.ndarray,
    ) -> None:
        """
        Simulate SSA/conversion/hybrid events in [t0, t1) holding PDE fixed.
        Updates `state` in-place.
        """
        n_species = self.reactions.n_species
        K = self.domain.K
        n_hybrid = len(self.reactions.hybrid_reactions)

        while t0 < t1:
            # masses + masks
            comb, pde_mass = combined_mass(state.ssa, state.pde, self.domain.pde_multiple, self.domain.dx)
            exceeds = self.conversion.exceeds_threshold_mask(comb)
            sufficient = sufficient_pde_concentration_mask(state.pde, self.domain.pde_multiple, self.domain.h)

            # propensities
            self.build_propensity_vector(state, pde_mass, exceeds, sufficient, out=propensity)

                        # DEBUG: decode first negative propensity
            if np.any(propensity < 0):
                j = int(np.where(propensity < 0)[0][0])
                block = j // K
                comp = j % K

                n_blocks = 3 * n_species + n_hybrid

                print("\n" + "=" * 80)
                print("NEGATIVE PROPENSITY")
                print(f"t0={t0}, t1={t1}")
                print(f"flat idx={j}, value={float(propensity[j])}")
                print(f"block={block}/{n_blocks-1}, compartment={comp}/{K-1}")

                if block < n_species:
                    print("Section: diffusion")
                    print("species:", self.reactions.species[block])

                elif block < 2 * n_species:
                    print("Section: CD (PDE -> SSA)")
                    print("species:", self.reactions.species[block - n_species])
                    print("pde_mass local:", pde_mass[:, comp])
                    print("sufficient local:", sufficient[:, comp])
                    print("exceeds local:", exceeds[:, comp])

                elif block < 3 * n_species:
                    print("Section: DC (SSA -> PDE)")
                    print("species:", self.reactions.species[block - 2 * n_species])
                    print("ssa local:", state.ssa[:, comp])
                    print("exceeds local:", exceeds[:, comp])

                else:
                    rxn_idx = block - 3 * n_species
                    hr = self.reactions.hybrid_reactions[rxn_idx]
                    print("Section: hybrid reaction")
                    print("rxn_idx:", rxn_idx)
                    print("label:", hr.label)
                    print("state_change:", hr.state_change)
                    if getattr(hr, "description", None):
                        print("desc:", hr.description)

                    # local D/C passed into lambda:
                    species = self.reactions.species
                    D_local = {sp: int(state.ssa[self._sp_to_idx[sp], comp]) for sp in species}
                    C_local = {sp: float(pde_mass[self._sp_to_idx[sp], comp]) for sp in species}
                    print("D_local:", D_local)
                    print("C_local:", C_local)

                print("SSA local:", state.ssa[:, comp])
                # also inspect PDE slice min in that compartment
                s = comp * self.domain.pde_multiple
                e = s + self.domain.pde_multiple
                print("PDE slice mins:", np.min(state.pde[:, s:e], axis=1))
                print("=" * 80 + "\n")

                raise ValueError("Negative propensity detected (decoded above)")

            # Gillespie draw
            tau, idx = gillespie_draw(propensity, rng, cumulative=cumulative)

            if not np.isfinite(tau):
                # no stochastic events possible
                return

            if t0 + tau < t1:
                t0 += float(tau)
                self.apply_event(idx, state, rng, pde_mass=pde_mass)
            else:
                return

    def run(
        self,
        initial_ssa: np.ndarray,
        initial_pde: Optional[np.ndarray],
        time: float,
        dt: float,
        seed: int = 0,
    ):
        """
        Run one simulation and return full time series results.
        """
        from srcm_engine.results import SimulationResults

        if time <= 0:
            raise ValueError("time must be > 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        rng = np.random.default_rng(seed)

        n_species = self.reactions.n_species
        K = self.domain.K
        Npde = self.domain.n_pde

        if initial_pde is None:
            initial_pde = np.zeros((n_species, Npde), dtype=float)

        state = HybridState(
            ssa=initial_ssa.copy().astype(int),
            pde=initial_pde.copy().astype(float),
        )
        state.assert_consistent(self.domain)

        # time grid (include t=0)
        n_steps = int(np.floor(time / dt)) + 1
        tvec = np.arange(n_steps, dtype=float) * dt

        # output arrays
        ssa_out = np.zeros((n_species, K, n_steps), dtype=int)
        pde_out = np.zeros((n_species, Npde, n_steps), dtype=float)

        # record initial
        ssa_out[:, :, 0] = state.ssa
        pde_out[:, :, 0] = state.pde

        # allocate propensity + cumulative once
        n_hybrid = len(self.reactions.hybrid_reactions)
        n_blocks = 3 * n_species + n_hybrid
        propensity = np.zeros(n_blocks * K, dtype=float)
        cumulative = np.empty_like(propensity)

        # main loop over PDE ticks
        for n in range(n_steps - 1):
            t0 = tvec[n]
            t1 = tvec[n + 1]

            # stochastic evolution on [t0, t1) with PDE frozen
            self._simulate_interval(state, t0, t1, rng, propensity, cumulative)

            # deterministic PDE step to t1
            state.pde = rk4_step(state.pde, t=t0, dt=dt, rhs=self.pde_rhs)

            # record at t1
            ssa_out[:, :, n + 1] = state.ssa
            pde_out[:, :, n + 1] = state.pde

        return SimulationResults(
            time=tvec,
            ssa=ssa_out,
            pde=pde_out,
            domain=self.domain,
            species=self.reactions.species,
        )

    # def run_repeats(
    #         self,
    #         initial_ssa: np.ndarray,
    #         initial_pde: Optional[np.ndarray],
    #         time: float,
    #         dt: float,
    #         repeats: int,
    #         seed: int = 0,
    #         progress: bool = True,
    #     ):
    #     """
    #     Run multiple independent simulations and return mean SSA/PDE time series.

    #     Parameters
    #     ----------
    #     progress : bool
    #         If True, show a tqdm progress bar.
    #     """
    #     from srcm_engine.results import SimulationResults

    #     if repeats <= 0:
    #         raise ValueError("repeats must be > 0")

    #     # Import tqdm only if needed (keeps core dependency optional-ish)
    #     if progress:
    #         try:
    #             from tqdm.auto import tqdm
    #         except ImportError:
    #             tqdm = None
    #             progress = False

    #     # Run first simulation to get shapes + time vector
    #     res0 = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed)

    #     ssa_sum = res0.ssa.astype(float)
    #     pde_sum = res0.pde.astype(float)

    #     iterator = range(1, repeats)
    #     if progress:
    #         iterator = tqdm(
    #             iterator,
    #             total=repeats - 1,
    #             desc="SRCM repeats",
    #             unit="run",
    #             dynamic_ncols=True,
    #         )

    #     # Remaining repeats
    #     for r in iterator:
    #         res_r = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
    #         ssa_sum += res_r.ssa
    #         pde_sum += res_r.pde

    #     ssa_mean = ssa_sum / repeats
    #     pde_mean = pde_sum / repeats

    #     return SimulationResults(
    #         time=res0.time,
    #         ssa=ssa_mean,
    #         pde=pde_mean,
    #         domain=self.domain,
    #         species=self.reactions.species,
    #     )


    def run_repeats(
    self,
    initial_ssa: np.ndarray,
    initial_pde: Optional[np.ndarray],
    time: float,
    dt: float,
    repeats: int,
    seed: int = 0,
    *,
    parallel: bool = False,
    n_jobs: int = -1,
    prefer: str = "processes",
    progress: bool = True,
):
        """
        Run multiple independent simulations and return mean SSA/PDE time series.

        parallel=True uses joblib to spread repeats across CPU cores.
        n_jobs=-1 uses all available cores (often best to set to os.cpu_count()-1).
        """
   

        if repeats <= 0:
            raise ValueError("repeats must be > 0")

        # First run locally for shapes/time vector
        res0 = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed)
        ssa_sum = res0.ssa.astype(float)
        pde_sum = res0.pde.astype(float)

        if repeats == 1:
            return SimulationResults(
                time=res0.time,
                ssa=ssa_sum,
                pde=pde_sum,
                domain=self.domain,
                species=self.reactions.species,
            )

        # -----------------------
        # Serial repeats
        # -----------------------
        if not parallel:
            iterator = range(1, repeats)

            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(
                        iterator,
                        total=repeats - 1,
                        desc="SRCM repeats",
                        unit="run",
                        dynamic_ncols=True,
                    )
                except ImportError:
                    pass  # no tqdm, fall back to plain range

            for r in iterator:
                res_r = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                ssa_sum += res_r.ssa
                pde_sum += res_r.pde

        # -----------------------
        # Parallel repeats (joblib)
        # -----------------------
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "Parallel repeats requires joblib. Install with: pip install joblib"
                ) from e

            # tqdm optional
            if progress:
                try:
                    from tqdm.auto import tqdm
                except ImportError:
                    tqdm = None
                    progress = False

            def one(r: int):
                res = self.run(initial_ssa, initial_pde, time=time, dt=dt, seed=seed + r)
                return res.ssa.astype(float), res.pde.astype(float)

            tasks = (delayed(one)(r) for r in range(1, repeats))

            # Stream results back as they complete (prevents huge RAM usage)
            par = Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")

            if n_jobs == -1:
                n_workers = os.cpu_count() or 1
            else:
                n_workers = n_jobs

            print(f"Running SRCM repeats in parallel on {n_workers} core(s)")

            # Stream results back as they complete
            par = Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")

            results_iter = par(tasks)
            if progress and tqdm is not None:
                results_iter = tqdm(
                    results_iter,
                    total=repeats - 1,
                    desc="SRCM repeats",
                    unit="run",
                    dynamic_ncols=True,
                )

            for ssa_r, pde_r in results_iter:
                ssa_sum += ssa_r
                pde_sum += pde_r

        ssa_mean = ssa_sum / repeats
        pde_mean = pde_sum / repeats

        return SimulationResults(
            time=res0.time,
            ssa=ssa_mean,
            pde=pde_mean,
            domain=self.domain,
            species=self.reactions.species,
        )
