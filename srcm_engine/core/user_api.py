from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ..domain import Domain
from ..conversion import ConversionParams
from ..reactions import HybridReactionSystem
from .engine import SRCMEngine


# User-facing PDE RHS:
#   lambda A, B, r: (dA, dB)
UserRHSFn = Callable[..., Union[Sequence[np.ndarray], np.ndarray]]


@dataclass
class HybridModel:
    """
    User-friendly wrapper around SRCMEngine.

    Users never touch:
      - HybridReactionSystem internals
      - propensity lambdas
      - SRCM bookkeeping

    They only specify:
      - species
      - domain / diffusion / conversion
      - macroscopic reactions
      - PDE reaction terms
      - numpy initial conditions
    """

    species: List[str]

    # configuration
    _domain: Optional[Domain] = None
    _conversion: Optional[ConversionParams] = None
    _diffusion: Optional[Dict[str, float]] = None

    # reactions
    _reactions: Optional[HybridReactionSystem] = None
    _rhs_user: Optional[UserRHSFn] = None

    # built engine
    _engine: Optional[SRCMEngine] = None
    _rates: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __post_init__(self):
        if not self.species:
            raise ValueError("species must be non-empty")
        if len(set(self.species)) != len(self.species):
            raise ValueError("species must be unique")

        self._domain = None
        self._conversion = None
        self._diffusion_rates = None   # ✅ add this
        self._rates = None             # ✅ add this
        self._engine = None            # ✅ add this

        self._reactions = HybridReactionSystem(species=list(self.species))


    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def domain(
        self,
        *,
        L: float,
        K: int,
        pde_multiple: int = 4,
        boundary: str = "zero-flux",
    ) -> "HybridModel":
        self._domain = Domain(
            length=float(L),
            n_ssa=int(K),
            pde_multiple=int(pde_multiple),
            boundary=str(boundary),
        )
        return self

    def diffusion(self, **rates: float) -> "HybridModel":
        for sp in self.species:
            if sp not in rates:
                raise ValueError(f"Missing diffusion rate for species '{sp}'")
        self._diffusion = {sp: float(rates[sp]) for sp in self.species}
        return self

    def conversion(
        self,
        *,
        threshold: float | dict[str, float] | list[float] | tuple[float, ...],
        rate: float | dict[str, float] | list[float] | tuple[float, ...] = 1.0,
    ) -> "HybridModel":
        """Configure SSA<->PDE conversion.

        You can supply either:
          - scalar values (global threshold/rate for all species), OR
          - per-species values via a dict keyed by species name, OR
          - per-species sequences aligned with `self.species`.

        Examples
        --------
        m.conversion(threshold=25, rate=0.5)  # global
        m.conversion(threshold={"A": 10, "B": 50}, rate={"A": 2.0, "B": 0.2})
        m.conversion(threshold=[10, 50], rate=[2.0, 0.2])  # aligned with species order
        """

        def _to_per_species(x, name: str):
            # scalar -> keep scalar
            if isinstance(x, (int, float)):
                return float(x)
            # dict -> ordered array
            if isinstance(x, dict):
                missing = [sp for sp in self.species if sp not in x]
                if missing:
                    raise ValueError(f"Missing {name} for species: {missing}")
                return [float(x[sp]) for sp in self.species]
            # sequence -> must match n_species
            if isinstance(x, (list, tuple)):
                if len(x) != len(self.species):
                    raise ValueError(
                        f"{name} must have length {len(self.species)} (one per species)"
                    )
                return [float(v) for v in x]
            raise TypeError(
                f"Unsupported type for {name}: {type(x)}. Use a scalar, dict, list, or tuple."
            )

        thr_val = _to_per_species(threshold, "threshold")
        rate_val = _to_per_species(rate, "rate")

        self._conversion = ConversionParams(
            threshold=thr_val,
            rate=rate_val,
        )
        return self

    # ------------------------------------------------------------------
    # PDE reaction terms (user-friendly)
    # ------------------------------------------------------------------
    def reaction_terms(self, fn: UserRHSFn) -> "HybridModel":
        """
        Register PDE reaction terms.

        Example:
            m.reaction_terms(lambda A, B, r: (
                r["beta"]*B - r["alpha"]*A,
                r["alpha"]*A - r["beta"]*B,
            ))
        """
        self._rhs_user = fn
        return self

    # ------------------------------------------------------------------
    # macroscopic reactions
    # ------------------------------------------------------------------
    def add_reaction(
        self,
        reactants: Dict[str, int],
        products: Dict[str, int],
        *,
        rate_name: str,
        rate: Optional[float] = None,
    ) -> "HybridModel":
        if self._reactions is None:
            raise RuntimeError("Internal reaction system not initialised")

        numeric_rate = 0.0 if rate is None else float(rate)

        print(numeric_rate)
        self._reactions.add_reaction_original(
            reactants,
            products,
            rate=numeric_rate,
            rate_name=str(rate_name),
        )
        return self


    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, *, rates: Dict[str, float]) -> "HybridModel":
        if self._domain is None:
            raise ValueError("domain() not set")
        if self._conversion is None:
            raise ValueError("conversion() not set")
        if self._diffusion is None:
            raise ValueError("diffusion() not set")
        if self._rhs_user is None:
            raise ValueError("reaction_terms() not set")

        self._rates = {str(k): float(v) for k, v in rates.items()}

        # ✅ update macroscopic stored rates for display
        for rec in self._reactions.pure_reactions:
            rn = rec.get("rate_name", None)
            if rn is not None and rn in self._rates:
                rec["rate"] = float(self._rates[rn])


        n = len(self.species)

        def pde_terms(C: np.ndarray, rates: Dict[str, float]) -> np.ndarray:
            args = [C[i] for i in range(n)]
            out = self._rhs_user(*args, rates)

            if isinstance(out, np.ndarray):
                return out.astype(float, copy=False)

            out = tuple(out)
            if len(out) != n:
                raise ValueError("reaction_terms returned wrong number of species")

            return np.array(out, dtype=float)

        self._engine = SRCMEngine(
            reactions=self._reactions,
            pde_reaction_terms=pde_terms,
            diffusion_rates=self._diffusion,
            domain=self._domain,
            conversion=self._conversion,
            reaction_rates=self._rates,
        )
        return self

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def _check_ic(self, init_ssa: np.ndarray, init_pde: np.ndarray):
        d = self._domain
        n = len(self.species)

        if init_ssa.shape != (n, d.K):
            raise ValueError("init_ssa has wrong shape")
        if init_pde.shape != (n, d.n_pde):
            raise ValueError("init_pde has wrong shape")

    def run(
        self,
        init_ssa: np.ndarray,
        init_pde: np.ndarray,
        *,
        time: float,
        dt: float,
        seed: int = 0,
    ):
        self._check_ic(init_ssa, init_pde)

        if self._engine is None:
            raise RuntimeError("Model not built yet. Call build(rates=...) first.")

        return self._engine.run(
            initial_ssa=init_ssa,
            initial_pde=init_pde,
            time=float(time),
            dt=float(dt),
            seed=int(seed),
        )


    def run_repeats(
        self,
        init_ssa: np.ndarray,
        init_pde: np.ndarray,
        *,
        time: float,
        dt: float,
        repeats: int,
        seed: int = 0,
        parallel: bool = False,
        n_jobs: int = -1,
        progress: bool = True,
        prefer: str = "processes",
    ):
        self._check_ic(init_ssa, init_pde)
        if self._engine is None:
            raise RuntimeError("Model not built yet. Call build(rates=...) first.")

        return self._engine.run_repeats(
            initial_ssa=init_ssa,
            initial_pde=init_pde,
            time=float(time),
            dt=float(dt),
            repeats=int(repeats),
            seed=int(seed),
            parallel=bool(parallel),
            n_jobs=int(n_jobs),
            prefer=str(prefer),
            progress=bool(progress),
        )


    def metadata(self) -> dict:
        if self._engine is None:
            raise RuntimeError("Model not built yet")

        d = self._domain
        if d is None:
            raise RuntimeError("Domain not configured")

        diffusion = dict(self._diffusion_rates) if self._diffusion_rates is not None else None
        conversion = self._conversion
        rates = dict(self._rates) if self._rates is not None else None

        return {
            "model": "SRCM Hybrid Model",
            "species": list(self.species),

            # domain
            "L": float(d.length),
            "K": int(d.K),
            "pde_multiple": int(d.pde_multiple),
            "boundary": str(d.boundary),

            # diffusion
            "diffusion_rates": diffusion,

            # conversion
            # (scalar for global params, list for per-species params)
            "threshold_particles": (
                (int(conversion.threshold) if isinstance(conversion.threshold, (int, float))
                 else [float(v) for v in conversion.threshold])
                if conversion is not None else None
            ),
            "conversion_rate": (
                (float(conversion.rate) if isinstance(conversion.rate, (int, float))
                 else [float(v) for v in conversion.rate])
                if conversion is not None else None
            ),

            # reactions
            "reaction_rates": rates,
            "hybrid_labels": self.hybrid_labels(),
        }





    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def describe_reactions(self) -> None:
        """
        Print macroscopic reactions and their SRCM decomposition.
        """
        self._reactions.describe()

    def hybrid_labels(self) -> List[str]:
        return [hr.label for hr in self._reactions.hybrid_reactions]
    


        # ------------------------------------------------------------------
    # compatibility + convenience
    # ------------------------------------------------------------------
    @property
    def domain_obj(self) -> Domain:
        if self._domain is None:
            raise RuntimeError("Domain not configured yet. Call m.domain(...) first.")
        return self._domain

    @property
    def reactions(self) -> HybridReactionSystem:
        return self._reactions

    @property
    def engine(self) -> SRCMEngine:
        if self._engine is None:
            raise RuntimeError("Model not built yet. Call m.build(rates=...) first.")
        return self._engine

    def describe_reactions(self) -> None:
        # prefer full description if available, otherwise fallback
        if hasattr(self._reactions, "describe_full"):
            self._reactions.describe_full()
        else:
            self._reactions.describe()

