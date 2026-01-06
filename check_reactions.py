"""
Quick manual sanity checks for HybridReactionSystem.

Run:
  python check_reactions.py
"""

import numpy as np
from srcm_engine.reactions import HybridReactionSystem


def main():
    # ----------------------------
    # 1) Define species + reactions
    # ----------------------------
    reactions = HybridReactionSystem(species=["U", "V"])

    # Pure reactions (optional / documentation)
    reactions.add_reaction({"U": 2}, {"U": 3}, rate=1.0)            # 2U -> 3U
    reactions.add_reaction({"U": 2}, {"U": 2, "V": 1}, rate=2.0)    # 2U -> 2U + V
    reactions.add_reaction({"U": 1, "V": 1}, {"V": 1}, rate=1.0)    # U + V -> V
    reactions.add_reaction({"V": 1}, {}, rate=3.5)                  # V -> 0

    # Hybrid reactions (the ones the engine will actually use)
    reactions.add_hybrid_reaction(
        reactants={"D_U": 2},
        products={"D_U": 3},
        propensity=lambda D, C, r, h: r["r_11"] * D["U"] * (D["U"] - 1) / h,
        state_change={"D_U": +1},
        label="R1_1",
        description="2 D^U -> 3 D^U"
    )

    reactions.add_hybrid_reaction(
        reactants={"D_U": 1, "C_U": 1},
        products={"D_U": 3, "C_U": 1},
        propensity=lambda D, C, r, h: 2.0 * r["r_11"] * D["U"] * C["U"] / h,
        state_change={"D_U": +1},
        label="R1_2",
        description="D^U + C^U -> 3 D^U (collapsed mixed, factor 2)"
    )

    reactions.add_hybrid_reaction(
        reactants={"D_V": 1},
        products={},
        propensity=lambda D, C, r, h: r["r_3"] * D["V"],
        state_change={"D_V": -1},
        label="R4_1",
        description="D^V -> 0"
    )

    # ----------------------------
    # 2) Fake a local compartment state
    # ----------------------------
    # SSA counts in compartment i
    D = {"U": 10, "V": 4}       # D^U, D^V
    # PDE mass in same compartment i (not concentration; integrated mass)
    C = {"U": 2.5, "V": 0.1}    # C^U, C^V

    rates = {"r_11": 1.0, "r_3": 3.5}
    h = 0.1

    # ----------------------------
    # 3) Evaluate propensities + show state changes
    # ----------------------------
    print("\n=== Hybrid reactions sanity check ===")
    print(f"Local state: D={D}, C={C}, h={h}, rates={rates}\n")

    for hr in reactions.hybrid_reactions:
        a = hr.propensity(D, C, rates, h)
        print(f"{hr.label:>4} | propensity = {a:.6g} | state_change = {hr.state_change}")
        if hr.description:
            print(f"      {hr.description}")

    # ----------------------------
    # 4) Quick check: labels + species
    # ----------------------------
    print("\n=== Stored metadata ===")
    print("Species:", reactions.species)
    print("Num hybrid reactions:", len(reactions.hybrid_reactions))
    print("Num pure reactions:", len(reactions.pure_reactions))

    print("\nAll good ✅\n")

        # ----------------------------
    # 3) Print all hybrid reactions
    # ----------------------------
    reactions.describe()

    # ----------------------------
    # 4) Evaluate propensities at a fake state
    # ----------------------------
    print("=== Propensities at test state ===")
    print(f"D = {D}, C = {C}, h = {h}, rates = {rates}\n")

    for hr in reactions.hybrid_reactions:
        a = hr.propensity(D, C, rates, h)
        print(f"{hr.label:>4} | propensity = {a:.6g}")


if __name__ == "__main__":
    main()
