import argparse
import json
from pathlib import Path
import numpy as np

def make_turing_bump_ic(
    K: int,
    L: float,
    omega: float,
    pert_a: float,
    bar_r_11: float,
    bar_r_12: float,
    bar_r_2: float,
    bar_r_3: float,
):
    # compartment width
    h = L / K

    # dimensionless steady states
    dim_U_ss = bar_r_11 * bar_r_3 / (bar_r_12 * bar_r_2)
    dim_V_ss = (bar_r_11**2) * bar_r_3 / (bar_r_12 * (bar_r_2**2))

    # Convert to compartment mass (your notebook logic)
    U_ss_mass = dim_U_ss * omega * h
    V_ss_mass = dim_V_ss * omega * h

    x = (np.arange(K) + 0.5) / K
    bump = 1.0 + pert_a * np.cos(2.0 * np.pi * (x - 0.5))

    U0 = np.round(U_ss_mass * bump).astype(int)
    V0 = np.round(V_ss_mass * bump).astype(int)

    U0 = np.clip(U0, 0, None)
    V0 = np.clip(V0, 0, None)

    init_ssa = np.zeros((2, K), dtype=int)
    init_ssa[0, :] = U0
    init_ssa[1, :] = V0

    return init_ssa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--pde-multiple", type=int, default=4)
    ap.add_argument("--omega", type=float, default=500)
    ap.add_argument("--pert-a", type=float, default=0.07)

    ap.add_argument("--bar-r-11", type=float, default=1.0)
    ap.add_argument("--bar-r-12", type=float, default=1.0)
    ap.add_argument("--bar-r-2", type=float, default=2.0)
    ap.add_argument("--bar-r-3", type=float, default=0.6)

    ap.add_argument("--out", type=str, default="ic_turing_bump.npz")
    args = ap.parse_args()

    init_ssa = make_turing_bump_ic(
        K=args.K,
        L=args.L,
        omega=args.omega,
        pert_a=args.pert_a,
        bar_r_11=args.bar_r_11,
        bar_r_12=args.bar_r_12,
        bar_r_2=args.bar_r_2,
        bar_r_3=args.bar_r_3,
    )

    n_pde = args.K * args.pde_multiple
    init_pde = np.zeros((2, n_pde), dtype=float)

    meta = {
        "ic_kind": "turing_bump",
        "species": ["U", "V"],
        "K": args.K,
        "L": args.L,
        "pde_multiple": args.pde_multiple,
        "omega": args.omega,
        "pert_a": args.pert_a,
        "bar_r_11": args.bar_r_11,
        "bar_r_12": args.bar_r_12,
        "bar_r_2": args.bar_r_2,
        "bar_r_3": args.bar_r_3,
    }

    out_path = Path(args.out).resolve()
    np.savez_compressed(
        out_path,
        init_ssa=init_ssa,
        init_pde=init_pde,
        species=np.array(["U", "V"], dtype=object),
        meta=json.dumps(meta),
    )
    print("Saved IC NPZ:", out_path)

if __name__ == "__main__":
    main()
