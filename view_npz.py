import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from srcm_engine.results.io import load_npz
from srcm_engine.animation_util import AnimationConfig, animate_results, plot_mass_time_series


def main():
    parser = argparse.ArgumentParser(description="View SRCM .npz results")
    parser.add_argument("npz", type=str, help="Path to .npz results file")
    parser.add_argument("--plot", action="store_true", help="Show mass time series plot")
    parser.add_argument("--anim", action="store_true", help="Show animation window")
    parser.add_argument("--stride", type=int, default=20, help="Animation stride (default: 20)")
    parser.add_argument("--interval", type=int, default=25, help="Animation interval ms (default: 25)")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    res, meta = load_npz(str(npz_path))

    # ---- Print metadata nicely ----
    print("\n=== META ===")
    for k in sorted(meta.keys()):
        v = meta[k]
        if k == "spec":
            print(f"{k}: <model spec dict ({len(v)} keys)>")
        else:
            print(f"{k}: {v}")

    # ---- Print reaction info if present ----
    print("\n=== REACTIONS (from saved spec, if available) ===")
    spec = meta.get("spec", None)
    if isinstance(spec, dict):
        rxns = spec.get("reactions", [])
        rates = spec.get("rates", {})
        if rxns:
            for i, rxn in enumerate(rxns, 1):
                reactants = rxn.get("reactants", {})
                products = rxn.get("products", {})
                rate_name = rxn.get("rate_name", "?")
                rate_val = rates.get(rate_name, None)
                print(f"{i}. {reactants} -> {products}   rate_name={rate_name}   rate={rate_val}")
        else:
            print("No reactions found in meta['spec']['reactions'].")
    else:
        print("No meta['spec'] saved in this npz file.")

    # ---- Plot ----
    if args.plot:
        plt.close("all")
        plot_mass_time_series(res, plot_mode="per_species")
        plt.tight_layout()
        plt.show()

    # ---- Animation ----
    if args.anim:
        plt.close("all")
        cfg = AnimationConfig(
            stride=args.stride,
            interval_ms=args.interval,
            threshold_particles=meta.get("threshold_particles", 0),
            title=f"SRCM: {npz_path.name}",
            mass_plot_mode="per_species",
        )
        ani = animate_results(res, cfg=cfg)
        # IMPORTANT: animate_results must call plt.show() OR return ani and we call plt.show()
        # Best: animate_results returns ani and we show:
        plt.show()

    if not args.plot and not args.anim:
        print("\nTip: add --plot and/or --anim to view visuals.")


if __name__ == "__main__":
    main()
