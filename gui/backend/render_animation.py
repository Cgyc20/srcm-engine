from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

from srcm_engine.animation_util import AnimationConfig, animate_results


def render_animation(res, cfg: AnimationConfig, fmt: str = "gif") -> Path:
    """
    Render animation to a file (GIF or MP4) and return the path.
    Requires animate_results(...) to RETURN a matplotlib.animation.FuncAnimation.
    """
    fmt = fmt.lower()
    if fmt not in ("gif", "mp4"):
        raise ValueError("fmt must be 'gif' or 'mp4'")

    out_dir = Path(tempfile.mkdtemp())
    out_path = out_dir / f"animation.{fmt}"

    plt.close("all")
    anim = animate_results(res, cfg=cfg)

    if anim is None:
        raise RuntimeError(
            "animate_results returned None. Update it to return the FuncAnimation object."
        )

    fps = max(1, int(1000 / cfg.interval_ms))

    if fmt == "gif":
        # pip install pillow
        anim.save(str(out_path), writer="pillow", fps=fps)
    else:
        # system ffmpeg required
        anim.save(str(out_path), writer="ffmpeg", fps=fps)

    return out_path
