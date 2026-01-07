import tempfile
from pathlib import Path
from typing import Tuple, Any, Dict

from srcm_engine.results.io import load_npz, save_npz


def load_npz_from_uploaded(uploaded_file) -> Tuple[Any, Dict]:
    """
    Streamlit uploader returns a file-like object.
    srcm_engine.load_npz expects a filesystem path.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    res, meta = load_npz(str(tmp_path))
    return res, meta


def save_npz_to_temp(res, meta: dict, filename: str = "result.npz") -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    out_path = tmp_dir / filename
    save_npz(res, str(out_path), meta=meta)
    return out_path
