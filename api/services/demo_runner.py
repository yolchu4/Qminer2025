from pathlib import Path
from typing import Optional, Dict
import subprocess
import sys
import json
import shutil
from api.services.validators import validate_datasets
from api.services.artifacts import prepare_output_dir


# ---------- helpers ----------

def _save_upload(upload, dst: Path):
    with open(dst, "wb") as f:
        f.write(upload.file.read())


# ---------- main orchestration ----------

def run_demo(
    train_file=None,
    calib_file=None,
    predict_file=None,
    compare_file=None,
    *,
    n_regimes: int,
    alpha: float,
    output_dir: Optional[str] = None,
    local_paths: Optional[Dict[str, Path]] = None,
):
    
    """
    Orchestrates a full demo run.

    Two modes:
    1) Upload mode
    2) Local mode

    Core script (scripts/run_demo.py) is executed via subprocess (Frozen).
    """

    # --------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------
    work_dir = prepare_output_dir(output_dir)

    # --------------------------------------------------
    # Resolve dataset paths
    # --------------------------------------------------
    paths: Dict[str, Path] = {}

    if local_paths is not None:
        for key in ["train", "calib", "predict", "compare"]:
            p = local_paths.get(key)
            if p is not None:
                paths[key] = Path(p)
    else:
        if train_file is None or calib_file is None or predict_file is None:
            raise ValueError("train, calib and predict files are required")

        train_path = work_dir / "train_dataset.csv"
        calib_path = work_dir / "calib_dataset.csv"
        predict_path = work_dir / "predict_dataset.csv"

        _save_upload(train_file, train_path)
        _save_upload(calib_file, calib_path)
        _save_upload(predict_file, predict_path)

        paths["train"] = train_path
        paths["calib"] = calib_path
        paths["predict"] = predict_path

        if compare_file is not None:
            compare_path = work_dir / "compare_dataset.csv"
            _save_upload(compare_file, compare_path)
            paths["compare"] = compare_path

    # --------------------------------------------------
    # Validate data contract (Frozen)
    # --------------------------------------------------
    validate_datasets(paths)

    # --------------------------------------------------
    # Call frozen core demo script (subprocess)
    # --------------------------------------------------
    cmd = [
        sys.executable,
        "scripts/run_demo.py",
        "--train", str(paths["train"]),
        "--calib", str(paths["calib"]),
        "--predict", str(paths["predict"]),
        "--n_regimes", str(n_regimes),
        "--alpha", str(alpha),
        "--output_dir", str(work_dir),
    ]

    if paths.get("compare") is not None:
        cmd.extend(["--compare", str(paths["compare"])])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    print("=== run_demo.py STDOUT ===")
    print(result.stdout)
    print("=== run_demo.py STDERR ===")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "run_demo.py failed\n"
            + result.stdout
            + "\n"
            + result.stderr
        )
    # --------------------------------------------------
# Copy compare_dataset.csv to output dir for UI plots
# --------------------------------------------------
    if "compare" in paths:
        shutil.copy(
        paths["compare"],
        work_dir / "compare_dataset.csv"
    )
  
    # --------------------------------------------------
    # Load metrics from metrics.json (SOURCE OF TRUTH)
    # --------------------------------------------------
       
# --------------------------------------------------
# Load metrics.json (robust: find latest written)
# --------------------------------------------------

    metrics = {}

    out_root = Path("out")
    metrics_files = list(out_root.rglob("metrics.json"))

    if metrics_files:
        latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
        with open(latest_metrics, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # --------------------------------------------------
    # Standard response (UI-friendly)
    # --------------------------------------------------
    return {
    "status": "success",
    "output_dir": "out/demo_run",
    "files": [
        "pred_mean.csv",
        "pred_q05.csv",
        "pred_q95.csv",
        "regime_weights.csv",
    ],
    "metrics": metrics,
    }






