from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path

from api.services.demo_runner import run_demo

router = APIRouter()

# --------------------------------------------------
# Upload-based run
# --------------------------------------------------
@router.post("/demo")
async def run_demo_upload(
    train_file: UploadFile = File(...),
    calib_file: UploadFile = File(...),
    predict_file: UploadFile = File(...),
    compare_file: UploadFile | None = File(None),
    n_regimes: int = Form(...),
    alpha: float = Form(0.90),
    output_dir: str | None = Form(None),
):
    try:
        return run_demo(
            train_file=train_file,
            calib_file=calib_file,
            predict_file=predict_file,
            compare_file=compare_file,
            n_regimes=n_regimes,
            alpha=alpha,
            output_dir=output_dir,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------
# Local data run (NO upload)
# --------------------------------------------------
@router.post("/demo-local")
def run_demo_local(
    n_regimes: int = Form(...),
    alpha: float = Form(0.90),
    data_dir: str = Form("data"),
    output_dir: str | None = Form(None),
):
    data_path = Path(data_dir)

    train_path = data_path / "train_dataset.csv"
    calib_path = data_path / "calib_dataset.csv"
    predict_path = data_path / "predict_dataset.csv"
    compare_path = data_path / "compare_dataset.csv"

    if not train_path.exists():
        raise HTTPException(400, "train_dataset.csv not found")
    if not calib_path.exists():
        raise HTTPException(400, "calib_dataset.csv not found")
    if not predict_path.exists():
        raise HTTPException(400, "predict_dataset.csv not found")

    return run_demo(
        n_regimes=n_regimes,
        alpha=alpha,
        output_dir=output_dir,
        local_paths={
            "train": train_path,
            "calib": calib_path,
            "predict": predict_path,
            "compare": compare_path if compare_path.exists() else None,
        },
    )
