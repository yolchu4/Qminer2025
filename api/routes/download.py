from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

@router.get("/{run_id}/{filename}")
def download_file(run_id: str, filename: str):
    file_path = Path("out") / run_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path=str(file_path), filename=filename, media_type="text/csv")
