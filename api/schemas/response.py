from pydantic import BaseModel
from typing import Dict, List

class MetricsResponse(BaseModel):
    mae: Dict[str, float]
    coverage: Dict[str, float]
    delta: Dict[str, float]

class RunDemoResponse(BaseModel):
    status: str
    output_dir: str
    metrics: MetricsResponse
    files: List[str]
##########################################3
run_id = Path(work_dir).name
return {
    "status": "success",
    "run_id": run_id,
    "output_dir": str(work_dir),
    "metrics": metrics,
    "files": ["pred_mean.csv","pred_q05.csv","pred_q95.csv","regime_weights.csv"],
}
