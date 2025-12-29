from pydantic import BaseModel, Field
from typing import Optional

class RunDemoRequest(BaseModel):
    n_regimes: int = Field(..., ge=1, le=20)
    alpha: float = Field(0.90, ge=0.5, le=0.99)
    output_dir: Optional[str] = None
