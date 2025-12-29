from pathlib import Path
from datetime import datetime

def prepare_output_dir(output_dir: str | None):
    if output_dir:
        p = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = Path("out") / f"demo_run_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p
