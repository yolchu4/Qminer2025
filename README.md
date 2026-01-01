# Intelligent Process Control System (Demo)

This repository contains a **demo-level intelligent process control system** featuring
regime-aware modeling, explicit uncertainty calibration, and a lightweight UI connected
to a local backend.

> **Important:**  
> This demo is intended to be evaluated by **running it locally**.  
> The UI alone does not represent the full functionality.

## Local Run (Recommended)

### Requirements
- Python 3.9+
- pip

### Install dependencies

pip install -r requirements.txt
Start the backend

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Open the UI
Run a local static server in the project root:

python -m http.server
Then open in your browser:

http://localhost:8000
Demo Behavior
On page load:

The UI automatically displays results from the last demo run
located in out/demo_run/

Tables and plots are pre-filled
When clicking Run Demo:
Parameters are sent to the backend
The demo script is executed
New outputs overwrite out/demo_run/
Tables and plots refresh automatically

Output Location
All demo outputs are written to:
out/demo_run/
Key files include:
pred_mean.csv
pred_q05.csv
pred_q95.csv
regime_weights.csv
metrics.json
UI-Only Deployment (Preview)