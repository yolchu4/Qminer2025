from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Qminer Demo API")

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Absolute project root
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "out"

print("PROJECT_ROOT =", PROJECT_ROOT)
print("OUT_DIR =", OUT_DIR)
print("OUT_DIR exists =", OUT_DIR.exists())

# -------------------------
# Mount static files
# -------------------------
app.mount(
    "/out",
    StaticFiles(directory=str(OUT_DIR), html=False),
    name="out",
)

# -------------------------
# Routers
# -------------------------
from api.routes.run import router as run_router
app.include_router(run_router, prefix="/run")
