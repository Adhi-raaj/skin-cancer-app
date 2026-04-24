"""
main.py — FastAPI backend for HAM10000 Skin Cancer Classifier
Run with:  uvicorn main:app --reload --port 8000
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from model import SkinCancerModel, CLASSES, CLASS_INFO

# ── Config ───────────────────────────────────────────────────────────────────

CHECKPOINT = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).parent / "models" / "convnext_best.pth"),
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ── Startup / Shutdown ───────────────────────────────────────────────────────

_model: SkinCancerModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print("🔄 Loading model…")
    try:
        _model = SkinCancerModel(CHECKPOINT)
        print("✅ Model ready!")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   API will return 503 until a checkpoint is placed in backend/models/")
    yield
    _model = None
    print("🛑 Model unloaded.")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HAM10000 Skin Cancer Classifier",
    description=(
        "ConvNeXt-Small + Focal Loss + 10-Pass TTA + Grad-CAM++ "
        "for 7-class dermoscopy classification"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend UI."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Skin Cancer Classifier API — visit /docs"})


@app.get("/health")
async def health():
    """Health check — returns model status."""
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "classes":      CLASSES,
        "num_classes":  len(CLASSES),
    }


@app.get("/classes")
async def get_classes():
    """Return class metadata for the frontend."""
    return {
        cls: {
            "name":  info["name"],
            "risk":  info["risk"],
            "color": info["color"],
            "desc":  info["desc"],
        }
        for cls, info in CLASS_INFO.items()
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    tta_passes: int  = 10,
):
    """
    Classify a dermoscopy image.

    - **file**: JPG or PNG image (max 10 MB recommended)
    - **tta_passes**: Test-time augmentation passes (1–10, default 10)

    Returns predictions, confidence scores, Grad-CAM++ overlay,
    and hair-removed image — all as base64 JPEGs.
    """
    # Model readiness
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Place 'convnext_best.pth' in backend/models/ and restart the server."
            ),
        )

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPG or PNG.",
        )

    # Clamp TTA passes
    tta_passes = max(1, min(tta_passes, 10))

    # Read and predict
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = _model.predict(image_bytes, tta_passes=tta_passes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return JSONResponse(content=result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
