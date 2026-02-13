"""
Solar Saathi — FastAPI Backend
================================
REST API for solar panel lifespan prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os
import uvicorn

from predictor import SolarSaathiPredictor

# --- App initialization ---
app = FastAPI(
    title="Solar Saathi API",
    description="Hybrid LSTM+XGBoost model for solar panel lifespan prediction",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# --- Model loading ---
predictor = SolarSaathiPredictor()


@app.on_event("startup")
async def startup():
    """Load ML models on server startup."""
    success = predictor.load_models()
    if not success:
        print("⚠ Models not loaded. Prediction endpoints will use fallback mode.")


# --- Request/Response models ---
class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude of installation site")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude of installation site")
    module_type: str = Field(default="Mono-Si", description="Panel technology: Mono-Si, Poly-Si, CdTe, CIGS")
    mounting_type: str = Field(default="fixed_tilt", description="Mounting: fixed_tilt, rooftop, open_rack")
    tilt_angle: float = Field(default=25.0, ge=0, le=90, description="Panel tilt angle in degrees")
    panel_wattage: int = Field(default=400, ge=100, le=800, description="Panel wattage (W)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# --- Endpoints ---
@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Solar Saathi API is running. Use /docs for API documentation."}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.loaded,
        version="1.0.0"
    )


@app.post("/predict")
async def predict(req: PredictionRequest):
    """
    Predict solar panel lifespan for a given location and configuration.
    
    Uses real-time NASA POWER API data + hybrid LSTM+XGBoost model.
    """
    if not predictor.loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Please train the models first."
        )
    
    try:
        result = predictor.predict(
            latitude=req.latitude,
            longitude=req.longitude,
            module_type=req.module_type,
            mounting_type=req.mounting_type,
            tilt_angle=req.tilt_angle,
            panel_wattage=req.panel_wattage,
        )
        return {"success": True, "data": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/locations")
async def get_sample_locations():
    """Return sample locations for quick testing."""
    return {
        "locations": [
            {"name": "Mumbai, India", "lat": 19.076, "lon": 72.878},
            {"name": "Delhi, India", "lat": 28.614, "lon": 77.209},
            {"name": "Chennai, India", "lat": 13.083, "lon": 80.270},
            {"name": "Dubai, UAE", "lat": 25.205, "lon": 55.271},
            {"name": "Phoenix, USA", "lat": 33.449, "lon": -112.074},
            {"name": "Berlin, Germany", "lat": 52.520, "lon": 13.405},
            {"name": "Sydney, Australia", "lat": -33.869, "lon": 151.209},
            {"name": "Nairobi, Kenya", "lat": -1.286, "lon": 36.817},
            {"name": "Sao Paulo, Brazil", "lat": -23.551, "lon": -46.634},
            {"name": "Tokyo, Japan", "lat": 35.682, "lon": 139.692},
            {"name": "Cairo, Egypt", "lat": 30.044, "lon": 31.236},
            {"name": "Stockholm, Sweden", "lat": 59.329, "lon": 18.069},
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
