# ===========================================
# GradSure: GBM Risk API Server (Final Fix)
# ===========================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager # Added for lifespan management

# -------------------------------------------
# 0. Core Prediction Logic 
# -------------------------------------------

# Global variables to store the loaded model components
# These will be populated during the application startup event.
gbm_pipeline: Optional[Any] = None
feature_names: Optional[list] = None

def _score_to_tier(risk_score: float) -> str:
    """Convert risk probability (0-1) to categorical tier (PREFERRED, STANDARD, etc.)."""
    if risk_score <= 0.15:
        return "PREFERRED"
    elif risk_score <= 0.35:
        return "STANDARD"  
    elif risk_score <= 0.65:
        return "SUBSTANDARD"
    else:
        return "HIGH_RISK"

def predict_risk_logic(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the core prediction logic using the loaded global model components."""
    global gbm_pipeline, feature_names
    
    if gbm_pipeline is None or feature_names is None:
         raise RuntimeError("Model components are not loaded. Risk service unavailable.")
         
    # Convert input dictionary to DataFrame for the pipeline
    input_df = pd.DataFrame([input_data])
    
    # Predict raw risk score (0-1) using the explicitly stored features
    risk_probability = float(gbm_pipeline.predict(input_df[feature_names])[0])
    
    # Clip and scale the output
    risk_probability = np.clip(risk_probability, 0.0, 1.0)
    risk_score_1_5 = 1 + (risk_probability * 4) # Convert 0-1 to 1-5 scale
    risk_tier = _score_to_tier(risk_probability)
    
    return {
        'risk_score': round(risk_score_1_5, 1),
        'risk_tier': risk_tier,
        'risk_probability': round(risk_probability, 4)
    }

# -------------------------------------------
# 1. Model Loading (Fixed using Lifespan Event)
# -------------------------------------------

MODEL_PATH = "final_gbm_pipeline.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan event handler to load the model before the server starts.
    This replaces the deprecated @app.on_event("startup").
    """
    global gbm_pipeline, feature_names
    try:
        # Load the dictionary containing the pipeline and feature names
        loaded_data = joblib.load(MODEL_PATH)
        
        # Check if the expected keys are present (pipeline and features)
        if 'pipeline' not in loaded_data or 'features' not in loaded_data:
            raise ValueError("Model file content is invalid. Expected keys 'pipeline' and 'features'.")
            
        gbm_pipeline = loaded_data['pipeline']
        feature_names = loaded_data['features']
        
        print(f"✅ Risk Assessment GBM components loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Risk model file '{MODEL_PATH}' not found. Please train the model first.")
        gbm_pipeline = None
    except Exception as e:
        # Catch errors from joblib, ValueError, etc.
        print(f"❌ Error loading model during startup: {e}") 
        gbm_pipeline = None
        
    # Yield control to the application, which will start serving requests
    yield
    
    # Optional: Clean up code would go here if needed

# -------------------------------------------
# 3. FastAPI App & Endpoints 
# -------------------------------------------

app = FastAPI(
    lifespan=lifespan, # <-- Updated to use the new lifespan context manager
    title="GradSure Risk Assessment API",
    description="GBM model for risk assessment - outputs continuous risk score and tier.",
    version="2.0"
)

# -------------------------------------------
# 2. Input and Output Schemas
# -------------------------------------------

class RiskInput(BaseModel):
    sponsor_age: int = Field(..., ge=18, le=80)
    sponsor_sex: str = Field(..., pattern="^(M|F)$")
    sponsor_relationship: str
    student_age: int = Field(..., ge=10, le=30) 
    student_sex: str = Field(..., pattern="^(M|F)$")
    income_band: str

class RiskOutput(BaseModel):
    risk_score: float = Field(..., description="1.0-5.0 scale risk score.")
    risk_tier: str = Field(..., description="PREFERRED/STANDARD/SUBSTANDARD/HIGH_RISK.")
    risk_probability: float = Field(..., description="0-1 raw probability score from the regressor.")

# -------------------------------------------
# 4. Endpoints
# -------------------------------------------

@app.post("/api/v2/assess-risk", response_model=RiskOutput)
def assess_risk(data: RiskInput):
    """
    Assess applicant risk using the GBM model based on demographic features.
    """
    global gbm_pipeline 
    if gbm_pipeline is None:
        # This will happen if the startup event failed to load the model
        raise HTTPException(status_code=503, detail="Risk assessment service unavailable. Model failed to load.")
    
    try:
        # Call the standalone prediction logic
        result = predict_risk_logic(data.model_dump())
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Risk assessment failed: {str(e)}")

@app.get("/health")
def health_check():
    """Simple endpoint to check API status and model load status."""
    return {"status": "healthy", "model_loaded": gbm_pipeline is not None}
