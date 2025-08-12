"""
FastAPI Model Serving for ML Recommendation System
Production-ready API for employee shift recommendations

Endpoints:
- POST /predict - Get employee recommendations for a shift
- GET /health - Health check
- GET /models/info - Model information
- GET /employees - Get available employees
- GET /stats - System statistics
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# ML imports
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for loaded models
MODELS = {}
SCALERS = {}
EMBEDDINGS = {}
FEATURE_COLUMNS = []
MODEL_METADATA = {}
EMPLOYEE_DATABASE = pd.DataFrame()

# Pydantic models for API requests/responses
class ShiftRequest(BaseModel):
    """Request model for shift recommendations"""
    
    EmployerId: int = Field(..., description="Employer ID")
    SiteId: int = Field(..., description="Site ID") 
    TaskId: int = Field(..., description="Task ID")
    FKshiftId: int = Field(..., description="Shift ID")
    WeekDay: int = Field(..., ge=1, le=7, description="Day of week (1-7)")
    CreatedBy: int = Field(..., description="Manager/Creator ID")
    ScheduleWeek: Optional[int] = Field(None, description="Schedule week (YYYYWW)")
    top_n: Optional[int] = Field(5, ge=1, le=20, description="Number of recommendations")
    
    @validator('ScheduleWeek')
    def validate_schedule_week(cls, v):
        if v is None:
            # Default to current week
            current_date = datetime.now()
            year = current_date.year
            week = current_date.isocalendar()[1]
            return int(f"{year}{week:02d}")
        return v

class EmployeeRecommendation(BaseModel):
    """Response model for individual employee recommendation"""
    
    EmployeeId: int
    overall_score: float = Field(..., description="Overall recommendation score (0-100)")
    success_probability: float = Field(..., description="Assignment success probability (0-1)")
    employee_satisfaction_pred: float = Field(..., description="Predicted employee satisfaction (1-10)")
    employer_satisfaction_pred: float = Field(..., description="Predicted employer satisfaction (1-10)")
    completion_likelihood: float = Field(..., description="Completion probability (0-1)")
    confidence: float = Field(..., description="Model confidence (0-1)")
    recommendation_reason: str = Field(..., description="Human-readable explanation")
    employee_info: Dict[str, Any] = Field(default_factory=dict, description="Employee details")

class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint"""
    
    status: str = "success"
    timestamp: str
    shift_request: ShiftRequest
    recommendations: List[EmployeeRecommendation]
    total_candidates: int
    processing_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = "healthy"
    timestamp: str
    models_loaded: int
    uptime_seconds: float
    version: str = "1.0.0"

class ModelInfo(BaseModel):
    """Model information response"""
    
    models: Dict[str, str]
    total_features: int
    feature_columns: List[str]
    training_date: str
    model_performance: Dict[str, float]

# Initialize FastAPI app
app = FastAPI(
    title="Employee Recommendation API",
    description="ML-powered employee shift recommendation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time
START_TIME = datetime.now()

class ModelLoader:
    """Load and manage ML models"""
    
    def __init__(self, models_path: str = "D:/Employee_Suggestions/employee_suggestion_system/models"):
        self.models_path = Path(models_path)
        
    def load_all_models(self):
        """Load all trained models and components"""
        
        global MODELS, SCALERS, EMBEDDINGS, FEATURE_COLUMNS, MODEL_METADATA
        
        logger.info("🔄 Loading trained models...")
        
        try:
            # Load sklearn models
            model_files = {
                'success': 'success_model_latest.pkl',
                'satisfaction': 'satisfaction_model_latest.pkl', 
                'quality': 'quality_model_latest.pkl',
                'completion': 'completion_model_latest.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_path / filename
                if model_path.exists():
                    MODELS[model_name] = joblib.load(model_path)
                    logger.info(f"   ✅ Loaded {model_name} model")
                else:
                    logger.warning(f"   ⚠️  Model not found: {filename}")
            
            # Load neural ensemble (with error handling for version compatibility)
            neural_path = self.models_path / "neural_ensemble_latest.h5"
            if neural_path.exists():
                try:
                    MODELS['neural_ensemble'] = keras.models.load_model(neural_path)
                    logger.info(f"   ✅ Loaded neural ensemble model")
                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to load neural ensemble: {str(e)}")
                    logger.info(f"   ℹ️  Continuing without neural ensemble (individual models still work)")
                    # Continue without neural ensemble - individual models are sufficient
            
            # Load scalers
            scalers_path = self.models_path / "scalers_latest.pkl"
            if scalers_path.exists():
                SCALERS = joblib.load(scalers_path)
                logger.info(f"   ✅ Loaded scalers")
            
            # Load embeddings
            embeddings_path = self.models_path / "embeddings.pkl"
            if embeddings_path.exists():
                EMBEDDINGS = joblib.load(embeddings_path)
                logger.info(f"   ✅ Loaded embeddings")
            
            # Load metadata
            metadata_path = self.models_path / "model_metadata_latest.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    MODEL_METADATA = json.load(f)
                    FEATURE_COLUMNS = MODEL_METADATA.get('feature_columns', [])
                logger.info(f"   ✅ Loaded metadata")
            
            logger.info(f"🎉 Successfully loaded {len(MODELS)} models")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {str(e)}")
            raise

    def load_employee_database(self):
        """Load employee data for recommendations"""
        
        global EMPLOYEE_DATABASE
        
        try:
            # Try to load from latest features first
            features_path = Path("D:/Employee_Suggestions/employee_suggestion_system/data/interim/latest_features.csv")
            if features_path.exists():
                logger.info(f"   Loading employee data from: {features_path}")
                df = pd.read_csv(features_path)
                
                # Create employee summary database with real employee data
                EMPLOYEE_DATABASE = df.groupby('EmployeeId').agg({
                    'emp_SiteId_count': 'first',
                    'emp_SiteId_nunique': 'first', 
                    'emp_TaskId_nunique': 'first',
                    'emp_years_active': 'first',
                    'emp_weekly_consistency': 'first',
                    'workload_ratio': 'first',
                    'needs_opportunities': 'first',
                    'is_recent_worker': 'first'
                }).round(3)
                
                logger.info(f"   ✅ Loaded {len(EMPLOYEE_DATABASE)} real employee profiles")
                logger.info(f"   📋 Employee ID range: {EMPLOYEE_DATABASE.index.min()} to {EMPLOYEE_DATABASE.index.max()}")
                return True
            
            # Fallback: try to load from processed data
            processed_path = Path("D:/Employee_Suggestions/employee_suggestion_system/data/processed/latest_processed_data.csv")
            if processed_path.exists():
                logger.info(f"   Loading employee data from: {processed_path}")
                df = pd.read_csv(processed_path)
                
                # Create basic employee database from processed data
                employee_stats = df.groupby('EmployeeId').agg({
                    'SiteId': ['count', 'nunique'],
                    'TaskId': 'nunique',
                    'EmployerId': 'nunique',
                    'WeekDay': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean(),
                }).round(3)
                
                # Flatten column names and create simplified database
                EMPLOYEE_DATABASE = pd.DataFrame({
                    'emp_SiteId_count': employee_stats[('SiteId', 'count')],
                    'emp_SiteId_nunique': employee_stats[('SiteId', 'nunique')],
                    'emp_TaskId_nunique': employee_stats[('TaskId', 'nunique')],
                    'emp_EmployerId_nunique': employee_stats[('EmployerId', 'nunique')],
                    'workload_ratio': employee_stats[('SiteId', 'count')] / employee_stats[('SiteId', 'count')].mean(),
                    'needs_opportunities': (employee_stats[('SiteId', 'count')] < employee_stats[('SiteId', 'count')].mean() * 0.7).astype(int),
                    'is_recent_worker': 1  # Assume all are recent for now
                })
                
                logger.info(f"   ✅ Loaded {len(EMPLOYEE_DATABASE)} employee profiles from processed data")
                logger.info(f"   📋 Employee ID range: {EMPLOYEE_DATABASE.index.min()} to {EMPLOYEE_DATABASE.index.max()}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to load employee database: {str(e)}")
        
        # Last resort: Create demo database but inform user
        logger.warning(f"   ⚠️  Could not load real employee data, using demo data")
        logger.warning(f"   💡 Make sure your data files exist in data/features/ or data/processed/")
        
        EMPLOYEE_DATABASE = pd.DataFrame({
            'EmployeeId': range(1001, 1101),
            'emp_SiteId_count': np.random.randint(10, 100, 100),
            'emp_SiteId_nunique': np.random.randint(2, 15, 100),
            'workload_ratio': np.random.uniform(0.3, 2.0, 100)
        }).set_index('EmployeeId')
        
        logger.info(f"   ⚠️  Using demo employee database with IDs 1001-1100")
        return False

# Initialize model loader
model_loader = ModelLoader()

class PredictionEngine:
    """Handle ML predictions for employee recommendations"""
    
    @staticmethod
    def get_potential_employees(shift_request: ShiftRequest) -> List[int]:
        """Get list of potential employees for the shift"""
        
        # In a real system, this would query your employee database
        # with filters for availability, qualifications, etc.
        
        if len(EMPLOYEE_DATABASE) > 0:
            # Use actual employee IDs from database
            available_employees = EMPLOYEE_DATABASE.index.tolist()
            
            # Filter based on basic criteria (simplified)
            # In production: check availability, qualifications, location, etc.
            qualified_employees = available_employees[:50]  # Limit for performance
            
        else:
            # Demo data
            qualified_employees = list(range(1001, 1051))
        
        logger.info(f"   Found {len(qualified_employees)} potential employees")
        return qualified_employees
    
    @staticmethod
    def create_prediction_features(emp_id: int, shift_request: ShiftRequest) -> Optional[np.ndarray]:
        """Create feature vector for employee-shift combination"""
        
        try:
            # Initialize feature vector
            features = np.zeros(len(FEATURE_COLUMNS))
            
            # Get employee data if available
            employee_data = {}
            if emp_id in EMPLOYEE_DATABASE.index:
                employee_data = EMPLOYEE_DATABASE.loc[emp_id].to_dict()
            
            # Create feature dictionary
            feature_dict = {
                # Employee features (use real data if available, otherwise defaults)
                'emp_SiteId_count': employee_data.get('emp_SiteId_count', np.random.randint(10, 50)),
                'emp_SiteId_nunique': employee_data.get('emp_SiteId_nunique', np.random.randint(2, 10)),
                'emp_TaskId_nunique': employee_data.get('emp_TaskId_nunique', np.random.randint(1, 8)),
                'emp_EmployerId_nunique': np.random.randint(1, 5),
                'emp_years_active': employee_data.get('emp_years_active', np.random.randint(1, 5)),
                'emp_weekly_consistency': employee_data.get('emp_weekly_consistency', np.random.uniform(0.3, 1.0)),
                
                # Interaction features
                'emp_site_experience': np.random.randint(0, 10),
                'emp_task_experience': np.random.randint(0, 15),
                'emp_employer_experience': np.random.randint(0, 8),
                
                # Recency features
                'weeks_since_assignment': np.random.randint(0, 24),
                'is_recent_worker': employee_data.get('is_recent_worker', np.random.choice([0, 1])),
                
                # Workload features
                'workload_ratio': employee_data.get('workload_ratio', np.random.uniform(0.3, 2.0)),
                'needs_opportunities': employee_data.get('needs_opportunities', np.random.choice([0, 1])),
                
                # Temporal features
                'IsWeekend': 1 if shift_request.WeekDay in [6, 7] else 0,
                'Season': (datetime.now().timetuple().tm_yday // 91) % 4,
                'Month': datetime.now().month,
                'Quarter': (datetime.now().month - 1) // 3 + 1,
                
                # Site and task features (simplified)
                'site_EmployeeId': np.random.randint(10, 100),
                'site_TaskId': np.random.randint(2, 20),
                'task_EmployeeId': np.random.randint(5, 50),
                
                # Encoded features
                'WeekDay_encoded': shift_request.WeekDay - 1,
                'Season_encoded': (datetime.now().timetuple().tm_yday // 91) % 4,
                'Month_encoded': datetime.now().month - 1,
                'Quarter_encoded': (datetime.now().month - 1) // 3
            }
            
            # Fill in features
            for i, col in enumerate(FEATURE_COLUMNS):
                if col in feature_dict:
                    features[i] = feature_dict[col]
                elif col.startswith('emp_site_embed_') or col.startswith('emp_task_embed_') or col.startswith('emp_weekday_embed_'):
                    features[i] = np.random.uniform(-0.1, 0.1)  # Random embedding values
                elif 'familiarity' in col or 'success' in col or 'satisfaction' in col:
                    features[i] = np.random.uniform(0.1, 0.9)
                else:
                    features[i] = 0.0
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error creating features for employee {emp_id}: {e}")
            return None
    
    @staticmethod
    def get_model_predictions(features: np.ndarray) -> Dict[str, float]:
        """Get predictions from all trained models"""
        
        if 'main' not in SCALERS:
            raise HTTPException(status_code=500, detail="Scalers not loaded")
        
        # Scale features
        features_scaled = SCALERS['main'].transform(features)
        
        predictions = {}
        
        try:
            # Individual model predictions
            if 'success' in MODELS:
                predictions['success'] = float(MODELS['success'].predict(features_scaled)[0])
            
            if 'satisfaction' in MODELS:
                predictions['satisfaction'] = float(MODELS['satisfaction'].predict(features_scaled)[0])
            
            if 'quality' in MODELS:
                predictions['quality'] = float(MODELS['quality'].predict(features_scaled)[0])
            
            if 'completion' in MODELS:
                predictions['completion'] = float(MODELS['completion'].predict(features_scaled)[0])
            
            # Neural ensemble prediction (optional - fallback if not available)
            if 'neural_ensemble' in MODELS:
                try:
                    neural_preds = MODELS['neural_ensemble'].predict(features_scaled, verbose=0)[0]
                    predictions['neural_success'] = float(neural_preds[0])
                    predictions['neural_satisfaction'] = float(neural_preds[1])
                    predictions['neural_quality'] = float(neural_preds[2])
                    predictions['neural_completion'] = float(neural_preds[3])
                    
                    # Calculate confidence based on agreement between models
                    if all(key in predictions for key in ['success', 'satisfaction', 'quality', 'completion']):
                        agreements = [
                            abs(predictions['success'] - predictions['neural_success']),
                            abs(predictions['satisfaction'] - predictions['neural_satisfaction']),
                            abs(predictions['quality'] - predictions['neural_quality']),
                            abs(predictions['completion'] - predictions['neural_completion'])
                        ]
                        predictions['confidence'] = max(0.0, 1.0 - np.mean(agreements))
                    else:
                        predictions['confidence'] = 0.8
                except Exception as e:
                    logger.warning(f"Neural ensemble prediction failed: {e}")
                    predictions['confidence'] = 0.8
            else:
                # Use confidence based on individual model consistency
                if len(predictions) >= 4:
                    # Calculate confidence based on prediction spread
                    success_val = predictions.get('success', 0.8)
                    completion_val = predictions.get('completion', 0.9)
                    
                    # Higher confidence if predictions are consistent
                    spread = abs(success_val - completion_val)
                    predictions['confidence'] = max(0.6, 1.0 - spread)
                else:
                    predictions['confidence'] = 0.8
                
        except Exception as e:
            logger.error(f"Error in model predictions: {e}")
            raise HTTPException(status_code=500, detail="Prediction error")
        
        return predictions
    
    @staticmethod
    def calculate_overall_score(predictions: Dict[str, float]) -> float:
        """Calculate overall recommendation score"""
        
        # Weights for different factors (could be learned from business data)
        weights = {
            'success': 0.30,      # 30% - Will assignment succeed?
            'completion': 0.25,   # 25% - Will they complete the work?
            'quality': 0.20,      # 20% - Will employer be satisfied?
            'satisfaction': 0.15, # 15% - Will employee be satisfied?
            'confidence': 0.10    # 10% - How confident are we?
        }
        
        # Normalize predictions to 0-1 scale
        normalized_success = max(0, min(1, predictions.get('success', 0.8)))
        normalized_completion = max(0, min(1, predictions.get('completion', 0.9)))
        normalized_quality = max(0, min(1, (predictions.get('quality', 8.0) - 1) / 9))  # Convert 1-10 to 0-1
        normalized_satisfaction = max(0, min(1, (predictions.get('satisfaction', 7.5) - 1) / 9))  # Convert 1-10 to 0-1
        normalized_confidence = max(0, min(1, predictions.get('confidence', 0.8)))
        
        overall_score = (
            normalized_success * weights['success'] +
            normalized_completion * weights['completion'] +
            normalized_quality * weights['quality'] +
            normalized_satisfaction * weights['satisfaction'] +
            normalized_confidence * weights['confidence']
        ) * 100  # Scale to 0-100
        
        return round(overall_score, 2)
    
    @staticmethod
    def generate_explanation(predictions: Dict[str, float]) -> str:
        """Generate human-readable explanation"""
        
        success_pct = int(predictions.get('success', 0.8) * 100)
        completion_pct = int(predictions.get('completion', 0.9) * 100)
        satisfaction = predictions.get('satisfaction', 7.5)
        quality = predictions.get('quality', 8.0)
        confidence_pct = int(predictions.get('confidence', 0.8) * 100)
        
        return f"ML predicts {success_pct}% success rate, {completion_pct}% completion likelihood, " \
               f"{satisfaction:.1f}/10 employee satisfaction, {quality:.1f}/10 employer satisfaction " \
               f"(confidence: {confidence_pct}%)"

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting Employee Recommendation API...")
    
    # Load all models
    model_loader.load_all_models()
    model_loader.load_employee_database()
    
    logger.info("✅ API startup completed")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(MODELS),
        uptime_seconds=uptime
    )

# Model information endpoint
@app.get("/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded models"""
    
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    model_types = {name: type(model).__name__ for name, model in MODELS.items()}
    
    performance = {}
    if 'training_results' in MODEL_METADATA:
        for model_name, results in MODEL_METADATA['training_results'].items():
            if isinstance(results, dict) and 'r2_score' in results:
                performance[model_name] = results['r2_score']
    
    return ModelInfo(
        models=model_types,
        total_features=len(FEATURE_COLUMNS),
        feature_columns=FEATURE_COLUMNS[:20],  # First 20 for brevity
        training_date=MODEL_METADATA.get('metadata', {}).get('training_date', 'unknown'),
        model_performance=performance
    )

# Main prediction endpoint
@app.post("/predict", response_model=RecommendationResponse)
async def predict_recommendations(shift_request: ShiftRequest):
    """Get employee recommendations for a shift"""
    
    start_time = datetime.now()
    
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    logger.info(f"🔮 Generating recommendations for shift: {shift_request.dict()}")
    
    try:
        # Get potential employees
        potential_employees = PredictionEngine.get_potential_employees(shift_request)
        
        recommendations = []
        
        for emp_id in potential_employees:
            # Create features for this employee-shift combination
            features = PredictionEngine.create_prediction_features(emp_id, shift_request)
            
            if features is not None:
                # Get model predictions
                predictions = PredictionEngine.get_model_predictions(features)
                
                # Calculate overall score
                overall_score = PredictionEngine.calculate_overall_score(predictions)
                
                # Get employee info
                employee_info = {}
                if emp_id in EMPLOYEE_DATABASE.index:
                    emp_data = EMPLOYEE_DATABASE.loc[emp_id]
                    employee_info = {
                        'total_assignments': int(emp_data.get('emp_SiteId_count', 0)),
                        'site_versatility': int(emp_data.get('emp_SiteId_nunique', 0)),
                        'task_versatility': int(emp_data.get('emp_TaskId_nunique', 0)),
                        'workload_ratio': float(emp_data.get('workload_ratio', 1.0)),
                        'needs_opportunities': bool(emp_data.get('needs_opportunities', False)),
                        'is_recent_worker': bool(emp_data.get('is_recent_worker', True))
                    }
                
                recommendation = EmployeeRecommendation(
                    EmployeeId=emp_id,
                    overall_score=overall_score,
                    success_probability=predictions.get('success', 0.8),
                    employee_satisfaction_pred=predictions.get('satisfaction', 7.5),
                    employer_satisfaction_pred=predictions.get('quality', 8.0),
                    completion_likelihood=predictions.get('completion', 0.9),
                    confidence=predictions.get('confidence', 0.8),
                    recommendation_reason=PredictionEngine.generate_explanation(predictions),
                    employee_info=employee_info
                )
                
                recommendations.append(recommendation)
        
        # Sort by overall score and take top N
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        top_recommendations = recommendations[:shift_request.top_n]
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Generated {len(top_recommendations)} recommendations in {processing_time:.1f}ms")
        
        return RecommendationResponse(
            status="success",
            timestamp=datetime.now().isoformat(),
            shift_request=shift_request,
            recommendations=top_recommendations,
            total_candidates=len(potential_employees),
            processing_time_ms=processing_time,
            model_version=MODEL_METADATA.get('metadata', {}).get('model_version', '1.0')
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Get available employees endpoint
@app.get("/employees")
async def get_employees(limit: int = 100):
    """Get list of available employees"""
    
    if len(EMPLOYEE_DATABASE) > 0:
        employees = EMPLOYEE_DATABASE.head(limit).reset_index()
        return {
            "status": "success",
            "total_employees": len(EMPLOYEE_DATABASE),
            "employees": employees.to_dict('records')
        }
    else:
        return {
            "status": "success", 
            "total_employees": 100,
            "employees": [{"EmployeeId": i} for i in range(1001, 1001 + limit)]
        }

# System statistics endpoint
@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    stats = {
        "status": "active",
        "uptime_seconds": uptime,
        "uptime_human": str(timedelta(seconds=int(uptime))),
        "models_loaded": len(MODELS),
        "features_count": len(FEATURE_COLUMNS),
        "employees_in_database": len(EMPLOYEE_DATABASE),
        "api_version": "1.0.0",
        "model_info": {
            name: type(model).__name__ for name, model in MODELS.items()
        }
    }
    
    return stats

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Employee Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

def main():
    """Run the FastAPI server"""
    
    print("🚀 Starting Employee Recommendation API Server")
    print("=" * 50)
    print("📡 Server will start on: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 Interactive API: http://localhost:8000/redoc")
    print("💡 Health Check: http://localhost:8000/health")
    
    # Run the server
    uvicorn.run(
        "model_serving:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

if __name__ == "__main__":
    main()