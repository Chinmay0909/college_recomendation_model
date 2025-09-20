#!/usr/bin/env python3
"""
College Recommendation FastAPI Backend
This FastAPI application loads a pretrained PKL model and provides college recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from typing import Dict, List, Optional
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class RecommendationRequest(BaseModel):
    branch: str = Field(..., description="Desired branch/stream")
    jee_rank: int = Field(..., gt=0, description="JEE Main rank")
    cet_rank: int = Field(..., gt=0, description="CET rank")
    top_n: int = Field(15, ge=1, le=50, description="Number of recommendations to return")

class StudentProfile(BaseModel):
    branch: str
    jee_rank: int
    cet_rank: int
    jee_percentile: float
    cet_percentile: float
    used_percentile: float
    used_rank: int

class CollegeRecommendation(BaseModel):
    college: str
    score: float
    base_score: float

class RecommendationResponse(BaseModel):
    success: bool
    student_profile: StudentProfile
    recommendations: List[CollegeRecommendation]
    total_colleges_analyzed: int
    model_info: Dict

class ErrorResponse(BaseModel):
    success: bool
    error: str
    available_branches: Optional[List[str]] = None

class ModelInfo(BaseModel):
    is_loaded: bool
    model_path: str
    total_branches: int
    total_colleges: int
    feature_importances: List[float]

# Global variable for the API instance
api_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global api_instance
    api_instance = CollegeRecommendationAPI()
    if not api_instance.is_loaded:
        logger.error("Failed to load model during startup")
    yield
    logger.info("Shutting down the application")

app = FastAPI(
    title="College Recommendation API",
    description="FastAPI backend for college recommendations based on student profiles",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class CollegeRecommendationAPI:
    """
    College Recommendation API class that loads pretrained PKL model
    and provides college recommendations based on student inputs.
    """
    
    def __init__(self, model_path: str = 'college_model.pkl'):
        """
        Initialize the API with model path
        
        Args:
            model_path (str): Path to the pretrained PKL model file
        """
        self.model_path = model_path
        self.branch_encoder = LabelEncoder()
        self.college_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.is_loaded = False
        
        # Load the model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the pretrained PKL model and encoders
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load all model data from PKL file
            model_data = joblib.load(self.model_path)
            
            # Extract all components
            self.model = model_data['model']
            self.branch_encoder = model_data['branch_encoder']
            self.college_encoder = model_data['college_encoder']
            self.scaler = model_data['scaler']
            self.feature_importances = model_data['feature_importances']
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Available branches: {len(self.branch_encoder.classes_)}")
            logger.info(f"Available colleges: {len(self.college_encoder.classes_)}")
            
            self.is_loaded = True
            return True
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def rank_to_percentile(self, rank: int, total_candidates: int = 1000000) -> float:
        """
        Convert rank to approximate percentile
        
        Args:
            rank (int): Student's rank
            total_candidates (int): Total number of candidates
            
        Returns:
            float: Approximate percentile
        """
        if rank is None or rank <= 0:
            return 0.0
        percentile = ((total_candidates - rank) / total_candidates) * 100
        return max(0.0, min(100.0, percentile))
    
    def calculate_enhanced_score(self, college_name: str, branch: str, 
                               base_score: float) -> float:
        """
        Calculate enhanced score with quality, branch, and location boosts
        
        Args:
            college_name (str): Name of the college
            branch (str): Desired branch
            base_score (float): Base prediction score
            
        Returns:
            float: Enhanced final score
        """
        # 1. College quality boost (based on name patterns)
        quality_boost = 0.0
        if any(keyword in college_name.upper() for keyword in ['IIT', 'NIT', 'BITS', 'IIIT']):
            quality_boost += 0.2
        elif any(keyword in college_name.upper() for keyword in ['INSTITUTE', 'TECHNOLOGY', 'ENGINEERING']):
            quality_boost += 0.1
        
        # 2. Branch relevance boost
        branch_boost = 0.0
        if branch.upper() in college_name.upper():
            branch_boost += 0.1
        
        # 3. Location boost (major cities)
        location_boost = 0.0
        major_cities = ['MUMBAI', 'PUNE', 'DELHI', 'BANGALORE', 'CHENNAI', 'HYDERABAD', 'KOLKATA']
        if any(city in college_name.upper() for city in major_cities):
            location_boost += 0.05
        
        # Calculate final score
        final_score = base_score + quality_boost + branch_boost + location_boost
        return max(0.0, min(1.0, final_score))
    
    def get_recommendations(self, branch: str, jee_rank: int, cet_rank: int, 
                          top_n: int = 15) -> Dict:
        """
        Get college recommendations for a student
        
        Args:
            branch (str): Desired branch
            jee_rank (int): JEE Main rank
            cet_rank (int): CET rank
            top_n (int): Number of top recommendations to return
            
        Returns:
            Dict: Dictionary containing recommendations and metadata
        """
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Model not loaded',
                'recommendations': []
            }
        
        try:
            # Convert ranks to percentiles
            jee_percentile = self.rank_to_percentile(jee_rank)
            cet_percentile = self.rank_to_percentile(cet_rank)
            
            # Use better percentile and rank
            student_percentile = max(jee_percentile, cet_percentile)
            student_rank = min(jee_rank, cet_rank)
            
            # Validate branch
            if branch not in self.branch_encoder.classes_:
                available_branches = list(self.branch_encoder.classes_)
                return {
                    'success': False,
                    'error': f'Branch "{branch}" not found in training data',
                    'available_branches': available_branches[:10],  # Show first 10
                    'recommendations': []
                }
            
            # Encode branch
            branch_encoded = self.branch_encoder.transform([branch])[0]
            
            # Get predictions for all colleges
            college_scores = []
            
            for college_idx in range(len(self.college_encoder.classes_)):
                college_name = self.college_encoder.inverse_transform([college_idx])[0]
                
                # Create input features
                X = np.array([[branch_encoded, student_percentile, student_rank]])
                X_scaled = self.scaler.transform(X)
                
                # Use the actual trained model for prediction
                base_score = self.model.predict(X_scaled)[0]
                
                # Calculate enhanced score
                final_score = self.calculate_enhanced_score(college_name, branch, base_score)
                
                college_scores.append({
                    'college': college_name,
                    'score': float(final_score),
                    'base_score': float(base_score)
                })
            
            # Sort by score and get top N
            college_scores.sort(key=lambda x: x['score'], reverse=True)
            top_recommendations = college_scores[:top_n]
            
            # Prepare response
            response = {
                'success': True,
                'student_profile': {
                    'branch': branch,
                    'jee_rank': jee_rank,
                    'cet_rank': cet_rank,
                    'jee_percentile': round(jee_percentile, 2),
                    'cet_percentile': round(cet_percentile, 2),
                    'used_percentile': round(student_percentile, 2),
                    'used_rank': student_rank
                },
                'recommendations': top_recommendations,
                'total_colleges_analyzed': len(self.college_encoder.classes_),
                'model_info': {
                    'model_path': self.model_path,
                    'feature_importances': self.feature_importances.tolist(),
                    'model_type': 'PKL'
                }
            }
            
            logger.info(f"Generated {len(top_recommendations)} recommendations for {branch}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }
    
    def get_available_branches(self) -> List[str]:
        """
        Get list of available branches
        
        Returns:
            List[str]: List of available branches
        """
        if not self.is_loaded:
            return []
        return list(self.branch_encoder.classes_)
    
    def get_available_colleges(self) -> List[str]:
        """
        Get list of available colleges
        
        Returns:
            List[str]: List of available colleges
        """
        if not self.is_loaded:
            return []
        return list(self.college_encoder.classes_)
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dict: Model information
        """
        return {
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'total_branches': len(self.branch_encoder.classes_) if self.is_loaded else 0,
            'total_colleges': len(self.college_encoder.classes_) if self.is_loaded else 0,
            'feature_importances': self.feature_importances.tolist() if self.is_loaded else []
        }

# FastAPI Endpoints

@app.get("/", summary="Health Check")
async def health_check():
    """Health check endpoint"""
    return {"message": "College Recommendation API is running", "status": "healthy"}

@app.post("/recommendations", response_model=RecommendationResponse, summary="Get College Recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    Get college recommendations based on student profile
    
    - **branch**: Desired branch/stream (e.g., "Computer Science Engineering")
    - **jee_rank**: JEE Main rank (must be > 0)
    - **cet_rank**: CET rank (must be > 0)
    - **top_n**: Number of recommendations to return (1-50, default: 15)
    """
    if not api_instance or not api_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server status.")
    
    result = api_instance.get_recommendations(
        branch=request.branch,
        jee_rank=request.jee_rank,
        cet_rank=request.cet_rank,
        top_n=request.top_n
    )
    
    if not result['success']:
        if 'available_branches' in result:
            raise HTTPException(
                status_code=400, 
                detail=result['error'],
                headers={"available_branches": str(result['available_branches'])}
            )
        else:
            raise HTTPException(status_code=500, detail=result['error'])
    
    return result

@app.get("/branches", response_model=List[str], summary="Get Available Branches")
async def get_available_branches():
    """Get list of all available branches/streams"""
    if not api_instance or not api_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server status.")
    
    return api_instance.get_available_branches()

@app.get("/colleges", response_model=List[str], summary="Get Available Colleges")
async def get_available_colleges():
    """Get list of all available colleges"""
    if not api_instance or not api_instance.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server status.")
    
    return api_instance.get_available_colleges()

@app.get("/model-info", response_model=ModelInfo, summary="Get Model Information")
async def get_model_info():
    """Get information about the loaded model"""
    if not api_instance:
        raise HTTPException(status_code=503, detail="API instance not initialized")
    
    return api_instance.get_model_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
