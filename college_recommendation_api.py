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
from cutoff_data_parser import CutoffDataParser, initialize_cutoff_data

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
    
    def __init__(self, model_path: str = 'models/college_recommendation_model.pkl'):
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
        self.cutoff_data = None
        
        # Load the model and cutoff data on initialization
        self.load_model()
        self.load_cutoff_data()
    
    def load_model(self) -> bool:
        """
        Load the pretrained PKL model and encoders from separate files
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if individual model files exist
            model_files = {
                'model': 'models/college_recommendation_model.pkl',
                'branch_encoder': 'models/branch_encoder.pkl',
                'college_encoder': 'models/college_encoder.pkl',
                'scaler': 'models/scaler.pkl'
            }
            
            # Load each component separately
            for component, file_path in model_files.items():
                if not os.path.exists(file_path):
                    logger.warning(f"Model file not found: {file_path}")
                    continue
                
                if component == 'model':
                    self.model = joblib.load(file_path)
                elif component == 'branch_encoder':
                    self.branch_encoder = joblib.load(file_path)
                elif component == 'college_encoder':
                    self.college_encoder = joblib.load(file_path)
                elif component == 'scaler':
                    self.scaler = joblib.load(file_path)
            
            # Check if we have the essential components
            if self.model is not None and hasattr(self.model, 'feature_importances_'):
                self.feature_importances = self.model.feature_importances_
            else:
                self.feature_importances = np.array([])
            
            # For cutoff-based recommendations, we don't strictly need the ML model
            # but we'll mark as loaded if we have the encoders or if cutoff data is available
            if (self.branch_encoder is not None and self.college_encoder is not None) or self.cutoff_data is not None:
                self.is_loaded = True
                logger.info("Model components loaded successfully")
                if self.branch_encoder is not None:
                    logger.info(f"Available branches: {len(self.branch_encoder.classes_)}")
                if self.college_encoder is not None:
                    logger.info(f"Available colleges: {len(self.college_encoder.classes_)}")
                return True
            else:
                logger.warning("No model components loaded, will rely on cutoff data only")
                self.is_loaded = True  # Still mark as loaded since we can work with cutoff data
                return True
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Even if model loading fails, we can still work with cutoff data
            self.is_loaded = True
            return True
    
    def load_cutoff_data(self) -> bool:
        """
        Load cutoff data from CSV files
        
        Returns:
            bool: True if cutoff data loaded successfully, False otherwise
        """
        try:
            parser, self.cutoff_data = initialize_cutoff_data()
            
            if self.cutoff_data.empty:
                logger.error("No cutoff data available!")
                return False
                
            logger.info(f"Loaded {len(self.cutoff_data)} cutoff records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading cutoff data: {e}")
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
    
    def calculate_recommendation_score(self, college_name: str, branch: str, 
                                     student_percentile: float, student_rank: int,
                                     cutoff_percentile: float, cutoff_rank: int) -> float:
        """
        Calculate recommendation score based on multiple factors
        
        Args:
            college_name (str): Name of the college
            branch (str): Desired branch
            student_percentile (float): Student's percentile
            student_rank (int): Student's rank
            cutoff_percentile (float): College's cutoff percentile
            cutoff_rank (int): College's cutoff rank
            
        Returns:
            float: Recommendation score (0-1)
        """
        score = 0.0
        
        # 1. Eligibility margin (how much better the student is than cutoff)
        if not pd.isna(cutoff_percentile):
            margin_percentile = student_percentile - cutoff_percentile
            eligibility_score = min(1.0, max(0.0, margin_percentile / 10.0))  # Normalize to 0-1
            score += 0.4 * eligibility_score
        
        if not pd.isna(cutoff_rank):
            margin_rank = cutoff_rank - student_rank
            rank_score = min(1.0, max(0.0, margin_rank / 10000.0))  # Normalize to 0-1
            score += 0.3 * rank_score
        
        # 2. College quality boost (based on name patterns)
        quality_boost = 0.0
        if any(keyword in college_name.upper() for keyword in ['IIT', 'NIT', 'BITS', 'IIIT']):
            quality_boost += 0.2
        elif any(keyword in college_name.upper() for keyword in ['INSTITUTE', 'TECHNOLOGY', 'ENGINEERING']):
            quality_boost += 0.1
        
        # 3. Branch relevance boost
        branch_boost = 0.0
        if branch.upper() in college_name.upper():
            branch_boost += 0.1
        
        # 4. Location boost (major cities)
        location_boost = 0.0
        major_cities = ['MUMBAI', 'PUNE', 'DELHI', 'BANGALORE', 'CHENNAI', 'HYDERABAD', 'KOLKATA']
        if any(city in college_name.upper() for city in major_cities):
            location_boost += 0.05
        
        # 5. Cutoff competitiveness (higher cutoff = more prestigious)
        competitiveness_score = 0.0
        if not pd.isna(cutoff_percentile):
            competitiveness_score = cutoff_percentile / 100.0
        elif not pd.isna(cutoff_rank):
            competitiveness_score = max(0.0, 1.0 - (cutoff_rank / 100000.0))
        
        # Combine all factors
        final_score = score + (0.1 * quality_boost) + (0.05 * branch_boost) + (0.05 * location_boost) + (0.1 * competitiveness_score)
        
        return max(0.0, min(1.0, final_score))
    
    def get_recommendations(self, branch: str, jee_rank: int, cet_rank: int, 
                          top_n: int = 15) -> Dict:
        """
        Get college recommendations for a student based on actual cutoff data
        
        Args:
            branch (str): Desired branch
            jee_rank (int): JEE Main rank
            cet_rank (int): CET rank
            top_n (int): Number of top recommendations to return
            
        Returns:
            Dict: Dictionary containing recommendations and metadata
        """
        if not self.is_loaded or self.cutoff_data is None or self.cutoff_data.empty:
            return {
                'success': False,
                'error': 'Model or cutoff data not loaded',
                'recommendations': []
            }
        
        try:
            # Convert ranks to percentiles
            jee_percentile = self.rank_to_percentile(jee_rank)
            cet_percentile = self.rank_to_percentile(cet_rank)
            
            # Use better percentile and rank
            student_percentile = max(jee_percentile, cet_percentile)
            student_rank = min(jee_rank, cet_rank)
            
            # Filter cutoff data for the specific branch
            branch_data = self.cutoff_data[
                self.cutoff_data['branch'].str.contains(branch, case=False, na=False)
            ].copy()
            
            if branch_data.empty:
                # Try to find similar branches
                all_branches = self.cutoff_data['branch'].unique()
                similar_branches = [b for b in all_branches if any(word in b.lower() for word in branch.lower().split())]
                
                return {
                    'success': False,
                    'error': f'Branch "{branch}" not found in cutoff data',
                    'available_branches': similar_branches[:10] if similar_branches else list(all_branches)[:10],
                    'recommendations': []
                }
            
            # Find eligible colleges based on cutoff data
            eligible_colleges = []
            
            for _, row in branch_data.iterrows():
                college_name = row['college']
                cutoff_percentile = row['percentile']
                cutoff_rank = row['rank']
                
                # Check if student is eligible (student should have better percentile/rank than cutoff)
                is_eligible = False
                if not pd.isna(cutoff_percentile) and student_percentile >= cutoff_percentile:
                    is_eligible = True
                elif not pd.isna(cutoff_rank) and student_rank <= cutoff_rank:
                    is_eligible = True
                
                if is_eligible:
                    # Calculate recommendation score based on multiple factors
                    score = self.calculate_recommendation_score(
                        college_name, branch, student_percentile, student_rank,
                        cutoff_percentile, cutoff_rank
                    )
                    
                    eligible_colleges.append({
                        'college': college_name,
                        'score': score,
                        'cutoff_percentile': cutoff_percentile,
                        'cutoff_rank': cutoff_rank,
                        'margin_percentile': student_percentile - cutoff_percentile if not pd.isna(cutoff_percentile) else None,
                        'margin_rank': cutoff_rank - student_rank if not pd.isna(cutoff_rank) else None
                    })
            
            # Remove duplicates and sort by score
            unique_colleges = {}
            for college in eligible_colleges:
                college_name = college['college']
                if college_name not in unique_colleges or college['score'] > unique_colleges[college_name]['score']:
                    unique_colleges[college_name] = college
            
            # Sort by score and get top N
            sorted_colleges = sorted(unique_colleges.values(), key=lambda x: x['score'], reverse=True)
            top_recommendations = sorted_colleges[:top_n]
            
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
                'total_colleges_analyzed': len(unique_colleges),
                'model_info': {
                    'model_path': self.model_path,
                    'feature_importances': self.feature_importances.tolist() if hasattr(self, 'feature_importances') else [],
                    'model_type': 'Cutoff-based'
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
        Get list of available branches from cutoff data
        
        Returns:
            List[str]: List of available branches
        """
        if self.cutoff_data is None or self.cutoff_data.empty:
            return []
        return sorted(self.cutoff_data['branch'].unique().tolist())
    
    def get_available_colleges(self) -> List[str]:
        """
        Get list of available colleges from cutoff data
        
        Returns:
            List[str]: List of available colleges
        """
        if self.cutoff_data is None or self.cutoff_data.empty:
            return []
        return sorted(self.cutoff_data['college'].unique().tolist())
    
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
