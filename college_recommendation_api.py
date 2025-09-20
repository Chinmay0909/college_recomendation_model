#!/usr/bin/env python3
"""
College Recommendation API Script
This script loads a pretrained H5 model and provides college recommendations.
Can be easily converted to Flask/FastAPI for backend deployment.
"""

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import os
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollegeRecommendationAPI:
    """
    College Recommendation API class that loads pretrained H5 model
    and provides college recommendations based on student inputs.
    """
    
    def __init__(self, model_path: str = 'college_model.h5'):
        """
        Initialize the API with model path
        
        Args:
            model_path (str): Path to the pretrained H5 model file
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
        Load the pretrained H5 model and encoders
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            with h5py.File(self.model_path, 'r') as f:
                # Load branch classes
                branch_classes = [s.decode('utf-8') for s in f['branch_classes'][:]]
                self.branch_encoder.fit(branch_classes)
                
                # Load college classes
                college_classes = [s.decode('utf-8') for s in f['college_classes'][:]]
                self.college_encoder.fit(college_classes)
                
                # Load scaler parameters
                self.scaler.mean_ = f['scaler_mean'][:]
                self.scaler.scale_ = f['scaler_scale'][:]
                
                # Load feature importances
                self.feature_importances = f['feature_importances'][:]
                
                logger.info(f"Model loaded successfully from {self.model_path}")
                logger.info(f"Available branches: {len(branch_classes)}")
                logger.info(f"Available colleges: {len(college_classes)}")
                
                self.is_loaded = True
                return True
                
        except Exception as e:
            logger.error(f"Error loading H5 model: {e}")
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
                
                # Calculate base score using feature importances
                base_score = np.sum(self.feature_importances * X_scaled[0])
                
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
                    'feature_importances': self.feature_importances.tolist()
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

def demo_get_top15():
    """
    Demo function showing how to get top 15 colleges
    """
    print("🎓 DEMO: Getting Top 15 College Recommendations")
    print("=" * 60)
    
    # Initialize the API
    api = CollegeRecommendationAPI()
    
    if not api.is_loaded:
        print("❌ Failed to load model. Please train the model first.")
        print("Run: python simple_h5_model.py")
        return
    
    # Example: Get top 15 recommendations
    result = api.get_recommendations(
        branch="Computer Science Engineering",
        jee_rank=15000,
        cet_rank=20000,
        top_n=15  # This ensures we get exactly 15 recommendations
    )
    
    if result['success']:
        print(f"✅ Successfully generated {len(result['recommendations'])} recommendations!")
        print(f"📊 Student Profile: {result['student_profile']['used_percentile']}% percentile")
        
        print(f"\n🏆 ALL {len(result['recommendations'])} COLLEGE RECOMMENDATIONS:")
        print("=" * 60)
        
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i:2d}. {rec['college']}")
            print(f"    Score: {rec['score']:.4f}")
            if i % 5 == 0:  # Add separator every 5 colleges
                print("    " + "-" * 50)
            print()
    else:
        print(f"❌ Error: {result['error']}")

def main():
    """
    Main function for testing the API
    """
    print("🎓 College Recommendation API - Testing")
    print("=" * 50)
    
    # Initialize the API
    api = CollegeRecommendationAPI()
    
    if not api.is_loaded:
        print("❌ Failed to load model. Please train the model first.")
        print("Run: python simple_h5_model.py")
        return
    
    # Test cases
    test_cases = [
        {
            'branch': 'Computer Science Engineering',
            'jee_rank': 15000,
            'cet_rank': 20000,
            'top_n': 15
        },
        {
            'branch': 'Mechanical Engineering',
            'jee_rank': 25000,
            'cet_rank': 30000,
            'top_n': 15
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}:")
        print(f"   Branch: {test_case['branch']}")
        print(f"   JEE Rank: {test_case['jee_rank']}")
        print(f"   CET Rank: {test_case['cet_rank']}")
        
        # Get recommendations
        result = api.get_recommendations(**test_case)
        
        if result['success']:
            print(f"✅ Success! Generated {len(result['recommendations'])} recommendations")
            print(f"   Student Profile: {result['student_profile']['used_percentile']}% percentile")
            
            print(f"\n🏆 Top {len(result['recommendations'])} Recommendations:")
            for j, rec in enumerate(result['recommendations'], 1):
                print(f"   {j:2d}. {rec['college']} (Score: {rec['score']:.4f})")
                if j % 5 == 0:  # Add separator every 5 colleges
                    print("   " + "-" * 50)
        else:
            print(f"❌ Error: {result['error']}")
    
    # Show model info
    print(f"\n📊 Model Information:")
    model_info = api.get_model_info()
    print(f"   Total Branches: {model_info['total_branches']}")
    print(f"   Total Colleges: {model_info['total_colleges']}")
    print(f"   Model Path: {model_info['model_path']}")
    
    # Run the top 15 demo
    print(f"\n" + "="*60)
    demo_get_top15()

if __name__ == "__main__":
    main()
