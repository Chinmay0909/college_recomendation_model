#!/usr/bin/env python3
"""
Simple PKL College Recommendation Model
Creates a machine learning model for college recommendations and saves it to PKL format.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from cutoff_data_parser import CutoffDataParser, initialize_cutoff_data

class SimplePKLModel:
    def __init__(self):
        self.model = None
        self.branch_encoder = LabelEncoder()
        self.college_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.cutoff_data = None
        self.is_trained = False
        
    def load_cutoff_data(self):
        """Load cutoff data from CSV files"""
        print("Loading cutoff data...")
        parser, self.cutoff_data = initialize_cutoff_data()
        
        if self.cutoff_data.empty:
            print("No cutoff data available!")
            return False
            
        print(f"Loaded {len(self.cutoff_data)} records")
        return True
    
    def prepare_training_data(self):
        """Prepare training data from cutoff data"""
        if self.cutoff_data is None or self.cutoff_data.empty:
            if not self.load_cutoff_data():
                return None
        
        training_data = []
        
        for _, row in self.cutoff_data.iterrows():
            branch = row['branch']
            college = row['college']
            percentile = row['percentile']
            rank = row['rank']
            
            if pd.isna(percentile) or pd.isna(rank) or not branch or not college:
                continue
            
            # Create training samples with different student profiles
            student_percentiles = [percentile + i for i in range(0, 15, 3)]  # 0-12% above cutoff
            student_ranks = [max(1, rank - i*2000) for i in range(0, 15, 3)]  # Better ranks
            
            for student_pct, student_rank in zip(student_percentiles, student_ranks):
                if student_pct > 100:
                    student_pct = 100
                
                # Calculate scores
                admission_prob = min(1.0, (student_pct - percentile + 10) / 20)
                admission_prob = max(0.0, admission_prob)
                quality_score = percentile / 100.0
                overall_score = (0.6 * admission_prob) + (0.4 * quality_score)
                
                training_data.append({
                    'branch': branch,
                    'college': college,
                    'student_percentile': student_pct,
                    'student_rank': student_rank,
                    'college_percentile': percentile,
                    'college_rank': rank,
                    'overall_score': overall_score
                })
        
        return pd.DataFrame(training_data)
    
    def train_model(self):
        """Train the model"""
        df = self.prepare_training_data()
        if df is None or df.empty:
            print("No training data available!")
            return False
        
        print(f"Prepared {len(df)} training samples")
        
        # Encode categorical variables
        df['branch_encoded'] = self.branch_encoder.fit_transform(df['branch'])
        df['college_encoded'] = self.college_encoder.fit_transform(df['college'])
        
        # Prepare features and target
        X = df[['branch_encoded', 'student_percentile', 'student_rank']].values
        y = df['overall_score'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Calculate R² score
        y_pred = self.model.predict(X_scaled)
        r2 = self.model.score(X_scaled, y)
        
        print(f"Model trained successfully!")
        print(f"R² Score: {r2:.4f}")
        
        self.is_trained = True
        return True
    
    def save_pkl_model(self, filename='college_model.pkl'):
        """Save model data to PKL file"""
        if not self.is_trained:
            print("Model not trained yet!")
            return False
        
        # Prepare all model data for saving
        model_data = {
            'model': self.model,
            'branch_encoder': self.branch_encoder,
            'college_encoder': self.college_encoder,
            'scaler': self.scaler,
            'feature_importances': self.model.feature_importances_,
            'is_trained': self.is_trained
        }
        
        # Save everything to a single PKL file
        joblib.dump(model_data, filename)
        
        # Save some sample predictions for validation
        sample_branches = self.branch_encoder.transform(self.branch_encoder.classes_[:5])
        sample_percentiles = np.array([85.0, 90.0, 95.0, 80.0, 88.0])
        sample_ranks = np.array([15000, 10000, 5000, 20000, 12000])
        
        sample_X = np.column_stack([sample_branches, sample_percentiles, sample_ranks])
        sample_X_scaled = self.scaler.transform(sample_X)
        sample_predictions = self.model.predict(sample_X_scaled)
        
        print(f"Model saved to {filename}")
        print(f"Sample predictions: {sample_predictions}")
        return True
    
    def load_pkl_model(self, filename='college_model.pkl'):
        """Load model from PKL file"""
        try:
            model_data = joblib.load(filename)
            
            # Load all components
            self.model = model_data['model']
            self.branch_encoder = model_data['branch_encoder']
            self.college_encoder = model_data['college_encoder']
            self.scaler = model_data['scaler']
            self.feature_importances = model_data['feature_importances']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filename}")
            print(f"Available branches: {len(self.branch_encoder.classes_)}")
            print(f"Available colleges: {len(self.college_encoder.classes_)}")
            return True
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_colleges(self, branch, jee_rank, cet_rank, top_n=15):
        """Predict college recommendations"""
        if not self.is_trained:
            print("Model not trained or loaded!")
            return []
        
        # Convert ranks to percentiles
        jee_percentile = self.rank_to_percentile(jee_rank)
        cet_percentile = self.rank_to_percentile(cet_rank)
        
        # Use better percentile and rank
        student_percentile = max(jee_percentile, cet_percentile)
        student_rank = min(jee_rank, cet_rank)
        
        print(f"Student Profile:")
        print(f"  Branch: {branch}")
        print(f"  JEE Rank: {jee_rank} (Percentile: {jee_percentile:.2f}%)")
        print(f"  CET Rank: {cet_rank} (Percentile: {cet_percentile:.2f}%)")
        print(f"  Using: {student_percentile:.2f}% percentile, Rank {student_rank}")
        
        # Encode branch
        if branch not in self.branch_encoder.classes_:
            print(f"Branch '{branch}' not found in training data")
            return []
        
        branch_encoded = self.branch_encoder.transform([branch])[0]
        
        # Get predictions for all colleges
        college_scores = []
        
        print(f"🔍 Analyzing {len(self.college_encoder.classes_)} colleges...")
        
        for college_idx in range(len(self.college_encoder.classes_)):
            college_name = self.college_encoder.inverse_transform([college_idx])[0]
            
            # Create input
            X = np.array([[branch_encoded, student_percentile, student_rank]])
            X_scaled = self.scaler.transform(X)
            
            # Predict base score
            base_score = self.model.predict(X_scaled)[0]
            
            # Enhance scoring with additional factors
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
            final_score = min(1.0, final_score)  # Cap at 1.0
            
            college_scores.append((college_name, final_score))
        
        # Sort by score and return top N
        college_scores.sort(key=lambda x: x[1], reverse=True)
        return college_scores[:top_n]
    
    def rank_to_percentile(self, rank, total_candidates=1000000):
        """Convert rank to approximate percentile"""
        if rank is None or rank <= 0:
            return 0.0
        percentile = ((total_candidates - rank) / total_candidates) * 100
        return max(0.0, min(100.0, percentile))

def main():
    """Main function to create and test the PKL model"""
    print("Creating Simple PKL College Recommendation Model")
    print("=" * 50)
    
    # Create model
    model = SimplePKLModel()
    
    # Train model
    if model.train_model():
        # Save PKL model
        model.save_pkl_model()
        
        # Test the model
        print("\n" + "=" * 30)
        print("TESTING THE MODEL")
        print("=" * 30)
        
        test_cases = [
            ("Computer Science Engineering", 15000, 20000),
            ("Mechanical Engineering", 25000, 30000),
            ("Electronics and Telecommunication Engineering", 10000, 15000)
        ]
        
        for branch, jee_rank, cet_rank in test_cases:
            print(f"\nStudent: {branch}, JEE Rank: {jee_rank}, CET Rank: {cet_rank}")
            recommendations = model.predict_colleges(branch, jee_rank, cet_rank, top_n=15)
            
            if recommendations:
                print("Top 15 recommendations:")
                for i, (college, score) in enumerate(recommendations, 1):
                    print(f"  {i:2d}. {college} (Score: {score:.4f})")
            else:
                print("  No recommendations found")
    else:
        print("Failed to train model!")

if __name__ == "__main__":
    main()
