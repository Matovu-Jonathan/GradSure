# ===========================================
# GradSure: GBM Risk Assessment (Final/Fixed)
# ===========================================

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class RiskAssessmentGBM:
    """
    A class to handle the entire lifecycle of the GBM Risk Assessment Model:
    data preparation, training, prediction, and saving/loading.
    
    This model predicts a continuous risk score (0-1) using regression 
    based on demographic features.
    """
    def __init__(self):
        self.gbm_pipeline = None
        self.feature_names = None
        
    def prepare_training_data(self, policies_df, panel_df):
        """Prepare training data by aggregating new risk indicators from panel data."""
        print("🔧 Preparing risk assessment training data...")
        
        # Aggregate panel data to create policy-level risk history
        risk_indicators = panel_df.groupby('policy_id').agg({
            'sponsor_died': 'max',      # Indicator: Did sponsor ever die (1/0)
            'sponsor_shock': 'max',      # Indicator: Did sponsor ever have health shock (1/0)
            'dropout_flag': 'max',      # Indicator: Did student ever drop out (1/0)
            'claim_flag': 'sum',         # Total number of claims
            'tuition_paid_flag': 'mean', # Reliability: Average of tuition paid flags (0.0 to 1.0)
            'semester_number': 'max'     # Policy duration: Max semester reached
        }).reset_index()
        
        risk_indicators.rename(columns={
            'claim_flag': 'total_claims',
            'tuition_paid_flag': 'payment_reliability',
            'semester_number': 'policy_duration'
        }, inplace=True)
        
        # Merge policy demographic data with risk history
        data = policies_df.merge(risk_indicators, on='policy_id', how='left')
        
        # Fill missing values for policies not found in panel (assumed new/no history)
        defaults = {
            'sponsor_died': 0, 'sponsor_shock': 0, 'dropout_flag': 0,
            'total_claims': 0, 'payment_reliability': 1.0, 'policy_duration': 1
        }
        data.fillna(defaults, inplace=True)
        
        return data
    
    def create_risk_target(self, data):
        """Creates a continuous risk score target (0-1) from weighted risk factors."""
        # Weighted composite of risk indicators (This is the key modeling change)
        data['risk_target'] = (
            data['sponsor_died'] * 0.4 +             
            data['sponsor_shock'] * 0.3 +            
            data['dropout_flag'] * 0.2 +             
            # Using (data['total_claims'] > 0) to convert to a binary "claims history" flag (0 or 1)
            (data['total_claims'] > 0) * 0.1         
        )
        
        data['risk_target'] = np.clip(data['risk_target'], 0.0, 1.0)
        
        return data
    
    def train_model(self, policies_df, panel_df):
        """Trains the GBM risk assessment model."""
        print("🚀 Training GBM Risk Assessment Model...")
        data = self.prepare_training_data(policies_df, panel_df)
        data = self.create_risk_target(data)
        
        # Features used for prediction (only applicant characteristics)
        gbm_features = [
            'sponsor_age', 'sponsor_sex', 'sponsor_relationship',
            'student_age', 'student_sex', 'income_band'
        ]
        
        self.feature_names = gbm_features
        X = data[gbm_features]
        y = data['risk_target']
        
        # --- Preprocessing Definition ---
        categorical_features = ['sponsor_sex', 'sponsor_relationship', 'student_sex', 'income_band']
        numerical_features = ['sponsor_age', 'student_age']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                ('num', StandardScaler(), numerical_features) # Scaling numerical features
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # --- Model Pipeline ---
        self.gbm_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                subsample=0.8
            ))
        ])
        
        # Split and Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.gbm_pipeline.fit(X_train, y_train)
        
        # Evaluate using R-squared
        train_score = self.gbm_pipeline.score(X_train, y_train)
        test_score = self.gbm_pipeline.score(X_test, y_test)
        print(f"✅ GBM Training R²: {train_score:.4f}")
        print(f"✅ GBM Testing R²: {test_score:.4f}")
        
        return self
    
    def predict_risk(self, applicant_data):
        """Predict risk assessment for new applicant data (a dictionary)."""
        if self.gbm_pipeline is None or self.feature_names is None:
             raise RuntimeError("Model is not trained. Call train_model() first or load a saved model.")
             
        input_df = pd.DataFrame([applicant_data])
        
        # Predict raw risk score (0-1)
        risk_probability = float(self.gbm_pipeline.predict(input_df[self.feature_names])[0])
        
        # Clip the output
        risk_probability = max(0.0, min(risk_probability, 1.0))
        
        # Convert to outputs
        risk_score_1_5 = 1 + (risk_probability * 4) # Convert 0-1 to 1-5 scale
        risk_tier = self._score_to_tier(risk_probability)
        
        return {
            'risk_score': round(risk_score_1_5, 1),
            'risk_tier': risk_tier,
            'risk_probability': round(risk_probability, 4)
        }
    
    def _score_to_tier(self, risk_score):
        """Convert risk probability (0-1) to categorical tier."""
        if risk_score <= 0.15:
            return "PREFERRED"
        elif risk_score <= 0.35:
            return "STANDARD"  
        elif risk_score <= 0.65:
            return "SUBSTANDARD"
        else:
            return "HIGH_RISK"
    
    # --- CRITICAL FIX: Save as Dictionary ---
    def save_model(self, filename='final_gbm_pipeline.pkl'):
        """Saves the essential components (pipeline and features) as a dictionary."""
        data_to_save = {
            'pipeline': self.gbm_pipeline,
            'features': self.feature_names
        }
        joblib.dump(data_to_save, filename)
        print(f"💾 Risk GBM components saved: {filename}")
        
# -------------------------------------------
# Training Execution Block
# -------------------------------------------
if __name__ == "__main__":
    
    # 1. SETUP AND LOAD DATA 
    # The outer 'try' block was incorrectly placed, causing a SyntaxError.
    
    # Get the base directory dynamically
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

    POLICIES_FILE = os.path.join(BASE_DIR, "policies(python).csv")
    PANEL_FILE = os.path.join(BASE_DIR, "semester_panel(python).csv")

    print(f"🔍 Loading datasets from: {BASE_DIR}")
    try:
        policies = pd.read_csv(POLICIES_FILE)
        panel = pd.read_csv(PANEL_FILE)
        print("✅ Data loaded successfully!")

        # 2. Train Model
        risk_gbm = RiskAssessmentGBM()
        risk_gbm.train_model(policies, panel)
        # Save the new dictionary structure
        risk_gbm.save_model(os.path.join(BASE_DIR, "final_gbm_pipeline.pkl"))
        
        # 3. Example Prediction
        print("\n✨ Running example prediction...")
        example_applicant = {
            'sponsor_age': 35, 
            'sponsor_sex': 'M', 
            'sponsor_relationship': 'FATHER',
            'student_age': 18, 
            'student_sex': 'F', 
            'income_band': 'HIGH'
        }
        
        assessment = risk_gbm.predict_risk(example_applicant)
        print(f"Input: {example_applicant}")
        print(f"Output: {assessment}")

    except FileNotFoundError as e:
        print(f"\n🛑 Error: Data file not found. Please ensure your files are named correctly.")
        print(f"Missing file detail: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during execution: {e}")
