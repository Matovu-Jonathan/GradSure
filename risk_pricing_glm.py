# ===========================================
# GradSure: GLM Premium Pricing Training (Final)
# ===========================================
# Trains Frequency (Poisson/NB) and Severity (Gamma) GLMs.
# The output model file is final_glm_pipeline.pkl, which is used by the API server.
#
# NOTE: This script uses synthetic 'risk_score' and 'risk_tier' data during training
# to mimic the upstream GBM output, allowing the pricing models to be trained.
# ===========================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib
import os
import sys

class PremiumPricingGLM:
    """
    Class to encapsulate the GLM frequency and severity models.
    The entire instance is saved and loaded by the API server.
    """
    def __init__(self):
        self.freq_glm = None
        self.sev_glm = None
        self.freq_features = None
        self.sev_features = None
        # Mappings are stored inside the class for consistent application at prediction time
        self.ordinal_mappings = {'income_band': {'low': 0, 'medium': 1, 'high': 2}}
        self.nominal_vars = ['sponsor_sex', 'student_sex', 'sponsor_relationship']
        self.severity_mean_fallback = 0.0 # Fallback for severity if no claims in training

    def prepare_training_data(self, policies_df, panel_df):
        """Prepare data for premium modeling by aggregating policy-level claims."""
        print("🔧 Preparing premium modeling data...")
        
        # Aggregate claims data (total claims and total claimed amount per policy)
        claims_data = panel_df.groupby('policy_id').agg({
            'claim_flag': 'sum',
            'claim_amount': 'sum'
        }).reset_index()
        
        claims_data.rename(columns={
            'claim_flag': 'total_claims',
            'claim_amount': 'total_claim_amount'
        }, inplace=True)
        
        # Merge with policy data
        data = policies_df.merge(claims_data, on='policy_id', how='left')
        data.fillna({'total_claims': 0, 'total_claim_amount': 0}, inplace=True)
        
        return data
    
    def train_models(self, policies_df, panel_df):
        """Train frequency and severity GLMs."""
        print("🚀 Training Premium Pricing GLMs...")
        
        data = self.prepare_training_data(policies_df, panel_df)
        
        # ENCODING
        data_encoded = data.copy()
        data_encoded = pd.get_dummies(data_encoded, columns=self.nominal_vars, drop_first=True)
        data_encoded['income_band_encoded'] = data_encoded['income_band'].map(self.ordinal_mappings['income_band'])
        
        # Convert risk tier to encoded (0-3) for use in the model
        tier_mapping = {'PREFERRED': 0, 'STANDARD': 1, 'SUBSTANDARD': 2, 'HIGH_RISK': 3}
        data_encoded['risk_tier_encoded'] = data_encoded['risk_tier'].map(tier_mapping)

        # GLM FEATURES: Financial/Academic + Risk inputs
        base_features = [
            # Financial/Academic factors
            'tuition_per_sem', 'expected_years', 'scholarship_flag',
            'income_band_encoded',
            # Risk inputs (from GBM)
            'risk_score', 'risk_tier_encoded' 
        ]
        
        # Add encoded categoricals
        categorical_features = [c for c in data_encoded.columns if c.startswith(
            ('sponsor_sex_', 'student_sex_', 'sponsor_relationship_')
        )]
        
        self.freq_features = base_features + categorical_features
        self.sev_features = base_features + categorical_features
        
        # --- 1. Train Frequency Model (Poisson/Negative Binomial) ---
        X_freq = data_encoded[self.freq_features].copy()
        # Align booleans/categoricals to be integers if needed
        for col in X_freq.select_dtypes(include=['bool']).columns:
            X_freq[col] = X_freq[col].astype(int)
            
        X_freq = sm.add_constant(X_freq)
        y_freq = data_encoded['total_claims']
        
        # Fit initial Poisson GLM
        poisson_glm = sm.GLM(
            y_freq, X_freq, 
            family=sm.families.Poisson(link=sm.families.links.Log()) # FIX 1: Replaced .log() with .Log()
        ).fit()

        # Check for overdispersion and switch to Negative Binomial if necessary
        # Dispersion > 1.5 is a common heuristic for switching from Poisson
        dispersion_ratio = y_freq.var() / y_freq.mean()
        print(f"\nFrequency data check: mean={y_freq.mean():.4f}, variance={y_freq.var():.4f}, ratio={dispersion_ratio:.2f}")

        if dispersion_ratio > 1.5:
            print("⚠️ Overdispersion detected (Ratio > 1.5). Switching to Negative Binomial.")
            self.freq_glm = sm.GLM(
                y_freq, X_freq, 
                family=sm.families.NegativeBinomial(link=sm.families.links.Log()) # FIX 2: Replaced .log() with .Log()
            ).fit()
            print("✅ Frequency GLM trained (Negative Binomial)")
        else:
            self.freq_glm = poisson_glm
            print("✅ Frequency GLM trained (Poisson)")
        
        # --- 2. Train Severity Model (Gamma) - only policies with claims ---
        severity_data = data_encoded[data_encoded['total_claim_amount'] > 0]
        
        if len(severity_data) > 10:
            X_sev = severity_data[self.sev_features].copy()
            # Align booleans/categoricals to be integers if needed
            for col in X_sev.select_dtypes(include=['bool']).columns:
                X_sev[col] = X_sev[col].astype(int)

            X_sev = sm.add_constant(X_sev)
            y_sev = severity_data['total_claim_amount']
            
            self.sev_glm = sm.GLM(
                y_sev, X_sev,
                family=sm.families.Gamma(link=sm.families.links.Log()) # FIX 3: Replaced .log() with .Log()
            ).fit()
            self.severity_mean_fallback = y_sev.mean()
            print(f"✅ Severity GLM trained (Gamma). Fallback mean: {self.severity_mean_fallback:,.0f}")
        else:
            self.sev_glm = None
            self.severity_mean_fallback = 1000000 # Default hardcoded fallback
            print(f"⚠️ Insufficient claims for severity model. Using hardcoded fallback: {self.severity_mean_fallback:,.0f}")
        
        return self
    
    def predict_premium(self, applicant_data, risk_assessment):
        """
        Calculates premium using financial data + risk assessment.
        The result is spread over the total policy duration to yield a per-semester premium.

        applicant_data: dictionary of policy features (financial/demographic)
        risk_assessment: dictionary from GBM output {'risk_score', 'risk_tier', ...}
        """
        # Combine all inputs
        input_data = {**applicant_data, **risk_assessment}
        
        # Prepare input DataFrame
        input_df = pd.DataFrame([input_data])
        
        # --- Encoding ---
        # 1. Nominal (One-Hot)
        input_df = pd.get_dummies(input_df, columns=self.nominal_vars, drop_first=True)
        # 2. Ordinal (Income Band)
        input_df['income_band_encoded'] = input_df['income_band'].map(self.ordinal_mappings['income_band'])
        # 3. Risk Tier
        tier_mapping = {'PREFERRED': 0, 'STANDARD': 1, 'SUBSTANDARD': 2, 'HIGH_RISK': 3}
        input_df['risk_tier_encoded'] = input_df['risk_tier'].map(tier_mapping)
        
        # --- Align Features ---
        # Ensure all columns used during training (self.freq_features) are present.
        X_input = input_df.reindex(columns=self.freq_features, fill_value=0)
        X_input = sm.add_constant(X_input, has_constant='add')
        
        # --- Prediction ---
        # Frequency (Claims per policy lifetime)
        expected_frequency = self.freq_glm.predict(X_input).iloc[0]
        
        # Severity (Claim amount | Claim > 0, over policy lifetime)
        if self.sev_glm is not None:
            expected_severity = self.sev_glm.predict(X_input).iloc[0]
        else:
            expected_severity = self.severity_mean_fallback
        
        # --- Premium Calculation ---
        pure_premium_lifetime = expected_frequency * expected_severity
        gross_premium_lifetime = pure_premium_lifetime * 1.25 # 25% Loading

        # Spread the lifetime premium across all semesters for billing
        # Assume 2 semesters per expected year for the payment period
        total_semesters = applicant_data['expected_years'] * 2
        
        pure_premium = pure_premium_lifetime / total_semesters # Per Semester Premium
        gross_premium = gross_premium_lifetime / total_semesters # Per Semester Premium
        
        return {
            # Final output for API/billing
            'pure_premium': float(pure_premium),
            'gross_premium': float(gross_premium),

            # Lifetime values for audit/reference
            'pure_premium_lifetime': float(pure_premium_lifetime),
            'gross_premium_lifetime': float(gross_premium_lifetime),

            'expected_claims_lifetime': float(expected_frequency),
            'expected_claim_amount_lifetime': float(expected_severity),
            'premium_ratio': float(pure_premium / applicant_data['tuition_per_sem']) 
        }
    
    def save_model(self, filename='final_glm_pipeline.pkl'):
        """Saves the entire class instance for stable loading."""
        joblib.dump(self, filename)
        print(f"💾 Premium GLM saved: {filename}")

# Training execution
if __name__ == "__main__":
    # Get the base directory dynamically
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

    POLICIES_FILE = os.path.join(BASE_DIR, "policies(python).csv")
    PANEL_FILE = os.path.join(BASE_DIR, "semester_panel(python).csv")

    try:
        policies = pd.read_csv(POLICIES_FILE)
        panel = pd.read_csv(PANEL_FILE)
        print("✅ Data loaded successfully for GLM training!")
        
        # --- CRITICAL: Add Synthetic GBM Outputs for Training ---
        policies['risk_score'] = np.random.uniform(1, 5, len(policies))
        policies['risk_tier'] = np.random.choice(['PREFERRED', 'STANDARD', 'SUBSTANDARD', 'HIGH_RISK'], len(policies))
        
        pricing_glm = PremiumPricingGLM()
        pricing_glm.train_models(policies, panel)
        pricing_glm.save_model(os.path.join(BASE_DIR, "final_glm_pipeline.pkl"))
        
        print("\n✨ Running example premium calculation...")
        # Example inputs: financial + demographic
        example_premium_input = {
            'sponsor_age': 45, 'sponsor_sex': 'M', 'sponsor_relationship': 'MOTHER',
            'student_age': 20, 'student_sex': 'F', 'income_band': 'medium',
            'tuition_per_sem': 2400000, 'expected_years': 4, 'scholarship_flag': 0,
        }
        # Example GBM output (must align with the format the API will pass)
        example_risk_output = {
             'risk_score': 2.2,
             'risk_tier': 'STANDARD',
             'risk_probability': 0.3 # Probability is ignored by GLM, but included here for context
        }
        
        result = pricing_glm.predict_premium(example_premium_input, example_risk_output)
        print(f"Input Risk: {example_risk_output['risk_tier']}")
        print("\n--- LIFETIME POLICY VALUES (Reference) ---")
        print(f"Expected Claims (Lifetime): {result['expected_claims_lifetime']:.4f}")
        print(f"Expected Claim Amount (Lifetime): UGX {result['expected_claim_amount_lifetime']:,.0f}")
        print(f"Pure Premium (Lifetime): UGX {result['pure_premium_lifetime']:,.0f}")
        print(f"Gross Premium (Lifetime): UGX {result['gross_premium_lifetime']:,.0f}")
        print("\n--- PER SEMESTER PREMIUM (Billing) ---")
        print(f"Pure Premium (Per Semester): UGX {result['pure_premium']:,.0f}")
        print(f"Gross Premium (Per Semester): UGX {result['gross_premium']:,.0f}")
        
    except FileNotFoundError as e:
        print(f"\n🛑 Error: Data file not found for GLM. Detail: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during GLM execution: {e}")