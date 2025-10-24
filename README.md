# 🎓 GradSure — Graduation Assurance Risk & Pricing Dashboard

**GradSure** is a Python-based Streamlit web application designed to estimate student graduation assurance risk and calculate actuarial premiums. This project was developed by **Group 9, Bachelor of Science in Actuarial Science III, Makerere University**.

---

## 📌 Features
- **GBM Risk Model**: Uses sponsor and student demographics to compute:
  - Risk Score (1.0–8.5)
  - Risk Tier (Preferred, Standard, Substandard, High Risk)
  - Claim Probability
- **GLM Pricing Model**: Combines financial and academic inputs with GBM outputs to compute:
  - Pure Premium (per semester)
  - Gross Premium (per semester)
  - Premium / Tuition ratio
- **Quote PDF Export**: Download a professional PDF summary combining risk assessment and premium pricing.
- **Interactive Dashboard**: Friendly UI using Streamlit with color-coded tiers and metrics.
- **Session Persistence**: Keeps inputs and outputs within session state for smooth navigation.

---

## 📂 Project Structure
V GROUP 9 DA3 GRADSURE
├─ app.py # Main Streamlit app
├─ risk_assessment_gbm.py # GBM risk model logic
├─ risk_pricing_glm.py # GLM premium pricing logic
├─ simulate_graduation_assurance.py # Simulation script for dataset
├─ gbm_api_server.py # Optional API server for GBM
├─ final_gbm_pipeline.pkl # Pre-trained GBM model
├─ final_glm_pipeline.pkl # Pre-trained GLM model
├─ policies.csv # Sample policies dataset
├─ semester_panel.csv # Sample semester panel dataset
└─ requirements.txt # Python dependencies

---

## ⚙️ Installation & Setup

1. Clone the repository or drag & drop files from GitHub.
2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

3. Install dependencies:
pip install -r requirements.txt

4. Run the Streamlit app:
streamlit run app.py

5. Open the displayed URL in your browser to interact with GradSure.

📊 Usage

Navigate to the Home tab for a project overview and model status.

GBM Risk tab: Enter sponsor & student demographics → compute risk.

GLM Pricing tab: Enter financial & academic details → compute premiums.

Quote tab: Download a professional PDF summarizing risk and pricing.

🛠️ Dependencies

Key Python packages:

streamlit

pandas

numpy

joblib

reportlab

All packages are listed in requirements.txt.

📄 License

This project is for academic purposes only. © 2025 Group 9, BSAS III, Makerere University.

📞 Contact

For questions or suggestions, contact the project team via jonathanmatovu769@gmail.com