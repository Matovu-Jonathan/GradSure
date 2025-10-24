# app.py
"""
GradSure — Final integrated Streamlit app
Features:
- Tabs: Home | GBM Risk | GLM Pricing | Quote
- GBM -> GLM flow, auto-run GLM after GBM
- Friendly, color-coded outputs (no JSON)
- PDF export (professional layout) with header/footer and color-coded tier
- Outputs persist via st.session_state
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from typing import Dict, Any

# PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import mm
except Exception:
    pass

# Attempt import for GLM class to help unpickling
try:
    from risk_pricing_glm import PremiumPricingGLM
except Exception:
    pass

st.set_page_config(page_title="GradSure Portal", layout="wide")
st.title("🎓 GradSure")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GBM_PATH = os.path.join(BASE_DIR, "final_gbm_pipeline.pkl")
GLM_PATH = os.path.join(BASE_DIR, "final_glm_pipeline.pkl")

# Tier color mapping
TIER_COLORS = {
    "PREFERRED": "#2ecc71",
    "STANDARD": "#f1c40f",
    "SUBSTANDARD": "#e67e22",
    "HIGH_RISK": "#e74c3c"
}

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_gbm_model(path=GBM_PATH):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load GBM model: {e}")
        return None

@st.cache_resource
def load_glm_model(path=GLM_PATH):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load GLM model: {e}")
        return None

gbm_local = load_gbm_model()
glm_local = load_glm_model()

# -------------------------
# Session state
# -------------------------
if "gbm_input" not in st.session_state: st.session_state.gbm_input = {}
if "gbm_result" not in st.session_state: st.session_state.gbm_result = {}
if "glm_input" not in st.session_state: st.session_state.glm_input = {}
if "glm_result" not in st.session_state: st.session_state.glm_result = {}
if "auto_glm_ran" not in st.session_state: st.session_state.auto_glm_ran = False

# -------------------------
# Helper functions
# -------------------------
def pretty_percent(x):
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "N/A"

def map_tier_color(tier):
    return TIER_COLORS.get(tier, "#95a5a6")

def predict_gbm(demo: Dict[str, Any]) -> Dict[str, Any]:
    if gbm_local is None:
        return {"risk_score": 2.5, "risk_tier": "STANDARD", "risk_probability": 0.3, "note": "GBM model not loaded; defaults used."}
    try:
        if isinstance(gbm_local, dict) and 'pipeline' in gbm_local and 'features' in gbm_local:
            pipeline = gbm_local['pipeline']
            features = gbm_local['features']
            df = pd.DataFrame([demo])
            proba = float(pipeline.predict(df[features])[0])
        else:
            pipeline = gbm_local
            df = pd.DataFrame([demo])
            proba = float(pipeline.predict(df)[0])
        proba = float(np.clip(proba, 0.0, 1.0))
        risk_score = round(1 + proba * 7.5, 1)
        if risk_score <= 1.0:
            tier = "PREFERRED"
        elif risk_score <= 2.5:
            tier = "STANDARD"
        elif risk_score <= 4.5:
            tier = "SUBSTANDARD"
        else:
            tier = "HIGH_RISK"
        return {"risk_score": risk_score, "risk_tier": tier, "risk_probability": round(proba, 4)}
    except Exception as e:
        return {"error": f"GBM prediction failed: {e}"}

def predict_glm(full_input: Dict[str, Any], risk_info: Dict[str, Any]) -> Dict[str, Any]:
    if glm_local is None:
        return {"error": "GLM model not loaded."}
    try:
        return glm_local.predict_premium(full_input, risk_info)
    except Exception as e:
        return {"error": f"GLM prediction failed: {e}"}

def build_pdf_bytes(applicant: Dict[str, Any], gbm: Dict[str, Any], glm: Dict[str, Any]) -> bytes:
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story = []

        # Header
        story.append(Paragraph("GradSure – Group 9 BSAS III (Makerere University)", styles["Title"]))
        story.append(Paragraph("Graduation Assurance Premium & Risk Quote", styles["Normal"]))
        story.append(Spacer(1, 8))

        # Applicant block
        story.append(Paragraph("<b>Applicant Details</b>", styles["Heading3"]))
        applicant_table_data = [
            ["Sponsor Relationship", str(applicant.get("sponsor_relationship", "N/A")).replace("_", " ").title()],
            ["Sponsor Age / Sex", f"{applicant.get('sponsor_age','N/A')} / {applicant.get('sponsor_sex','N/A')}"],
            ["Student Age / Sex", f"{applicant.get('student_age','N/A')} / {applicant.get('student_sex','N/A')}"],
            ["Tuition (per semester)", f"UGX {int(applicant.get('tuition_per_sem',0)):,}"],
            ["Study Duration (years)", f"{applicant.get('expected_years','N/A')}"],
            ["Scholarship", f"{applicant.get('scholarship_flag','N/A')}"],
            ["Income band", f"{applicant.get('income_band','N/A')}"]
        ]
        t = Table(applicant_table_data, colWidths=[110*mm, 60*mm])
        t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),("BOX",(0,0),(-1,-1),0.5,colors.grey)]))
        story.append(t)
        story.append(Spacer(1, 12))

        # Risk block
        tier = gbm.get("risk_tier","N/A")
        tier_color = map_tier_color(tier)
        story.append(Paragraph("<b>Risk Assessment</b>", styles["Heading3"]))
        risk_table_data = [
            ["Risk Score (1–8.5)", str(gbm.get("risk_score","N/A"))],
            ["Risk Tier", tier],
            ["Claim Probability", f"{pretty_percent(gbm.get('risk_probability',0))}"]
        ]
        tr = Table(risk_table_data, colWidths=[110*mm, 60*mm])
        tr.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),("BOX",(0,0),(-1,-1),0.5,colors.grey)]))
        story.append(tr)
        story.append(Spacer(1, 8))

        # Pricing block
        story.append(Paragraph("<b>Premium Pricing</b>", styles["Heading3"]))
        price_table_data = [
            ["Pure premium (per semester)", f"UGX {int(glm.get('pure_premium',0)):,}"],
            ["Gross premium (per semester)", f"UGX {int(glm.get('gross_premium',0)):,}"],
            ["Expected claims (full study period)", str(glm.get("expected_claims_lifetime","N/A"))],
            ["Premium / Tuition", f"{glm.get('premium_ratio',0)*100:.2f}%"]
        ]
        tp = Table(price_table_data, colWidths=[110*mm, 60*mm])
        tp.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),("BOX",(0,0),(-1,-1),0.5,colors.grey)]))
        story.append(tp)
        story.append(Spacer(1, 12))

        # Notes
        story.append(Paragraph("<b>Notes & Interpretation</b>", styles["Heading4"]))
        notes = "This quote combines a GBM-based risk assessment and a GLM actuarial pricing model. " \
                "Risk tier guides underwriting decisions; gross premium includes a loading for expenses/profit."
        story.append(Paragraph(notes, styles["Normal"]))
        story.append(Spacer(1, 20))

        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("© 2025 GradSure – Group 9 BSAS III (Makerere University)", styles["Normal"]))

        doc.build(story)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        return f"GradSure Quote\n\nError generating PDF: {e}".encode("utf-8")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Home", "GBM Risk", "GLM Pricing", "Quote"])

# ----- HOME -----
with tabs[0]:
    st.header("🎓 Welcome to GradSure")
    st.markdown(
        """
**GradSure – Group 9 BSAS III (Makerere University)**

GradSure estimates student graduation assurance risk and actuarial premiums using two complementary models:
- **GBM Risk Model** — demographic-only input (sponsor & student) producing a **risk score (1.0–8.5)**, a **risk tier**, and a **claim probability**.
- **GLM Pricing Model** — uses financial/academic inputs combined with the GBM outputs to compute **pure** and **gross** premiums.

**Suggested flow**
1. Run **GBM Risk** with demographics (one-time entry).  
2. GLM will auto-run after GBM completes; or go to **GLM Pricing** to supply financial details.  
3. Open **Quote** to download a branded PDF summary.
"""
    )
    col1, col2 = st.columns(2)
    with col1:
        if gbm_local is None: st.error("GBM model not loaded (final_gbm_pipeline.pkl not found).")
        else: st.success("GBM model loaded ✅")
    with col2:
        if glm_local is None: st.error("GLM model not loaded (final_glm_pipeline.pkl not found).")
        else: st.success("GLM model loaded ✅")
    st.markdown("---")
    st.caption("© 2025 GradSure – Group 9 BSAS III (Makerere University)")

# ----- GBM RISK -----
with tabs[1]:
    st.header("⚠️ GBM Risk Assessment — Demographics Only")
    st.markdown("Enter sponsor & student demographics (this will be used for both risk and pricing).")
    with st.form("gbm_form"):
        demo = {
            "sponsor_age": st.number_input("Sponsor Age", 18, 80, 40, key="gbm_sponsor_age"),
            "sponsor_sex": st.selectbox("Sponsor Sex", ["M", "F"], key="gbm_sponsor_sex"),
            "sponsor_relationship": st.selectbox("Sponsor Relationship", ["parent", "grandparent", "other_relative", "other"], key="gbm_sponsor_rel"),
            "student_age": st.number_input("Student Age", 10, 30, 20, key="gbm_student_age"),
            "student_sex": st.selectbox("Student Sex", ["M", "F"], key="gbm_student_sex"),
            "income_band": st.selectbox("Income Band", ["low", "medium", "high"], key="gbm_income_band")
        }
        if st.form_submit_button("Run GBM Risk"):
            st.session_state.gbm_input = demo
            st.session_state.gbm_result = predict_gbm(demo)
            st.session_state.auto_glm_ran = False

    if st.session_state.gbm_result:
        res = st.session_state.gbm_result
        if "error" in res: st.error(res["error"])
        else:
            st.success("Risk assessment complete")
            c1, c2, c3 = st.columns([2,2,2])
            with c1: st.metric("Risk Score (1–8.5)", res.get("risk_score", "N/A"))
            with c2:
                tier = res.get("risk_tier", "N/A")
                color = map_tier_color(tier)
                st.markdown(f"<div style='padding:6px;border-radius:6px;background:{color};width:fit-content;color:#ffffff;font-weight:700;text-align:center'>{tier}</div>", unsafe_allow_html=True)
            with c3: st.metric("Claim Probability", pretty_percent(res.get("risk_probability", 0)))
            st.markdown("**Interpretation**")
            notes_map = {
                "PREFERRED": "Low predicted claim likelihood. Good candidate for preferential pricing.",
                "STANDARD": "Average claim likelihood — typical profile.",
                "SUBSTANDARD": "Above-average risk — consider underwriting attention.",
                "HIGH_RISK": "High risk — may require stricter terms or higher premiums."
            }
            st.info(notes_map.get(tier, "GBM output unavailable."))
    st.markdown("---")
    st.caption("© 2025 GradSure – Group 9 BSAS III (Makerere University)")

# ----- GLM PRICING -----
with tabs[2]:
    st.header("🧮 GLM Premium Pricing — Financial Inputs")
    st.markdown("Enter financial & academic inputs. Demographics and risk from GBM are used automatically if GBM has been run.")
    with st.form("glm_form"):
        tuition_per_sem = st.number_input("Tuition per Semester (UGX)", 100_000, 20_000_000, 2_400_000, step=100_000, key="glm_tuition")
        expected_years = st.number_input("Expected Years", 1, 10, 4, key="glm_years")
        scholarship_flag = st.selectbox("Scholarship Flag (0/1)", [0, 1], key="glm_scholar")
        demographics = st.session_state.gbm_input.copy() if st.session_state.gbm_input else {}
        submit_glm = st.form_submit_button("Compute Premium (GLM)")
        if submit_glm:
            risk_info = st.session_state.gbm_result if st.session_state.gbm_result else {"risk_score":2.5,"risk_tier":"STANDARD","risk_probability":0.3}
            full_input = {
                "tuition_per_sem": tuition_per_sem,
                "expected_years": expected_years,
                "scholarship_flag": scholarship_flag,
                "income_band": demographics.get("income_band"),
                "sponsor_age": demographics.get("sponsor_age"),
                "sponsor_sex": demographics.get("sponsor_sex"),
                "sponsor_relationship": demographics.get("sponsor_relationship"),
                "student_age": demographics.get("student_age"),
                "student_sex": demographics.get("student_sex"),
                "risk_score": risk_info.get("risk_score"),
                "risk_tier": risk_info.get("risk_tier"),
                "risk_probability": risk_info.get("risk_probability")
            }
            glm_out = predict_glm(full_input, risk_info)
            st.session_state.glm_input = full_input
            st.session_state.glm_result = glm_out
            st.session_state.auto_glm_ran = True

    if st.session_state.glm_result:
        out = st.session_state.glm_result
        if "error" in out: st.error(out["error"])
        else:
            st.success("GLM pricing complete")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Pure Premium (UGX/semester)", f"{int(out.get('pure_premium',0)):,}")
            with c2: st.metric("Gross Premium (UGX/semester)", f"{int(out.get('gross_premium',0)):,}")
            with c3:
                ratio = out.get("premium_ratio")
                st.metric("Premium / Tuition", f"{ratio*100:.2f}%" if ratio else "N/A")
            st.markdown("**Details**")
            st.write(f"- Expected claims (full study period): {out.get('expected_claims_lifetime','N/A')}")
            st.write(f"- Expected claim amount (full study period): UGX {int(out.get('expected_claim_amount_lifetime',0)):,}" if out.get("expected_claim_amount_lifetime") else "-")
            st.info("Pure premium is the actuarial cost. Gross premium includes a loading for expenses/profit.")
    st.markdown("---")
    st.caption("© 2025 GradSure – Group 9 BSAS III (Makerere University)")

# ----- QUOTE -----
with tabs[3]:
    st.header("🔗 Complete Quote Summary")
    st.markdown("Consolidated summary of the risk assessment and GLM premium. Use the button below to download a branded PDF.")
    if not st.session_state.gbm_result and not st.session_state.glm_result:
        st.info("Run GBM Risk and GLM Pricing first to generate a quote.")
    else:
        # Merge GBM + GLM inputs
        applicant = {}
        if st.session_state.gbm_input: applicant.update(st.session_state.gbm_input)
        if st.session_state.glm_input: applicant.update(st.session_state.glm_input)
        gbm = st.session_state.gbm_result
        glm = st.session_state.glm_result
        pdf_bytes = build_pdf_bytes(applicant, gbm, glm)
        st.download_button("📄 Download Quote PDF", pdf_bytes, file_name="GradSure_Quote.pdf", mime="application/pdf")
