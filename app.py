import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))  # preprocessor.pkl için


st.set_page_config(
    page_title="CardioCheck AI - Clinical Decision Support",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        height: 3em;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
        color: #333333; /* <-- BU SATIRI EKLEYİN (Yazıyı koyu gri yapar) */
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_tools():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
    try:
        model = pickle.load(open(os.path.join(base_path, "best_model.pkl"), "rb"))
        pre = pickle.load(open(os.path.join(base_path, "preprocessor.pkl"), "rb"))
        dicts = pickle.load(open(os.path.join(base_path, "dicts.pkl"), "rb"))
        with open(os.path.join(base_path, "model_summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        return model, pre, dicts, summary
    except FileNotFoundError as e:
        st.error(f"System Error: Vital component missing. Details: {e}")
        return None

tools = load_tools()

if tools:
    model, pre, dicts, summary = tools


    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=90)
    with col2:
        st.title("CardioCheck AI: Risk Assessment System")
        st.markdown(f"**Clinical Decision Support Tool** | _Model: {summary['best_model']}_")
        st.info("**Disclaimer:** This interface is designed for research and decision support purposes only. It does not replace professional medical diagnosis.")

    st.markdown("---")

    # --- CLINICAL INPUT FORM ---
    with st.form("patient_form"):
        st.subheader("📋 Patient Clinical Data")
        
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown("##### Demographics")
            age = st.selectbox("Age Category", list(dicts['age_category'].keys()))
            sex = st.selectbox("Biological Sex", ["Male", "Female"])
            race = st.selectbox("Race / Ethnicity", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"])
        
        with c2:
            st.markdown("##### Vitals & Physiology")
            bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 28.5)
            sleep_time = st.number_input("Avg. Sleep Duration (Hours)", 1, 24, 7)
            gen_health = st.selectbox("Self-Reported Health", list(dicts['gen_health'].keys()))
            
        with c3:
            st.markdown("##### Medical History")
            diabetic = st.selectbox("Diabetes Status", ["No", "No, borderline diabetes", "Yes (during pregnancy)", "Yes"])
            asthma = st.selectbox("History of Asthma", ["No", "Yes"])
            kidney = st.selectbox("Kidney Disease", ["No", "Yes"])
            skin_cancer = st.selectbox("History of Skin Cancer", ["No", "Yes"])

        with c4:
            st.markdown("##### Lifestyle & Risks")
            smoking = st.selectbox("Smoker (Lifetime >100 cigs)", ["No", "Yes"])
            alcohol = st.selectbox("Heavy Alcohol Consumption", ["No", "Yes"])
            diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"])
            physical_activity = st.selectbox("Physical Activity (Last 30 Days)", ["No", "Yes"])
            stroke = st.selectbox("History of Stroke", ["No", "Yes"])
            
        st.markdown("---")
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            phys_health = st.slider("Physical Health: Days of 'poor health' in last 30 days", 0, 30, 0)
        with c_p2:
            ment_health = st.slider("Mental Health: Days of 'poor health' in last 30 days", 0, 30, 0)

        submit_btn = st.form_submit_button("RUN RISK ASSESSMENT")


    if submit_btn:

        input_data = pd.DataFrame({
            "BMI": [bmi],
            "Smoking": [1 if smoking == "Yes" else 0],
            "AlcoholDrinking": [1 if alcohol == "Yes" else 0],
            "Stroke": [1 if stroke == "Yes" else 0],
            "PhysicalHealth": [phys_health],
            "MentalHealth": [ment_health],
            "DiffWalking": [1 if diff_walking == "Yes" else 0],
            "Sex": [1 if sex == "Female" else 0], # Mapping: Female=1, Male=0 based on training
            "AgeCategory": [dicts['age_category'][age]],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [1 if physical_activity == "Yes" else 0], # Kept original column name to avoid errors
            "GenHealth": [dicts['gen_health'][gen_health]],
            "SleepTime": [sleep_time],
            "Asthma": [1 if asthma == "Yes" else 0],
            "KidneyDisease": [1 if kidney == "Yes" else 0],
            "SkinCancer": [1 if skin_cancer == "Yes" else 0]
        })


        try:

            # Eğitimdeki ön işlemenin aynısı (kodlayıcılar + ölçekleyiciler, aynı sütun sırası)
            input_data = pre.transform(input_data)


            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]


            st.markdown("### Assessment Results")
            
            res_col1, res_col2 = st.columns([2, 3])
            
            with res_col1:
                if prediction == 1:
                    st.error("HIGH RISK DETECTED")
                    st.markdown(f"""
                        <div class="metric-card" style="border-left: 6px solid #dc3545;">
                            <h3 style="color: #dc3545;">Positive Prediction</h3>
                            <p>The model indicates a high probability of heart disease based on the input parameters.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("LOW RISK DETECTED")
                    st.markdown(f"""
                        <div class="metric-card" style="border-left: 6px solid #28a745;">
                            <h3 style="color: #28a745;">Negative Prediction</h3>
                            <p>The model does not detect significant indicators for heart disease at this time.</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.write("#### Probability Estimate (Confidence)")
                st.progress(float(probability))
                st.caption(f"Model Confidence Score: **{probability*100:.2f}%**")
                
                if probability > 0.5:
                    st.warning("⚠️ **Clinical Note:** Probability exceeds 50%. Further cardiology consultation is recommended.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.write("Please check the input formats.")

with st.sidebar:
    st.header(" Model Specifications")
    if tools:
        m = summary["metrics_default"]
        st.markdown(f"Held-out test set: **{summary['test_size']:,}** records "
                    f"(heart disease rate {summary['positive_rate']:.1%})")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
            st.metric("Recall", f"{m['recall']:.1%}")
        with col_s2:
            st.metric("Precision", f"{m['precision']:.1%}")
            st.metric("F1-Score", f"{m['f1']:.3f}")
        st.caption("Imbalanced data: precision is low by design because the model is tuned to catch most patients.")

        st.markdown("---")
        st.markdown(f"**Algorithm:** {summary['best_model']}")
    st.markdown("**Dataset:** CDC BRFSS 2020 — Personal Key Indicators of Heart Disease")
    
    st.markdown("---")
    st.caption("Applied Informatics Project | v1.0.2")
    st.caption("© 2025 CardioCheck AI")