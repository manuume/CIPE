import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

st.set_page_config(
    page_title="CIPE Uplift Modeling",
    page_icon="�",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        color: #E2E8F0;
    }

    .stApp {
        background-color: #0F1116;
    }
    
    .header-container {
        background: linear-gradient(145deg, #1e293b 0%, #0f1116 100%);
        border-radius: 12px;
        padding: 2.5rem 1rem;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }

    .title-text {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        font-size: 1.75rem;
        font-style: normal;
        font-weight: 600;
        color: #FFA500;
        text-align: center;
        text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .stSlider, .stSelectbox {
        border-radius: 10px;
        padding: 10px;
        background-color: #1E293B;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    
    .stButton > button {
        border: 2px solid #FFA500;
        border-radius: 10px;
        color: #FFA500;
        padding: 10px 20px;
        background-color: transparent;
        transition: all 0.3s ease-in-out;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #FFA500;
        color: #0F1116;
        border-color: #FFA500;
    }

    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .st-emotion-cache-1g6gooi p {
        font-size: 1.1rem;
        color: #94A3B8;
    }
    
    .st-emotion-cache-1g6gooi div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    model_treat = joblib.load('../models/model_treat.pkl')
    model_ctrl = joblib.load('../models/model_ctrl.pkl')
    return model_treat, model_ctrl

model_treat, model_ctrl = load_models()

# --- Dynamic Header Section ---
st.markdown("""
<div class="header-container">
    <p class="title-text">Customer Influence & Persuasion Engine</p>
    <p class="subtitle-text">Moving Beyond "Who Will Buy?" to "Who Can We Influence?"</p>
</div>
""", unsafe_allow_html=True)


with st.expander("What is this application?"):
    st.write("""
        This tool uses an **Uplift Model** to predict which customers should receive a marketing offer (like a discount).
        Instead of just predicting who will buy, it identifies customers who will buy *only because* they received an offer. This helps maximize marketing ROI by targeting the right people.
    """)


st.header("Analyze a Customer Profile")
st.write("") 

input_container = st.container()
with input_container:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider('Age', 18, 70, 35)
        recency = st.slider('Recency (days since last visit)', 1, 365, 50)
    with col2:
        total_spend = st.slider('Total Spend ($)', 0, 1000, 200)
        pages_viewed = st.slider('Pages Viewed', 1, 100, 15)
    with col3:
        is_new_customer = st.selectbox('Is New Customer?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
        platform = st.selectbox('Platform', ['Android', 'iOS', 'Web'])

    states = ['Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi', 'Gujarat', 'Haryana', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'West Bengal']
    state = st.selectbox('State', states)

st.write("")

if st.button('Analyze Customer'):
    with st.spinner('Calculating uplift score...'):
        time.sleep(1)

        input_data = {'age': [age], 'is_new_customer': [is_new_customer], 'total_spend': [total_spend], 'recency': [recency], 'pages_viewed': [pages_viewed]}
        input_df = pd.DataFrame(input_data)

        platforms_encoded = ['iOS', 'Web']
        for s in states:
            input_df[f'state_{s}'] = 1 if s == state else 0
        for p in platforms_encoded:
            input_df[f'platform_{p}'] = 1 if p == platform else 0

        training_columns = model_treat.get_booster().feature_names
        input_df = input_df.reindex(columns=training_columns, fill_value=0)

        p_treat = model_treat.predict_proba(input_df)[:, 1][0]
        p_ctrl = model_ctrl.predict_proba(input_df)[:, 1][0]
        uplift_score = p_treat - p_ctrl

    st.markdown("---")
    st.header("Analysis Results")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="Uplift Score", value=f"{uplift_score:.4f}")
        with st.expander("See Probabilities"):
            st.write(f"Conversion (with offer): **{p_treat:.2%}**")
            st.write(f"Conversion (no offer): **{p_ctrl:.2%}**")

    with res_col2:
        if uplift_score > 0.1:
            st.success("✅ Persona: Persuadable")
            st.write("**Recommendation:** This customer is highly likely to convert with a discount. **TARGET.**")
            st.write("**Strategy:** This segment represents your highest ROI. Prioritize them for all discount-based campaigns.")
        elif p_ctrl > 0.6:
            st.warning("🤷 Persona: Sure Thing")
            st.write("**Recommendation:** This customer is likely to convert anyway. **Do not waste a discount.**")
            st.write("**Strategy:** Instead of discounts, consider engaging them with loyalty programs or new product announcements to foster long-term value.")
        elif uplift_score < -0.1:
            st.error("🙅 Persona: Do Not Disturb")
            st.write("**Recommendation:** This customer may be annoyed by a discount. **DO NOT TARGET.**")
            st.write("**Strategy:** Exclude this segment from promotional campaigns. Over-targeting them can lead to churn.")
        else:
            st.info("❌ Persona: Lost Cause")
            st.write("**Recommendation:** This customer is unlikely to convert, even with a discount. **Do not target.**")
            st.write("**Strategy:** Marketing spend on this segment has a very low return. Focus your budget on more promising segments.")
