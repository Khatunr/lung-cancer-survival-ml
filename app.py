import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Lung Cancer Survival Dashboard",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Lung Cancer 5-Year Survival Dashboard")

st.warning(
    "Educational prototype only. "
    "Not for clinical decision-making."
)

# File locations
MODEL_PATH = Path(
    r"C:\Users\ruman\Documents\lung-cancer-survival-ml\Assignment 6\Notebooks\lung_cancer_survival_pipeline.joblib"
)

DATA_PATH = Path(
    r"C:\Users\ruman\Documents\lung-cancer-survival-ml\Assignment 6\data\Lung_cancer_data.csv.csv"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Explore Cohort",
    "Demo Prediction"
])


# TAB 1
with tab1:

    st.header("Project Overview")

    st.write("""
    This application demonstrates a machine learning
    workflow for lung cancer survival group classification.

    The model is an educational prototype and should
    not be used for medical decisions.
    """)

    st.subheader("Important Limitations")

    st.write("""
    - Small dataset
    - Possible class imbalance
    - Limited external validation
    - Predictions are not clinical recommendations
    """)


# TAB 2
with tab2:

    st.header("Explore Patient Dataset")

    try:

        data = load_data()

        st.success("Dataset loaded successfully!")

        st.metric("Total Patients", len(data))

        st.subheader("Dataset Preview")

        st.dataframe(data.head(20), use_container_width=True)

        st.subheader("Dataset Statistics")

        st.write(data.describe(include="all"))

        if "Diagnosis Age" in data.columns:

            st.subheader("Patient Age Distribution")

            st.bar_chart(
                data["Diagnosis Age"].dropna().value_counts().sort_index()
            )

    except Exception as e:

        st.error(f"Dataset Error: {e}")


# TAB 3
with tab3:

    st.header("Demo Prediction")

    st.caption(
        "Educational demonstration only. "
        "Not intended for patient care."
    )

    try:

        model = load_model()

        age = st.slider(
            "Diagnosis Age", 18, 100, 65
        )

        sex = st.selectbox(
            "Sex", ["Female", "Male"]
        )

        radiation = st.selectbox(
            "Radiation Therapy", ["No", "Yes"]
        )

        mutation = st.number_input(
            "Mutation Count",
            min_value=0,
            max_value=5000,
            value=100
        )

        fraction = st.number_input(
            "Fraction Genome Altered",
            min_value=0.0,
            max_value=1.0,
            value=0.25
        )

        stage = st.selectbox(
            "AJCC Stage",
            [
                "Stage I",
                "Stage II",
                "Stage III",
                "Stage IV"
            ]
        )

        if st.button("Estimate Survival Group"):

            row = pd.DataFrame([{
                "Diagnosis Age": age,
                "Sex": sex,
                "Radiation Therapy": radiation,
                "Mutation Count": mutation,
                "Fraction Genome Altered": fraction,
                "Stage_Simplified": stage
            }])

            prediction = model.predict(row)[0]

            st.success(
                f"Model Output: {prediction}"
            )

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(row)[0]

                st.write({
                    str(c): round(float(p), 3)
                    for c, p in zip(model.classes_, probabilities)
                })

    except FileNotFoundError:

        st.error(
            "Model file not found. Please check MODEL_PATH."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")
