import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model (replace 'model.pkl' with your actual saved model file)
# model = joblib.load('model.pkl')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df

df = load_data()

# Sidebar - Title & Info
st.sidebar.title("💧 Water Potability Predictor")
st.sidebar.info("This app predicts whether water is potable based on its chemical properties.")

# Display Raw Data
if st.checkbox("Show Raw Data", value=True):
    st.subheader("📊 Raw Water Quality Data")
    st.dataframe(df.style.highlight_max(axis=0).set_properties(**{'background-color': '#f2f2f2'}))

# Input Section
st.subheader("🔧 Enter Water Parameters")

# Helper function to create sliders
def user_input_features():
    ph = st.slider("pH", float(df['ph'].min()), float(df['ph'].max()), float(df['ph'].mean()))
    hardness = st.slider("Hardness", float(df['Hardness'].min()), float(df['Hardness'].max()), float(df['Hardness'].mean()))
    solids = st.slider("Solids", float(df['Solids'].min()), float(df['Solids'].max()), float(df['Solids'].mean()))
    chloramines = st.slider("Chloramines", float(df['Chloramines'].min()), float(df['Chloramines'].max()), float(df['Chloramines'].mean()))
    sulfate = st.slider("Sulfate", float(df['Sulfate'].min()), float(df['Sulfate'].max()), float(df['Sulfate'].mean()))
    conductivity = st.slider("Conductivity", float(df['Conductivity'].min()), float(df['Conductivity'].max()), float(df['Conductivity'].mean()))
    organic_carbon = st.slider("Organic Carbon", float(df['Organic_carbon'].min()), float(df['Organic_carbon'].max()), float(df['Organic_carbon'].mean()))
    trihalomethanes = st.slider("Trihalomethanes", float(df['Trihalomethanes'].min()), float(df['Trihalomethanes'].max()), float(df['Trihalomethanes'].mean()))
    turbidity = st.slider("Turbidity", float(df['Turbidity'].min()), float(df['Turbidity'].max()), float(df['Turbidity'].mean()))

    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Show input
st.subheader("📥 Your Input Parameters:")
st.write(input_df)

# Dummy prediction (replace with real model)
# prediction = model.predict(input_df)[0]
# prediction_proba = model.predict_proba(input_df)[0]

# Fake output for demo purposes
prediction = np.random.choice([0, 1])
prediction_proba = [0.6, 0.4] if prediction == 0 else [0.4, 0.6]

# Display Prediction
st.subheader("🎯 Prediction Result")

if prediction == 1:
    st.success(f"The water is **potable**! ✅ Probability: {prediction_proba[1]:.2%}")
else:
    st.error(f"The water is **not potable**! ❌ Probability: {prediction_proba[0]:.2%}")

# Optional: Feature Distribution Plots
if st.checkbox("Show Feature Distributions"):
    st.subheader("📈 Feature Distributions")
    selected_feature = st.selectbox("Select feature to visualize", df.columns[:-1])  # exclude target
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax, color='skyblue')
    ax.axvline(input_df[selected_feature][0], color='red', linestyle='--', label='Your Input')
    ax.legend()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit | Dataset Source: Internal")
