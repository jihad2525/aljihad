import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Page configuration
st.set_page_config(page_title="💧 Water Potability Checker", layout="centered")

# Title and description
st.title("💧 Water Potability Prediction App")
st.markdown("""
This app predicts whether water is **potable (safe to drink)** based on its chemical properties.
Special attention is given to **pH levels**, since neutral pH (6.5–8.5) is essential for safe drinking water.
""")

# Load data with error handling
@st.cache_data
def load_data():
    file_path = "water_potability.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: `{file_path}`. Please make sure it's in the same folder as this app.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")
        st.stop()

df = load_data()

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Train model
X = df.drop(columns=["Potability"])
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sidebar info
with st.sidebar:
    st.header("📘 About This App")
    st.markdown("""
    - Uses **Random Forest Classifier**
    - Accuracy: **{:.2%}**
    
    💡 WHO Guidelines:
    - **pH 6.5 – 8.5** → Safe for drinking
    - Too acidic or alkaline? ❌ Not safe!
    """.format(accuracy))

# Input form
st.subheader("🛠️ Enter Water Sample Properties")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0)
    solids = st.number_input("Solids (ppm)", min_value=0.0, value=20000.0)

with col2:
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=250.0)
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=500.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=15.0)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=60.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=3.0)

# Prepare input data
input_data = pd.DataFrame({
    'ph': [ph],
    'Hardness': [hardness],
    'Solids': [solids],
    'Chloramines': [chloramines],
    'Sulfate': [sulfate],
    'Conductivity': [conductivity],
    'Organic_carbon': [organic_carbon],
    'Trihalomethanes': [trihalomethanes],
    'Turbidity': [turbidity]
})

# Make prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Display result
st.subheader("✅ Prediction Result")

if prediction == 1:
    st.success(f"✔️ This water sample is likely **potable** with {probability:.2%} confidence.")
else:
    st.error(f"✘ This water sample is likely **not potable** with {(1 - probability):.2%} confidence.")

# pH interpretation
st.info(f"""
📌 **pH Interpretation**: 
Your input pH is **{ph:.2f}**.  
According to WHO standards:
- 🔴 Below 6.5 → Acidic / Unsafe  
- 🟢 6.5 – 8.5 → Neutral / **Safe for drinking**  
- 🔵 Above 8.5 → Alkaline / May be unsafe
""")

# Show dataset preview (optional)
if st.checkbox("📊 Show Dataset Sample"):
    st.write(df.head(10))
