import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="💧 Water Potability Predictor", layout="wide")

# Title
st.title("💧 Water Potability Prediction using Machine Learning")
st.markdown("This app predicts whether water is safe to drink based on chemical properties, especially pH.")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("water_potability (1).csv")
        return df
    except FileNotFoundError:
        st.error("❌ File not found: `water_potability (1).csv`. Please make sure it's uploaded.")
        st.stop()

df = load_data()

# Show dataset preview
if st.checkbox("📊 Show Dataset Preview"):
    st.subheader("Raw Data")
    st.write(df.head(10))

# Sidebar info
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    - This app uses a **Random Forest Classifier**
    - Dataset source: `water_potability (1).csv`
    
    💡 WHO Standard for pH: **6.5 – 8.5**  
    Outside this range, water is likely unsafe to drink.
    """)

# Preprocessing
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Fill missing values
X.fillna(X.mean(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show model performance
st.subheader("🧠 Model Performance")
st.write(f"Model Accuracy: **{accuracy:.2%}**")

# Feature importance
st.markdown("#### 🔍 Top Features Affecting Potability")
importances = model.feature_importances_
feature_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feature_df = feature_df.sort_values(by="Importance", ascending=False)
st.bar_chart(feature_df.set_index("Feature"))

# Input form
st.subheader("🛠️ Enter Water Sample Properties")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input("Hardness", min_value=0.0, value=float(df['Hardness'].mean()))
    solids = st.number_input("Solids", min_value=0.0, value=float(df['Solids'].mean()))

with col2:
    chloramines = st.number_input("Chloramines", min_value=0.0, value=float(df['Chloramines'].mean()))
    sulfate = st.number_input("Sulfate", min_value=0.0, value=float(df['Sulfate'].mean()))
    conductivity = st.number_input("Conductivity", min_value=0.0, value=float(df['Conductivity'].mean()))

with col3:
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=float(df['Organic_carbon'].mean()))
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=float(df['Trihalomethanes'].mean()))
    turbidity = st.number_input("Turbidity", min_value=0.0, value=float(df['Turbidity'].mean()))

# Prepare input
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

# Result display
st.subheader("✅ Prediction Result")
if prediction == 1:
    st.success(f"✔️ This water sample is likely **potable** with {probability:.2%} confidence.")
else:
    st.error(f"✘ This water sample is likely **not potable** with {(1 - probability):.2%} confidence.")

# pH interpretation
st.info(f"""
📌 **pH Interpretation**: 
Your input pH is **{ph:.2f}**.

- 🔴 Below 6.5 → Acidic / Unsafe  
- 🟢 6.5 – 8.5 → Neutral / Safe for drinking  
- 🔵 Above 8.5 → Alkaline / May be unsafe
""")
