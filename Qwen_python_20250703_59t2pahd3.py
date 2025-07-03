import streamlit as st

# Page config
st.set_page_config(page_title="💧 Water Potability Checker", layout="centered")

# Title and description
st.title("💧 Water Potability Checker")
st.markdown("""
This app determines whether water is **potable (safe to drink)** based on its chemical properties.
It uses **rule-based criteria**, especially focusing on **pH**, which should be between **6.5 and 8.5** for safe drinking water.
""")

# Sidebar info
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    This app checks if water is safe to drink using WHO standards:
    
    | Parameter        | Acceptable Range         |
    |------------------|--------------------------|
    | pH               | **6.5 – 8.5** (Neutral)  |
    | Hardness         | < 200 mg/L               |
    | Turbidity        | < 5 NTU                  |
    | Chloramines      | < 4 ppm                  |
    | Sulfate          | < 250 mg/L               |
    """)

# Input form
st.subheader("🛠️ Enter Water Sample Properties")

col1, col2 = st.columns(2)

with col1:
    ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=3.0)
    
with col2:
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=2.0)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=200.0)

# Evaluation logic
is_potable = True
reasons = []

if not (6.5 <= ph <= 8.5):
    is_potable = False
    reasons.append(f"🔴 pH ({ph:.2f}) out of safe range (6.5–8.5)")

if hardness > 200:
    is_potable = False
    reasons.append(f"🔴 Hardness ({hardness} mg/L) too high (>200 mg/L)")

if turbidity > 5:
    is_potable = False
    reasons.append(f"🔴 Turbidity ({turbidity} NTU) too high (>5 NTU)")

if chloramines > 4:
    is_potable = False
    reasons.append(f"🔴 Chloramines ({chloramines} ppm) too high (>4 ppm)")

if sulfate > 250:
    is_potable = False
    reasons.append(f"🔴 Sulfate ({sulfate} mg/L) too high (>250 mg/L)")

# Display result
st.subheader("✅ Final Assessment")

if is_potable:
    st.success("✔️ This water sample is **potable** (safe to drink).")
else:
    st.error("✘ This water sample is **not potable** (unsafe to drink).")
    st.markdown("#### Reasons:")
    for reason in reasons:
        st.markdown(f"- {reason}")

# pH interpretation
st.info(f"""
📌 **pH Interpretation**: 
Your input pH is **{ph:.2f}**.

- 🔴 Below 6.5 → Acidic / Unsafe  
- 🟢 6.5 – 8.5 → Neutral / Safe for drinking  
- 🔵 Above 8.5 → Alkaline / May be unsafe
""")

# Optional: Show tips
if st.checkbox("💡 Show Tips for Improving Water Quality"):
    st.markdown("""
    - Use filtration systems to reduce hardness and turbidity
    - Monitor pH regularly and use neutralizers if needed
    - Test for contaminants like sulfates and chlorine compounds
    """)
