import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="Water Potability App", layout="wide")

st.title("💧 Water Potability Prediction")
st.markdown("This app loads and displays the Water Potability dataset.")

# Function to load data with error handling
@st.cache_data
def load_data():
    file_path = "water_potability.csv"
    
    # Debugging info (uncomment if needed)
    # st.write("Current working directory:", os.getcwd())
    # st.write("Files available:", os.listdir())

    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found: `{file_path}`. Please make sure the file exists in the correct location.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        st.stop()

# Load the dataset
df = load_data()

# Show preview of the dataframe
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Optional: Show basic statistics
st.subheader("🧮 Basic Statistics")
st.write(df.describe())

# Optional: Show missing values
st.subheader("❌ Missing Values")
st.write(df.isnull().sum())

# Optional: Plot histogram for any numeric column
st.subheader("📈 Histogram of Numeric Columns")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)

if selected_col:
    st.bar_chart(df[selected_col].dropna())