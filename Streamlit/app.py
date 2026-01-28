import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Weather CSV Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# Title
st.title("🌈 Weather Data Dashboard")
# st.markdown("### CSV File Visualization with Colorful Tables & Components")

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("weather_linear_regression_10000.csv")

df = load_data()

# Sidebar
st.sidebar.header("🔍 Filter Options")
selected_columns = st.sidebar.multiselect(
    "Select columns to display",
    df.columns.tolist(),
    default=df.columns.tolist()
)

# Display basic info
st.subheader("📌 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

# Show DataFrame (styled)
st.subheader("🎨 Colorful Data Table")

styled_df = (
    df[selected_columns]
    .head(100)
    .style
    .background_gradient(cmap="viridis")
    .set_properties(**{
        "border": "1px solid white",
        "font-size": "12pt"
    })
)

st.dataframe(styled_df, use_container_width=True)

# Statistical summary
st.subheader("📊 Statistical Summary")
st.dataframe(
    df.describe().style.background_gradient(cmap="plasma"),
    use_container_width=True
)

# Column selector for charts
st.subheader("📈 Data Visualization")

num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
selected_col = st.selectbox("Select a numeric column", num_cols)

# Line chart
st.markdown("#### 📉 Line Chart")
st.line_chart(df[selected_col])

# Histogram
st.markdown("#### 📊 Histogram")
fig, ax = plt.subplots()
ax.hist(df[selected_col], bins=30)
st.pyplot(fig)

# Checkbox interaction
if st.checkbox("Show Raw Data"):
    st.subheader("🧾 Raw CSV Data")
    st.write(df)

# Footer
st.markdown("---")
st.markdown("💡 *Built with Streamlit – Interactive, colorful & simple!*")
