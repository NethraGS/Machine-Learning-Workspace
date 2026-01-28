import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Loan Prediction Dashboard")
st.markdown("### Interactive Loan Approval System with ML")

# ================= SQLITE SETUP =================
conn = sqlite3.connect("loan_predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS loan_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Gender TEXT,
    Married TEXT,
    Dependents TEXT,
    Education TEXT,
    Self_Employed TEXT,
    ApplicantIncome REAL,
    CoapplicantIncome REAL,
    LoanAmount REAL,
    Loan_Amount_Term REAL,
    Credit_History REAL,
    Property_Area TEXT,
    Prediction TEXT,
    Timestamp TEXT
)
""")
conn.commit()

def save_to_db(data, prediction):
    cursor.execute("""
        INSERT INTO loan_predictions (
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Property_Area,
            Prediction, Timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["Gender"],
        data["Married"],
        data["Dependents"],
        data["Education"],
        data["Self_Employed"],
        data["ApplicantIncome"],
        data["CoapplicantIncome"],
        data["LoanAmount"],
        data["Loan_Amount_Term"],
        data["Credit_History"],
        data["Property_Area"],
        prediction,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return joblib.load("loan_pretrained.pkl")

model = load_model()

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("Loan.csv")

df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("📌 Menu")
menu = st.sidebar.radio(
    "Select Option",
    ["📊 View Loan Data", "🔍 Search Loan by ID", "🤖 Predict Loan Approval"]
)

# =====================================================
# 📊 VIEW LOAN DATA
# =====================================================
if menu == "📊 View Loan Data":

    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loans", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("🎨 Loan Records (First 100 Rows)")

    styled_df = (
        df.head(100)
        .style
        .background_gradient(cmap="viridis")
        .set_properties(**{"border": "1px solid white"})
    )

    st.dataframe(styled_df, use_container_width=True)

    st.subheader("📈 Loan Status Distribution")
    st.bar_chart(df["Loan_Status"].value_counts())

# =====================================================
# 🔍 SEARCH LOAN BY ID
# =====================================================
elif menu == "🔍 Search Loan by ID":

    st.subheader("🔍 Find Loan Details")

    loan_id = st.text_input("Enter Loan ID")

    if st.button("Search"):
        loan = df[df["Loan_ID"] == loan_id]

        if loan.empty:
            st.error(f"❌ Loan ID `{loan_id}` not found")
        else:
            st.success("✅ Loan Found")
            st.dataframe(
                loan.style.background_gradient(cmap="plasma"),
                use_container_width=True
            )

# =====================================================
# 🤖 LOAN PREDICTION
# =====================================================
elif menu == "🤖 Predict Loan Approval":

    st.subheader("🤖 Loan Approval Prediction")

    with st.form("loan_form"):

        col1, col2 = st.columns(2)

        with col1:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Married = st.selectbox("Married", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

        with col2:
            ApplicantIncome = st.number_input("Applicant Income", min_value=0)
            CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
            LoanAmount = st.number_input("Loan Amount", min_value=0)
            Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
            Credit_History = st.selectbox("Credit History", [0.0, 1.0])
            Property_Area = st.selectbox(
                "Property Area", ["Urban", "Semiurban", "Rural"]
            )

        submit = st.form_submit_button("🔮 Predict Loan Status")

    if submit:
        input_data = {
            "Gender": Gender,
            "Married": Married,
            "Dependents": Dependents,
            "Education": Education,
            "Self_Employed": Self_Employed,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome": CoapplicantIncome,
            "LoanAmount": LoanAmount,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Credit_History": Credit_History,
            "Property_Area": Property_Area
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        result_text = "Loan Approved" if prediction == 1 else "Loan Rejected"

        # 🔐 SAVE TO SQLITE
        save_to_db(input_data, result_text)

        # 🎉💔 RESULT UI
        if prediction == 1:
            st.balloons()
            st.toast("Loan Approved 🎉", icon="✅")
            st.success("🎉 Congratulations! Your Loan is Approved")
        else:
            st.toast("Loan Rejected 💔", icon="❌")
            st.snow()
            st.error("💔 Sorry! Your Loan is Rejected")

        st.subheader("📥 Submitted Data")
        st.dataframe(
            input_df.assign(Prediction=result_text)
            .style.background_gradient(cmap="coolwarm"),
            use_container_width=True
        )

# ================= FOOTER =================
st.markdown("---")
st.markdown("🚀 Built with **Streamlit + Machine Learning + SQLite**")
