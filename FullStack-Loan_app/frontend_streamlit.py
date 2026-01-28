import streamlit as st
import pandas as pd
import requests

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Loan Prediction Dashboard")
st.markdown("### Interactive Loan Approval System with ML")

# ================= LOAD CSV =================
@st.cache_data
def load_csv():
    return pd.read_csv("Loan.csv")

df = load_csv()

# ================= SIDEBAR =================
menu = st.sidebar.radio(
    "Select Option",
    [
        "📊 View Loan Data",
        "🔍 Search Loan by ID",
        "🤖 Predict Loan Approval",
        "📈 DB Analytics"
    ]
)

# =====================================================
# 📊 VIEW LOAN DATA
# =====================================================
if menu == "📊 View Loan Data":

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loans", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(
        df.head(100).style.background_gradient(cmap="viridis"),
        use_container_width=True
    )

    st.bar_chart(df["Loan_Status"].value_counts())

# =====================================================
# 🔍 SEARCH LOAN BY ID
# =====================================================
elif menu == "🔍 Search Loan by ID":

    loan_id = st.text_input("Enter Loan ID")

    if st.button("Search"):
        loan = df[df["Loan_ID"] == loan_id]
        if loan.empty:
            st.error("❌ Loan ID not found")
        else:
            st.success("✅ Loan Found")
            st.dataframe(loan.style.background_gradient(cmap="plasma"))

# =====================================================
# 🤖 LOAN PREDICTION
# =====================================================
elif menu == "🤖 Predict Loan Approval":

    with st.form("loan_form"):
        c1, c2 = st.columns(2)

        with c1:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Married = st.selectbox("Married", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

        with c2:
            ApplicantIncome = st.number_input("Applicant Income", min_value=0)
            CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
            LoanAmount = st.number_input("Loan Amount", min_value=0)
            Loan_Amount_Term = st.number_input("Loan Term", min_value=0)
            Credit_History = st.selectbox("Credit History", [0.0, 1.0])
            Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submit = st.form_submit_button("🔮 Predict Loan Status")

    if submit:
        payload = {
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

        res = requests.post("http://127.0.0.1:5000/predict", json=payload)
        result = res.json()["prediction"]

        if result == "Loan Approved":
            st.balloons()
            st.success("🎉 Loan Approved")
        else:
            st.snow()
            st.error("💔 Loan Rejected")

# =====================================================
# 📈 DB ANALYTICS
# =====================================================
elif menu == "📈 DB Analytics":

    res = requests.get("http://127.0.0.1:5000/db-data")
    db_df = pd.DataFrame(res.json())

    if db_df.empty:
        st.warning("No data found")
    else:
        a = (db_df["Prediction"] == "Loan Approved").sum()
        r = (db_df["Prediction"] == "Loan Rejected").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(db_df))
        c2.metric("Approved", a)
        c3.metric("Rejected", r)

        st.bar_chart(pd.crosstab(db_df["Gender"], db_df["Prediction"]))
        st.bar_chart(pd.crosstab(db_df["Property_Area"], db_df["Prediction"]))

        st.scatter_chart(
            db_df,
            x="ApplicantIncome",
            y="LoanAmount",
            color="Prediction"
        )
