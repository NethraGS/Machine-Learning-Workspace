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

def load_db_data():
    return pd.read_sql("SELECT * FROM loan_predictions", conn)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return joblib.load("loan_pretrained.pkl")

model = load_model()

# ================= LOAD CSV DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("Loan.csv")

df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("📌 Menu")
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
# 📊 VIEW LOAN DATA (CSV)
# =====================================================
if menu == "📊 View Loan Data":

    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loans", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("🎨 Loan Records (First 100 Rows)")
    st.dataframe(
        df.head(100)
        .style.background_gradient(cmap="viridis"),
        use_container_width=True
    )

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

        save_to_db(input_data, result_text)

        if prediction == 1:
            st.balloons()
            st.success("🎉 Loan Approved")
        else:
            st.snow()
            st.error("💔 Loan Rejected")

        st.dataframe(
            input_df.assign(Prediction=result_text)
            .style.background_gradient(cmap="coolwarm"),
            use_container_width=True
        )

# =====================================================
# 📈 DB ANALYTICS
# =====================================================
elif menu == "📈 DB Analytics":

    st.subheader("📈 Loan Analytics")

    db_df = load_db_data()

    if db_df.empty:
        st.warning("No prediction records found.")
    else:
        # ================= METRICS =================
        approved = (db_df["Prediction"] == "Loan Approved").sum()
        rejected = (db_df["Prediction"] == "Loan Rejected").sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total", len(db_df))
        m2.metric("Approved", approved)
        m3.metric("Rejected", rejected)

        st.divider()

        # ================= ROW 1 =================
        c1, c2 = st.columns(2)

        with c1:
            st.caption("Approval Ratio")
            pie = db_df["Prediction"].value_counts()
            st.pyplot(pie.plot.pie(
                autopct="%1.0f%%",
                ylabel=""
            ).figure)

        with c2:
            st.caption("Credit History Impact")
            credit = pd.crosstab(db_df["Credit_History"], db_df["Prediction"])
            st.bar_chart(credit)

        # ================= ROW 2 =================
        c3, c4 = st.columns(2)

        with c3:
            st.caption("Gender-wise Outcome")
            gender = pd.crosstab(db_df["Gender"], db_df["Prediction"])
            st.bar_chart(gender)

        with c4:
            st.caption("Property Area")
            area = pd.crosstab(db_df["Property_Area"], db_df["Prediction"])
            st.bar_chart(area)

        # ================= ROW 3 =================
        st.caption("Income vs Loan Amount")
        st.scatter_chart(
            db_df,
            x="ApplicantIncome",
            y="LoanAmount",
            size="LoanAmount",
            color="Prediction"
        )

        # ================= DATA (HIDDEN) =================
        with st.expander("🗄️ View Records"):
            st.dataframe(
                db_df.sort_values("Timestamp", ascending=False)
                .style.background_gradient(cmap="Blues"),
                use_container_width=True
            )


# ================= FOOTER =================
st.markdown("---")
st.markdown("🚀 Built with **Streamlit + Machine Learning + SQLite Analytics**")
