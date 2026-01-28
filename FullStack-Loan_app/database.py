import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "loan_predictions.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_connection()
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
    conn.close()

def save_to_db(data, prediction):
    conn = get_connection()
    cursor = conn.cursor()
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
    conn.close()

def load_db_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM loan_predictions", conn)
    conn.close()
    return df
