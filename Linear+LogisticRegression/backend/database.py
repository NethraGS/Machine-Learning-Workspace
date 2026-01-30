import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area_sqft REAL,
            bedrooms INTEGER,
            bathrooms INTEGER,
            floors INTEGER,
            age_years INTEGER,
            location_score REAL,
            predicted_price REAL,
            sold_probability REAL,
            sold_within_week INTEGER
        )
    """)

    conn.commit()
    conn.close()
