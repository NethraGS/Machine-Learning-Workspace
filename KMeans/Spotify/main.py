from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from database import create_table, get_connection

app = FastAPI(title="Spotify KMeans API")

# ---------------- LOAD MODEL ----------------
print("Loading ML model...")
model = joblib.load("spotify_kmeans_pipeline.pkl")
print("Model loaded successfully")

# ---------------- CLUSTER NAMES ----------------
cluster_names = {
    0: "High Energy Party",
    1: "Happy / Dance",
    2: "Chill / Sad",
    3: "Mainstream Mix"
}

FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "duration_ms",
    "popularity"
]

# ---------------- INPUT SCHEMA ----------------
class SongInput(BaseModel):
    danceability: float
    energy: float
    valence: float
    tempo: float
    duration_ms: float
    popularity: float

# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup():
    print("Creating database table...")
    create_table()
    print("Startup completed")

# ---------------- HELPER: LAZY LOAD SONGS ----------------
def get_songs_df():
    """
    Load clustered dataset only when needed
    (prevents slow API startup)
    """
    return pd.read_csv(
        "spotify_clustered.csv",
        usecols=["track_name", "artist", "popularity", "cluster"]
    )

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict_cluster(song: SongInput):

    data = pd.DataFrame([song.dict()])
    cluster = int(model.predict(data)[0])
    cluster_name = cluster_names.get(cluster, "Unknown")

    # Save to DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions
        (danceability, energy, valence, tempo, duration_ms, popularity, cluster, cluster_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        song.danceability,
        song.energy,
        song.valence,
        song.tempo,
        song.duration_ms,
        song.popularity,
        cluster,
        cluster_name
    ))
    conn.commit()
    conn.close()

    return {
        "cluster": cluster,
        "cluster_name": cluster_name
    }

# ---------------- RECOMMEND ----------------
@app.post("/recommend")
def recommend_songs(song: SongInput, top_n: int = 5):

    data = pd.DataFrame([song.dict()])
    cluster = int(model.predict(data)[0])
    cluster_name = cluster_names.get(cluster, "Unknown")

    # ✅ SAVE TO DB HERE
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions
        (danceability, energy, valence, tempo, duration_ms, popularity, cluster, cluster_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        song.danceability,
        song.energy,
        song.valence,
        song.tempo,
        song.duration_ms,
        song.popularity,
        cluster,
        cluster_name
    ))
    conn.commit()
    conn.close()

    songs_df = get_songs_df()

    recommendations = (
        songs_df[songs_df["cluster"] == cluster]
        .sort_values("popularity", ascending=False)
        .head(top_n)
    )

    return {
        "cluster": cluster,
        "cluster_name": cluster_name,
        "recommendations": recommendations.to_dict(orient="records")
    }


# ---------------- HISTORY ----------------
@app.get("/history")
def get_history():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM predictions ORDER BY created_at DESC",
        conn
    )
    conn.close()
    return df.to_dict(orient="records")
