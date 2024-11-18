import json
import boto3
from fastapi.staticfiles import StaticFiles
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, Body, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import requests
import os

os.environ["MINIO_ENDPOINT"] = "http://s3:9000"
os.environ["MINIO_ACCESS_KEY"] = "minio"
os.environ["MINIO_SECRET_KEY"] = "minio123"
MINIO_BUCKET = "data"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

# Initialize MinIO client
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# Function to load a file locally or from MinIO
def load_file(key, local_path):
    if os.path.exists(local_path):
        print(f"Loading {key} from local storage...")
        return local_path
    else:
        print(f"{key} not found locally, attempting to download from MinIO...")
        try:
            response = s3_client.get_object(Bucket=MINIO_BUCKET, Key=key)
            with open(local_path, 'wb') as file:
                file.write(response['Body'].read())
            print(f"Downloaded {key} from MinIO.")
        except Exception as e:
            print(f"Could not download {key} from MinIO: {e}")
            raise FileNotFoundError(f"{key} not found in MinIO or locally.")
        return local_path

def load_json(key, local_path):
    local_file = load_file(key, local_path)
    with open(local_file, 'r') as file:
        return json.load(file)


def load_model_and_embeddings():
    try:
        # Ensure model path exists
        model_path = load_file('models/challenger_spotify_model.h5', 'files/challenger_spotify_model.h5')

        # Load TensorFlow model
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully from:", model_path)

        # Load embeddings
        embeddings_dict = load_json('data_info/song_embeddings.json', 'files/song_embeddings.json')
        embeddings = np.array(embeddings_dict["embeddings"])
        print(f"Loaded {len(embeddings)} embeddings successfully.")

        # Load scaler
        scaler_dict = load_json('data_info/spotify_scaler.json', 'files/spotify_scaler.json')
        scaler_mean = np.array(scaler_dict["mean"])
        scaler_range = np.array(scaler_dict["range"])
        numerical_columns = scaler_dict["features"]
        print("Scaler loaded successfully.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise HTTPException(status_code=500, detail=f"Required file missing: {e}")
    except Exception as e:
        print(f"Error loading model or data: {e}")
        raise HTTPException(status_code=500, detail="Error loading model or embeddings.")
    
    return model, embeddings, scaler_mean, scaler_range, numerical_columns

model, embeddings, scaler_mean, scaler_range, numerical_columns = load_model_and_embeddings()
if model is None or embeddings is None:
    raise RuntimeError("Failed to load the model or embeddings. Check logs for details.")

def load_song_dataset():
    try:
        response = s3_client.get_object(Bucket=MINIO_BUCKET, Key='spotify/spotify_songs.csv')
        df = pd.read_csv(response['Body'])
        return df
    except Exception as e:
        print(f"Error loading song dataset: {e}")
        return None

song_dataset = load_song_dataset()


class SongFeatures(BaseModel):
    danceability: float
    energy: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: float
    year: int

app = FastAPI()

@app.get("/")
async def read_root():
    return JSONResponse({"message": "Welcome to the Spotify Recommender API"})

@app.post("/predict/")
def recommend_song(features: SongFeatures):
    """
    Recommend songs based on cosine similarity of embeddings.
    """
    if model is None or embeddings is None:
        raise HTTPException(status_code=500, detail="Model or embeddings not loaded")

    # Convert input features to embedding
    input_df = pd.DataFrame([features.dict()])
    input_array = input_df.values.astype("float32")

    # Normalize the input features using the loaded scaler
    input_array = (input_array - scaler_mean) / scaler_range

    input_embedding = model.predict(input_array)

    # Calculate cosine similarity
    similarities = np.dot(embeddings, input_embedding[0]) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding)
    )

    # Get top 5 similar songs
    top_indices = np.argsort(similarities)[-5:][::-1]

    # Filter out indices that are out of bounds for the dataset
    valid_indices = [int(idx) for idx in top_indices if idx < len(song_dataset)]
    if not valid_indices:
        raise HTTPException(status_code=500, detail="No valid similar songs found.")

    # Convert similarities to Python floats
    top_similarities = [float(similarities[idx]) for idx in valid_indices]

    # Fetch song names for valid indices
    similar_songs = song_dataset.iloc[valid_indices]["track_name"].tolist()

    # Return valid results
    return {
        "similar_songs": similar_songs,
        "similarities": top_similarities
    }

@app.post("/predict_by_name/")
def recommend_song_by_name(song_name: str):
    """
    Recommend songs based on the cosine similarity of embeddings for a given song name.
    """
    if model is None or embeddings is None or song_dataset is None:
        raise HTTPException(status_code=500, detail="Model, embeddings, or dataset not loaded")
    
    # Step 1: Find the song in the dataset
    song_row = song_dataset[song_dataset['track_name'].str.lower() == song_name.lower()]
    
    if song_row.empty:
        raise HTTPException(status_code=404, detail="Song not found in the dataset")

    # Step 2: Extract features for the given song
    features = ["danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness",
                "valence", "tempo", "duration_ms", "year"]
    input_features = song_row[features].values.astype('float32')

    # Step 3: Normalize features using the scaler
    input_features = (input_features - scaler_mean) / scaler_range

    # Step 4: Generate the embedding for the song
    input_embedding = model.predict(input_features)

    # Step 5: Calculate cosine similarity with precomputed embeddings
    similarities = np.dot(embeddings, input_embedding[0]) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding))

    # Step 6: Get top 5 similar songs
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_similarities = similarities[top_indices].tolist()

    # Step 7: Retrieve song names for the top results
    similar_songs = song_dataset.iloc[top_indices]['track_name'].tolist()

    return {"similar_songs": similar_songs, "similarities": top_similarities}

@app.post("/evaluate/")
def evaluate_model():
    """
    Evaluate the current champion model on the latest dataset.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    response = s3_client.get_object(Bucket=MINIO_BUCKET, Key='spotify/spotify_songs.csv')
    df = pd.read_csv(response['Body'])
    X = df[numerical_columns].values
    y = df["track_popularity"].values

    # Normalize features
    X = (X - scaler_mean) / scaler_range

    # Evaluate model
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred.flatten()))
    return {"champion_mae": mae}

@app.post("/add_data/")
def add_data(new_data: List[SongFeatures]):
    """
    Add new data and trigger the Airflow DAG to process and retrain the model.
    """
    new_df = pd.DataFrame([item.dict() for item in new_data])
    new_df.to_json("datasets/new_data.json", orient='records')

    # Upload to MinIO
    s3_client.upload_file('datasets/new_data.json', MINIO_BUCKET, 'spotify/new_data.json')

    # Trigger Airflow DAG
    airflow_trigger_response = trigger_airflow_dag()
    return {"message": "New data received", "airflow_response": airflow_trigger_response}

def trigger_airflow_dag():
    """
    Trigger the Airflow DAG to process the new data and retrain the model.
    """
    airflow_url = "http://airflow-webserver:8080/api/v1/dags/spotify_recommender_dag_v3/dagRuns"
    headers = {
        "Content-Type": "application/json"
    }
    auth = ('airflow', 'airflow') 
    try:
        response = requests.post(airflow_url, headers=headers, auth=auth, json={})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error triggering Airflow DAG: {response.status_code} {response.text}")
            return {"error": response.text}
    except Exception as e:
        print(f"Failed to trigger Airflow DAG: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def sanitize_data(data):
    # Convert any NaN, Inf, -Inf to finite numbers
    if isinstance(data, np.ndarray):
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    return data


def sanitize_data(data):
    """Sanitize data to convert NaN, Inf, and -Inf to finite values."""
    if isinstance(data, np.ndarray):
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    return data

@app.get("/get_songs/")
async def get_song_list():
    """Return a list of all available songs."""
    if song_dataset is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    # Extract all song names
    song_names = song_dataset['track_name'].dropna().unique().tolist()
    return {"songs": song_names}

@app.post("/predict_by_selection/")
async def predict_by_selection(song_name: str, num_pred: int = 5):
    """
    Predict similar songs using cosine similarity of embeddings
    and calculate RMSE for comparison.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if embeddings is None:
        raise HTTPException(status_code=500, detail="Embeddings not loaded")
    if song_dataset is None:
        raise HTTPException(status_code=500, detail="Song dataset not loaded")

    # Preprocess 'year' if it doesn't exist
    if 'year' not in song_dataset.columns:
        try:
            song_dataset['year'] = pd.to_datetime(
                song_dataset['track_album_release_date'], errors='coerce'
            ).dt.year

            song_dataset['year'] = song_dataset['year'].fillna(
                song_dataset['track_album_release_date'].where(
                    song_dataset['track_album_release_date'].str.len() == 4
                )
            )

            song_dataset['year'] = song_dataset['year'].fillna(
                song_dataset['track_album_release_date'].str[:4].where(
                    song_dataset['year'].isnull()
                )
            )

            song_dataset['year'] = pd.to_numeric(song_dataset['year'], errors='coerce')
            print("Preprocessed dataset to add 'year' column.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing 'year' column: {e}")

    # Find the song in the dataset
    song_row = song_dataset[song_dataset['track_name'].str.lower() == song_name.lower()]
    if song_row.empty:
        raise HTTPException(status_code=404, detail="Song not found")

    # Get the index of the song in the dataset
    song_index = song_row.index[0]

    # Extract embeddings for all songs and the target song
    song_embeddings = embeddings
    target_embedding = song_embeddings[song_index]

    # Compute cosine similarity between the target song and all other songs
    similarities = cosine_similarity([target_embedding], song_embeddings)[0]

    # Get top num_pred most similar songs (excluding the song itself)
    similar_indices = np.argsort(similarities)[-(num_pred + 1):-1][::-1]

    # Filter out invalid indices
    valid_indices = [idx for idx in similar_indices if idx < len(song_dataset)]
    if not valid_indices:
        raise HTTPException(status_code=500, detail="No valid similar songs found.")
    
    # Retrieve similar songs
    similar_songs = song_dataset.iloc[valid_indices]

    # Features for comparison
    features = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "year"
    ]

    # Extract original song features
    original_song_features = song_row[features].iloc[0].astype(float)

    # Calculate mean of features for similar songs
    similar_mean = similar_songs[features].mean()

    # Calculate mean of the whole dataset for features
    dataset_mean = song_dataset[features].mean()

    # Prepare a comparison DataFrame
    comparison_df = pd.DataFrame({
        "Original song": original_song_features,
        "Similar songs": similar_mean,
        "All songs": dataset_mean
    })

    # Calculate RMSE between original song and similar songs
    comparison_df["Similarity (RMSE)"] = np.sqrt(
        (comparison_df["Original song"] - comparison_df["Similar songs"]) ** 2
    )

    # Calculate RMSE between similar songs and the overall dataset
    comparison_df["Difference (RMSE)"] = np.sqrt(
        (comparison_df["Similar songs"] - comparison_df["All songs"]) ** 2
    )

    # Calculate Efficacy (%)
    comparison_df["Efficacy (%)"] = (
        comparison_df["Difference (RMSE)"] /
        (comparison_df["Similarity (RMSE)"] + comparison_df["Difference (RMSE)"])
    ) * 100

    # Build a result payload
    results = {
        "similar_songs": similar_songs[["track_name", "track_artist", "track_album_name"]].to_dict(orient="records"),
        "similarity_scores": [similarities[idx] for idx in valid_indices],
        "comparison": comparison_df.round(3).to_dict(orient="index")
    }

    return results

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/view/", response_class=HTMLResponse)
async def view_page():
    with open("templates/index.html", "r") as file:
        return file.read()