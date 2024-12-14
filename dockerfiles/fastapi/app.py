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


s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)


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
        
        model_path = load_file('models/challenger_spotify_model.h5', 'files/challenger_spotify_model.h5')

        
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully from:", model_path)

        
        embeddings_dict = load_json('data_info/song_embeddings.json', 'files/song_embeddings.json')
        embeddings = np.array(embeddings_dict["embeddings"])
        print(f"Loaded {len(embeddings)} embeddings successfully.")

        
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

    
    input_df = pd.DataFrame([features.dict()])
    input_array = input_df.values.astype("float32")

    
    input_array = (input_array - scaler_mean) / scaler_range

    input_embedding = model.predict(input_array)

    
    similarities = np.dot(embeddings, input_embedding[0]) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding)
    )

    
    top_indices = np.argsort(similarities)[-5:][::-1]

    
    valid_indices = [int(idx) for idx in top_indices if idx < len(song_dataset)]
    if not valid_indices:
        raise HTTPException(status_code=500, detail="No valid similar songs found.")

   
    top_similarities = [float(similarities[idx]) for idx in valid_indices]

    
    similar_songs = song_dataset.iloc[valid_indices]["track_name"].tolist()

    
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
    
    
    song_row = song_dataset[song_dataset['track_name'].str.lower() == song_name.lower()]
    
    if song_row.empty:
        raise HTTPException(status_code=404, detail="Song not found in the dataset")

    
    features = ["danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness",
                "valence", "tempo", "duration_ms", "year"]
    input_features = song_row[features].values.astype('float32')

   
    input_features = (input_features - scaler_mean) / scaler_range

    
    input_embedding = model.predict(input_features)

   
    similarities = np.dot(embeddings, input_embedding[0]) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(input_embedding))

   
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_similarities = similarities[top_indices].tolist()

    
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
    
    # Extract some song names. Otherwise it crashes! But you can try other songs
    song_names = sorted(song_dataset['track_name'].dropna().unique().tolist())[:1000]
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

    
    song_index = song_row.index[0]

    
    song_embeddings = embeddings
    target_embedding = song_embeddings[song_index]

    
    similarities = cosine_similarity([target_embedding], song_embeddings)[0]

    
    similar_indices = np.argsort(similarities)[-(num_pred + 1):-1][::-1]

    
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

    
    original_song_features = song_row[features].iloc[0].astype(float)

    
    similar_mean = similar_songs[features].mean()

    
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

    
    comparison_df["Efficacy (%)"] = (
        comparison_df["Difference (RMSE)"] /
        (comparison_df["Similarity (RMSE)"] + comparison_df["Difference (RMSE)"])
    ) * 100

    
    results = {
        "similar_songs": similar_songs[["track_id", "track_name", "track_artist", "track_album_name"]].to_dict(orient="records"),
        "similarity_scores": [similarities[idx] for idx in valid_indices],
        "comparison": comparison_df.round(3).to_dict(orient="index")
    }

    return results

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/view/", response_class=HTMLResponse)
async def view_page():
    with open("templates/index.html", "r") as file:
        return file.read()
    

from fastapi import File, UploadFile
from io import StringIO

@app.post("/predict_csv/")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with song data and get predictions for track popularity.
    Each row is processed independently.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode("utf-8")))

        # valation of required columns
        required_columns = numerical_columns
        if not all(col in df.columns for col in required_columns):
            missing_columns = set(required_columns) - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")

        
        predictions = []

        
        for _, row in df.iterrows():
            # Convert row to a DataFrame
            input_df = pd.DataFrame([row[required_columns].to_dict()])
            input_array = input_df.values.astype("float32")

            # Normalize the input features using the scaler
            input_array = (input_array - scaler_mean) / scaler_range

            # Predict track popularity
            prediction = model.predict(input_array).flatten()[0]
            predictions.append(prediction)

        
        df["predicted_track_popularity"] = predictions

        
        output_csv = StringIO()
        df.to_csv(output_csv, index=False)
        output_csv.seek(0)

        # Return the CSV as a response
        return JSONResponse({
            "message": "Predictions generated successfully",
            "predictions": df[["predicted_track_popularity"]].to_dict(orient="records")
        })

    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")