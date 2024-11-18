from datetime import timedelta
from airflow.decorators import dag, task
import datetime

# DAG documentation markdown
markdown_text = """
### ETL Process for Spotify Data

This DAG handles the data acquisition, preprocessing, and model training for Spotify song recommendations using MLflow for model tracking and evaluation.
"""

# DAG configuration
default_args = {
    'owner': "Martín Horn",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=30),
}

@dag(
    dag_id="spotify_recommender_dag_v3",
    description="ETL process for Spotify data including data acquisition, preprocessing, and model training with MLflow.",
    doc_md=markdown_text,
    tags=["ETL", "Spotify", "Recommendation"],
    default_args=default_args,
    catchup=False,
)
def spotify_recommender_dag_v3():

    @task.virtualenv(
        task_id="get_dataset",
        requirements=["pandas==2.0.3", "boto3==1.28.0"],
        system_site_packages=True
    )
    def get_dataset():
        import pandas as pd
        import boto3
        import os
        
        os.environ["MINIO_ENDPOINT"] = "http://s3:9000"
        os.environ["MINIO_ACCESS_KEY"] = "minio"
        os.environ["MINIO_SECRET_KEY"] = "minio123"

        MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
        MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
        MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY
        )
        
        # Check if the dataset exists in MinIO
        try:
            response = s3_client.get_object(Bucket="data", Key='spotify/spotify_songs.csv')
            df = pd.read_csv(response['Body'])
        except s3_client.exceptions.NoSuchKey:
            url = "https://example.com/path/to/spotify_songs.csv"
            df = pd.read_csv(url)
            s3_client.put_object(Bucket="data", Key='spotify/spotify_songs.csv', Body=df.to_csv(index=False))

        return df.to_json()

    @task.virtualenv(
        task_id="preprocess_and_split",
        requirements=["pandas==2.0.3", "scikit-learn==1.3.2", "boto3==1.28.0"],
        system_site_packages=True
    )
    def preprocess_and_split(df_json):
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        import boto3
        import os
        import json

        os.environ["MINIO_ENDPOINT"] = "http://s3:9000"
        os.environ["MINIO_ACCESS_KEY"] = "minio"
        os.environ["MINIO_SECRET_KEY"] = "minio123"

        MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
        MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
        MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

        df = pd.read_json(df_json)

        df['year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year

        # Step 2: Replace the years that are already in 4-digit format
        df['year'] = df['year'].fillna(
            df['track_album_release_date'].where(df['track_album_release_date'].str.len() == 4)
        )

        # Step 3: Use the first 4 characters if the year is still missing
        df['year'] = df['year'].fillna(
            df['track_album_release_date'].str[:4].where(df['year'].isnull())
        )

        # Step 4: Ensure all values in 'year' are numeric or NaN
        df['year'] = pd.to_numeric(df['year'], errors='coerce')

        # Preprocessing
        features = ["danceability", "energy", "loudness", "speechiness",
                    "acousticness", "instrumentalness", "liveness",
                    "valence", "tempo", "duration_ms", "year"]
        X = df[features].values
        y = df["track_popularity"].values

        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save the scaler to MinIO
        scaler_dict = {
            "mean": scaler.data_min_.tolist(),
            "range": (scaler.data_max_ - scaler.data_min_).tolist(),
            "features": features
        }
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY
        )

        def save_to_s3(df, key):
            csv_data = df.to_csv(index=False)
            s3_client.put_object(Bucket="data", Key=key, Body=csv_data)

        save_to_s3(pd.DataFrame(X_train), "spotify/train/X_train.csv")
        save_to_s3(pd.DataFrame(X_test), "spotify/test/X_test.csv")
        save_to_s3(pd.DataFrame(y_train), "spotify/train/y_train.csv")
        save_to_s3(pd.DataFrame(y_test), "spotify/test/y_test.csv")
        
        scaler_json = json.dumps(scaler_dict)
        s3_client.put_object(Bucket="data", Key='data_info/spotify_scaler.json', Body=scaler_json)
        print("Scaler saved to MinIO.")

        return "Data preprocessed and saved."

  

    # Define DAG dependencies
    dataset_json = get_dataset()
    preprocess_and_split(dataset_json)
   

dag_instance = spotify_recommender_dag_v3()
