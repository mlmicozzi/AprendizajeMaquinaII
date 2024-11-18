from datetime import timedelta
from airflow.decorators import dag, task
import datetime
import json
import boto3
import os

# DAG Configuration
MINIO_BUCKET = "data"
MINIO_ENDPOINT = "http://s3:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_ACCESS_KEY", "minio123")
MLFLOW_TRACKING_URI = 'http://mlflow:5000'

default_args = {
    'owner': "Martín Horn",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=30),
}

@dag(
    dag_id="spotify_retrain_challenge_model_v3",
    description="Retrain and challenge the Spotify recommendation model with MLflow tracking.",
    tags=["Model Retrain", "Spotify", "Model Comparison"],
    default_args=default_args,
    catchup=False,
)
def spotify_retrain_challenge_dag():

    @task.virtualenv(
        task_id="train_challenger_model",
        requirements=["tensorflow==2.13.0", "pandas==2.0.3", "mlflow==2.10.2", "boto3==1.28.0", "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def train_challenger_model():
        import pandas as pd
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout
        from sklearn.metrics import mean_absolute_error
        import mlflow
        import boto3
        import os
        import numpy as np
        import json

        os.environ["MINIO_ENDPOINT"] = "http://s3:9000"
        os.environ["MINIO_ACCESS_KEY"] = "minio"
        os.environ["MINIO_SECRET_KEY"] = "minio123"
        os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
        MINIO_BUCKET = "data"
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
        MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
        MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        s3_client = boto3.client('s3', endpoint_url=MINIO_ENDPOINT,
                                 aws_access_key_id=MINIO_ACCESS_KEY,
                                 aws_secret_access_key=MINIO_SECRET_KEY)

        # Load pre-split data from MinIO
        def load_data_from_minio(key):
            response = s3_client.get_object(Bucket=MINIO_BUCKET, Key=key)
            return pd.read_csv(response['Body'])
        
        def load_json_from_minio(key):
            """Load JSON data from MinIO."""
            try:
                response = s3_client.get_object(Bucket=MINIO_BUCKET, Key=key)
                return json.loads(response['Body'].read().decode())
            except Exception as e:
                print(f"Error loading {key} from MinIO: {e}")
                return None

        def save_json_to_minio(data, minio_key):
            """Save JSON data directly to MinIO without using a local file."""
            try:
                # Convert data to JSON string
                json_data = json.dumps(data)

                # Upload JSON string directly to MinIO
                s3_client.put_object(
                    Bucket=MINIO_BUCKET,
                    Key=minio_key,
                    Body=json_data,
                    ContentType='application/json'
                )
                print(f"Successfully uploaded {minio_key} to MinIO.")
            except Exception as e:
                print(f"Error saving {minio_key} to MinIO: {e}")

        X_train = load_data_from_minio('spotify/train/X_train.csv').values
        y_train = load_data_from_minio('spotify/train/y_train.csv').values.flatten()
        X_test = load_data_from_minio('spotify/test/X_test.csv').values
        y_test = load_data_from_minio('spotify/test/y_test.csv').values.flatten()

        embeddings_dict = load_json_from_minio('data_info/song_embeddings.json')
        if embeddings_dict is not None and "embeddings" in embeddings_dict:
            embeddings = np.array(embeddings_dict["embeddings"])
        else:
            print("No existing embeddings found, initializing a new array.")
            embeddings = np.empty((0, 11))  # Assuming the embedding size is 32

        # Ensure embeddings are two-dimensional
        if embeddings.ndim != 2:
            print(f"Invalid embeddings shape: {embeddings.shape}. Initializing as empty.")
            embeddings = np.empty((0, 11))

        # Define the challenger embedding model
        inputs = Input(shape=(X_train.shape[1],))
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        embedding_layer = Dense(32, activation='relu', name="embedding_layer")(x)
        x = Dropout(0.3)(embedding_layer)
        outputs = Dense(X_train.shape[1], activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.2)

        # Get embeddings and log to MLflow
        new_embeddings = model.predict(X_train)

        # Update the embeddings by concatenating new ones with the existing ones
        updated_embeddings = np.concatenate((embeddings, new_embeddings), axis=0)

        experiment = mlflow.set_experiment("Spotify Recommender")
        with mlflow.start_run(run_name="Challenger_Model", experiment_id=experiment.experiment_id):
            mlflow.log_params({"model_type": "embedding_model"})
            train_mae = mean_absolute_error(X_train, model.predict(X_train))
            mlflow.log_metric("train_mae", train_mae)

            # Log the model to MLflow
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="model",
                registered_model_name="spotify_model_prod"
            )
            model_path = "models/challenger_model.h5"
            model.save(model_path, save_format='h5')

            # Upload the .h5 model file to MinIO
            updated_embeddings_dict = {"embeddings": updated_embeddings.tolist()}
            save_json_to_minio(
                updated_embeddings_dict,
                'data_info/song_embeddings.json'
            )
            

        return "Challenger model trained and logged."

    @task.virtualenv(
        task_id="evaluate_champion",
        requirements=["tensorflow==2.13.0", "pandas==2.0.3", "mlflow==2.10.2", "boto3==1.28.0", "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def evaluate_champion():
        import pandas as pd
        import tensorflow as tf
        from sklearn.metrics import mean_absolute_error
        import mlflow
        import boto3
        import os
        
        os.environ["MINIO_ENDPOINT"] = "http://s3:9000"
        os.environ["MINIO_ACCESS_KEY"] = "minio"
        os.environ["MINIO_SECRET_KEY"] = "minio123"
        os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
        
        MINIO_BUCKET = "data"
        MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
        MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
        MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()

        s3_client = boto3.client('s3', endpoint_url=MINIO_ENDPOINT,
                                 aws_access_key_id=MINIO_ACCESS_KEY,
                                 aws_secret_access_key=MINIO_SECRET_KEY)

        # Load pre-split test data from MinIO
        def load_data_from_minio(key):
            response = s3_client.get_object(Bucket="data", Key=key)
            return pd.read_csv(response['Body'])

        X_test = load_data_from_minio('spotify/test/X_test.csv').values
        y_test = load_data_from_minio('spotify/test/y_test.csv').values.flatten()

        # Load the challenger model
        response = s3_client.get_object(Bucket="data", Key='models/challenger_model.h5')
        with open("models/challenger_model.h5", 'wb') as f:
            f.write(response['Body'].read())
        challenger_model = tf.keras.models.load_model("models/challenger_model.h5")

        # Load the champion model
        
        response = s3_client.get_object(Bucket="data", Key='models/challenger_spotify_model.h5')
        with open("models/challenger_spotify_model.h5", 'wb') as f:
            f.write(response['Body'].read())
        champion_model = tf.keras.models.load_model("models/challenger_spotify_model.h5")

        # Evaluate both models
        y_pred_challenger = challenger_model.predict(X_test)
        challenger_mae = mean_absolute_error(X_test, y_pred_challenger)

        y_pred_champion = champion_model.predict(X_test)
        champion_mae = mean_absolute_error(X_test, y_pred_champion)

        # Log evaluation results to MLflow
        experiment = mlflow.set_experiment("Spotify Recommender")
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_metric("test_mae_challenger", challenger_mae)
            mlflow.log_metric("test_mae_champion", champion_mae)

            if challenger_mae < champion_mae:
                print("Promoting challenger to champion.")
                client.set_registered_model_alias("spotify_model_prod", "champion", model_data.version)
                s3_client.copy_object(
                    Bucket="data",
                    CopySource=f"data/models/challenger_spotify_model.h5",
                    Key='models/spotify_recommender.h5'
                )

    # Task dependencies
    train_challenger_model() >> evaluate_champion()

dag = spotify_retrain_challenge_dag()
