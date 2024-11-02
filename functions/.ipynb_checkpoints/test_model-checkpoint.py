import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error
from datetime import datetime

def load_datasets(path_X_test: str, path_y_test: str) -> tuple:
    """
    Carga el dataset de testeo, tanto las entradas como las salidas

    :param path_X_test: String con el path del csv con las entradas de testeo
    :type path_X_test: str
    :param path_y_test: String con el path del csv con la salida de testeo
    :type path_y_test: str
    :returns: Tupla con las entradas y salida de testeo
    :rtype: tuple
    """

    X_test = np.loadtxt(path_X_test, delimiter=",", dtype=float)
    y_test = np.loadtxt(path_y_test, delimiter=",", dtype=float,
                        skiprows=1, usecols=1)

    return X_test, y_test


def load_model(path_model: str):
    """
    Carga el artefacto del modelo

    :param path_model: Ubicación para leer el artefacto del modelo
    :type path_model: str
    :returns: Modelo binario
    :rtype: sklearn model
    """

    return pickle.load(open(path_model, 'rb'))

def test_model(model, X_test):
    """
    Testea el modelo

    :param model: Modelo de machine learning
    :type model: sklearn model
    :param X_test: Array de numpy con las entradas de testeo
    :type X_test: np.array
    :returns: y_pred
    :rtype: np.array
    """

    y_pred = model.predict(X_test)
    return y_pred


def model_metrics(y_test, y_pred):
    """
    Guardamos las diferentes métricas del modelo

    :param y_test: Array de numpy con las entradas de testeo
    :type y_test: np.array
    :param y_pred: Array de numpy con las predicciones
    :type y_pred: np.array
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Generamos artefacto
    with open("./log_testing_metric.txt", "a") as f:
        f.write(f"{timestamp} - R2-Score: {r2}\n")
        f.write(f"{timestamp} - Root mean squared error: {rmse}\n")  

