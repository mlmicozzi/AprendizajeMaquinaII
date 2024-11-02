
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

def evaluate_nn_model_time(model, X_test_tensor, y_test, num_estimations=20):
    execution_times = []  # Lista para almacenar los tiempos de ejecución
    mae, rmse, mape, r2 = None, None, None, None  # Inicializa variables para las métricas

    for i in range(num_estimations):
        start_time = time.time()  # Inicia medición de tiempo de ejecución

        # Realizar predicciones en el conjunto de prueba
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).squeeze().numpy()

        # Calcular las métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  
        r2 = r2_score(y_test, y_pred)

        end_time = time.time()  # Finaliza medición de tiempo de ejecución
        total_time = end_time - start_time
        execution_times.append(total_time)  # Almacena el tiempo de ejecución

    # Calcula el promedio y la desviación estándar de los tiempos de ejecución
    avg_time = np.mean(execution_times)
    std_time = np.std(execution_times)

    print(f"\nEstadísticas de tiempo de ejecución ({num_estimations} iteraciones):")
    print(f"- Tiempo promedio: {avg_time:.5f} segundos")
    print(f"- Desviación estándar: {std_time:.5f} segundos")

    # Imprime las métricas calculadas en la última iteración
    print(f"\nMétricas en la última iteración:")
    print(f"- MAE: {mae:.5f}")
    print(f"- RMSE: {rmse:.5f}")
    print(f"- MAPE: {mape:.3e}")
    print(f"- R2: {r2:.5f}")

    return avg_time, std_time