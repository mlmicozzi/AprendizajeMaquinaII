# Trabajo Práctico - Aprendizaje de Máquinas II

### Carrera de Especialización en Inteligencia Artificial - Cohorte 17

## Autores:
- **Martín Horn**
- **Alejandro Lloveras**
- **Diego Martín Méndez**
- **María Luz Micozzi**
- **Juan Ruíz Otondo**

---

## Dataset

Los datos que alimentan al modelo de recomendación provienen de un proceso de exploración y transformación del dataset público en Kaggle `30000-spotify-songs`, que fue construido a través de un proceso de scrapeo de la API de Spotify (`spotifyr`).

El dataset cuenta con información de canciones de Spotify, actualizada hasta fines del 2023, organizadas en 6 categorias (EDM, Latin, Pop, R&B, Rap y Rock).

De los 32833 registros del dataset original, nuestro set de datos se reduce a 23081 entradas (**-29.70%**) luego del proceso de limpieza aplicado.

#### Dataset original: [30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)

#### Dataset procesado: [TP Final - Aprendizaje de Máquinas I](https://github.com/mlmicozzi/SPOTIFY/)

## Estructura del Proyecto

El repositorio está compuesto por los siguientes archivos y directorios:

- Airflow: Es el directorio donde encontraremos los dags para procesar en airflow. Contiene:
  - Entrenamiento y procesamiento de los datos
  - Competencia entre modelos anteriores y nuevos modelos

- datasets: 
  - En esta carpeta encontraremos los archivos .csv donde utlizados para nuestros modelos.

- Dockerfiles: Contamos con distintos notebooks donde se entrenan y exportan los diferentes modelos.
    - airflow:
      * Gestión de DAGS.
    - fastapi:
      * Aqui encontraremos la definicion de app.py desde donde controlamos todas nuestras API.
    - mlflow:
      * Getión de metricas de los modelos.
    - postgres:
      * Gestión de la base de datos


- functions:
  - Fue desarrollada para este proyecto una API de spotify que nos permitia recuperar todos los atributos de una canción para mejorar aún más los modelos. Desafortunadamente el 27/11/2024 cambiaron las politicas empresariales y no pudimos avanzar con este proyecto.


## Ejecución del Proyecto

Para levantar el proyecto correr desde el root el comando:

```
docker compose --profile all up
```

Recomendaciones: 

- Probar la siguiente ruta [Spotify recommender](http://localhost:8800/view/) en la misma podrán ver el resultado de nuestro trabajo de los ultimos 4 meses.

- En caso de querer probar las predicciones en bache utilizar la url: [Predicción en bache](http://localhost:8800/predict_csv/). Se deja en este repositorio un .csv de ejemplo para tal caso.

```
curl -X POST "http://localhost:8800/predict_csv/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/your/csv.csv"
```

Sample output:

```
{
    "message": "Predictions generated successfully",
    "predictions": [
        {
            "predicted_track_popularity": 0.7880306839942932
        },
        {
            "predicted_track_popularity": 0.5666085481643677
        }
    ]
}
```


- De todas formas, otros links utiles son:

  - [MinIO](http://localhost:9000) 
  - [MLFlow](http://localhost:5000) 
  - [Airflow](http://localhost:8080) 