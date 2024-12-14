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

El dataset contiene más de 30000 registros de canciones de Spotify de 6 categorías (EDM, Latin, Pop, R&B, Rap y Rock), tomadas por medio de la API de Spotify.

El dataset cuenta con información actualizada a fines del 2023.

Nos centraremos en el campo `track_popularity` y buscaremos su relación con otros campos que describen las características musicales como: `key`, `tempo`, `danceability`, `energy`, entre otros; además del género/subgénero y el año de lanzamiento.

#### Dataset original: [30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)

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

- Probar la siguiente ruta [text](http://localhost:8800/view/) en la misma podrán ver el resultado de nuestro trabajo de los ultimos 4 meses.

- De todas formas, otros links utiles son:

  - [text](http://localhost:9000) - MinIO
  - [text](http://localhost:5000) - MLFlow
  - [text](http://localhost:8080) - Airflow