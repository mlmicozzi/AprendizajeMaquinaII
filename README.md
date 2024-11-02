# Trabajo Práctico Integrador - Aprendizaje de Máquinas I

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

- TPFinal.ipynb: Es el notebook principal del proyecto. Contiene:
  - Exploración y Comprensión de los Datos
  - Técnicas de Visualización
  - Transformación de Variables y Selección de Features
  - Reducción de la Dimensionalidad
  - Entrenamiento de modelos
  - Pruebas con PCA
  - Conclusiones

- Carpeta models: esta carperta contiene todos los modelos exportados para luego ser utilizados en el notebook TPFinal y realizar la evaluación de los mismos.

- Notebooks: Contamos con distintos notebooks donde se entrenan y exportan los diferentes modelos.
    - decisionTreeRegressor.ipynb
    - KNeighborsRegressor.ipynb
    - neuralNetwork.ipynb
    - ridgeRegression.ipynb
    - supportVectorRegressor.ipynb

 - Carpeta datasets: contamos con distintas versiones del dataset de Spotify.
     - spotify_songs.csv: dataset original descargado de Kaggle
     - df_scaled.csv: dateset original con las variables escaladas (se usó MinMaxScaler())
     - df_pca6.csv: dataset de PCA con 6 componentes
     - df_pca9.csv: dataset de PCA con 9 componentes 
