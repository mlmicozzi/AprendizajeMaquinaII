// Elementos del DOM
const searchPage = document.getElementById('searchPage');
const recommendationsPage = document.getElementById('recommendationsPage');

const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchResults = document.getElementById('searchResults');
const searchPlayerSection = document.getElementById('searchPlayerSection');
const searchPlayer = document.getElementById('searchPlayer');
const confirmBtn = document.getElementById('confirmBtn');

const recommendationsList = document.getElementById('recommendationsList');
const recommendationsPlayerSection = document.getElementById('recommendationsPlayerSection');
const recommendationsPlayer = document.getElementById('recommendationsPlayer');
const backBtn = document.getElementById('backBtn');
const shareBtn = document.getElementById('shareBtn');
const addToSpotifyBtn = document.getElementById('addToSpotifyBtn');

// Variables globales
let selectedTrackId = null;
let recommendationsData = [];

// Función para buscar canciones (simulación)
async function searchTracks(query) {
  // Aquí iría una llamada real a la API: fetch('/search?query=' + query)
  // Datos simulados:
  return [
    {
      track_id: '1A',
      track_name: 'Canción Ejemplo 1',
      artist_name: 'Artista X',
      album_cover_url: 'https://via.placeholder.com/60'
    },
    {
      track_id: '2B',
      track_name: 'Canción Ejemplo 2',
      artist_name: 'Artista Y',
      album_cover_url: 'https://via.placeholder.com/60'
    }
  ];
}

// Función para obtener recomendaciones (simulación)
async function getRecommendations(songId) {
  // Aquí iría fetch('/recommend?song_id=' + songId)
  // Datos simulados:
  return {
    "recommendations": [
      {
        "track_id": "R1",
        "track_name": "Recomendada 1",
        "artist_name": "Artista Recom 1",
        "album_cover_url": "https://via.placeholder.com/60"
      },
      {
        "track_id": "R2",
        "track_name": "Recomendada 2",
        "artist_name": "Artista Recom 2",
        "album_cover_url": "https://via.placeholder.com/60"
      },
      {
        "track_id": "R3",
        "track_name": "Recomendada 3",
        "artist_name": "Artista Recom 3",
        "album_cover_url": "https://via.placeholder.com/60"
      },
      {
        "track_id": "R4",
        "track_name": "Recomendada 4",
        "artist_name": "Artista Recom 4",
        "album_cover_url": "https://via.placeholder.com/60"
      },
      {
        "track_id": "R5",
        "track_name": "Recomendada 5",
        "artist_name": "Artista Recom 5",
        "album_cover_url": "https://via.placeholder.com/60"
      }
    ]
  };
}

// Manejar búsqueda
searchBtn.addEventListener('click', async () => {
  const query = searchInput.value.trim();
  if (!query) return;

  const results = await searchTracks(query);
  renderSearchResults(results);
});

function renderSearchResults(results) {
  searchResults.innerHTML = '';
  results.forEach(track => {
    const item = document.createElement('div');
    item.className = 'recommendation-item';

    const img = document.createElement('img');
    img.className = 'album-cover';
    img.src = track.album_cover_url;
    img.alt = track.track_name;

    const info = document.createElement('div');
    info.className = 'track-info';

    const title = document.createElement('h3');
    title.innerText = track.track_name;

    const artist = document.createElement('p');
    artist.innerText = track.artist_name;

    info.appendChild(title);
    info.appendChild(artist);

    const playBtn = document.createElement('button');
    playBtn.className = 'play-btn';
    playBtn.innerText = 'Escuchar';
    playBtn.addEventListener('click', () => {
      selectedTrackId = track.track_id;
      showSelectedTrackPlayer(track.track_id);
    });

    item.appendChild(img);
    item.appendChild(info);
    item.appendChild(playBtn);

    searchResults.appendChild(item);
  });
}

function showSelectedTrackPlayer(trackId) {
  searchPlayerSection.style.display = 'block';
  // Actualiza el src del iframe con el track_id seleccionado
  searchPlayer.src = `https://open.spotify.com/embed/track/${trackId}`;
}

// Confirmar selección
confirmBtn.addEventListener('click', async () => {
  if (!selectedTrackId) return;
  // Llamar a la API para obtener recomendaciones
  const data = await getRecommendations(selectedTrackId);
  recommendationsData = data.recommendations;
  // Cambiar a la página de recomendaciones
  searchPage.classList.remove('visible');
  recommendationsPage.classList.add('visible');
  renderRecommendations(recommendationsData);
});

function renderRecommendations(recs) {
  recommendationsList.innerHTML = '';
  recs.forEach(track => {
    const item = document.createElement('div');
    item.className = 'recommendation-item';

    const img = document.createElement('img');
    img.className = 'album-cover';
    img.src = track.album_cover_url;
    img.alt = track.track_name;

    const info = document.createElement('div');
    info.className = 'track-info';

    const title = document.createElement('h3');
    title.innerText = track.track_name;

    const artist = document.createElement('p');
    artist.innerText = track.artist_name;

    info.appendChild(title);
    info.appendChild(artist);

    const playBtn = document.createElement('button');
    playBtn.className = 'play-btn';
    playBtn.innerText = 'Reproducir';
    playBtn.addEventListener('click', () => {
      showRecommendationPlayer(track.track_id);
    });

    item.appendChild(img);
    item.appendChild(info);
    item.appendChild(playBtn);

    recommendationsList.appendChild(item);
  });
}

function showRecommendationPlayer(trackId) {
  recommendationsPlayerSection.style.display = 'block';
  recommendationsPlayer.src = `https://open.spotify.com/embed/track/${trackId}`;
}

// Botón volver
backBtn.addEventListener('click', () => {
  recommendationsPage.classList.remove('visible');
  searchPage.classList.add('visible');
  // Limpiar
  recommendationsPlayerSection.style.display = 'none';
  selectedTrackId = null;
  searchPlayerSection.style.display = 'none';
  searchResults.innerHTML = '';
  searchInput.value = '';
});

// Compartir (ejemplo simple)
shareBtn.addEventListener('click', () => {
  alert("Funcionalidad de compartir en redes sociales.");
});

// Agregar playlist a Spotify (ejemplo)
addToSpotifyBtn.addEventListener('click', () => {
  alert("Funcionalidad para agregar playlist a la cuenta de Spotify.");
});
