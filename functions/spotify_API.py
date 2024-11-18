#from dotenv import load_dotenv
#import os
import requests
import base64

def GetAccessToken() :
    """
    Devuelve un access token para utilizar en las distintas llamadas de la API de Spotify

    :returns: Access token
    """

    client_id = 'dd5330a3e8a748aa8b74f19336b196f1' # guardar en .env
    client_secret = 'ee6e6be9304c403fa6072e2f7c5d249c' # guardar en .env

    # Codificar las credenciales en base64
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    token_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {'grant_type': 'client_credentials'}

    response = requests.post(token_url, headers=headers, data=data)
    access_token = response.json().get('access_token')

    return access_token


def GetTrackInfo(track_id: str) :
    """
    Dado un id devuelve la informacion de la track

    :param track_id: track id
    :type track_id: str
    :returns: track info
    :rtype: json
    """

    track_url = f'https://api.spotify.com/v1/tracks/{track_id}'
    access_token = GetAccessToken()
    headers = {'Authorization': f'Bearer {access_token}'}

    response = requests.get(track_url, headers=headers)
    track_info = response.json()
    
    return track_info

def GetTracksInfo(track_ids: list) :
    """
    Dado una lista de ids devuelve la informacion de la track

    :param track_ids: lista de track ids
    :type track_ids: list
    :returns: lista de track info
    :rtype: list
    """

    track_url = f"https://api.spotify.com/v1/tracks?ids={','.join(track_ids)}"
    
    access_token = GetAccessToken()
    headers = {'Authorization': f'Bearer {access_token}'}

    response = requests.get(track_url, headers=headers)
    track_info = response.json()
    return track_info

def SearchByName(track_name: str, limit: int = 5) :
    """
    Dado un track name devuelve una lista de tracks que coincidan

    :param track_name: lista de track ids
    :type track_name: list
    :param limit: limite de resultados de la búqueda. Es 5 por default. Máximo 50
    :type limit: int
    :returns: lista de tracks que coincidan con el track name
    :rtype: list
    """

    url = 'https://api.spotify.com/v1/search'
    params = {
        'q': track_name,
        'type': 'track',
        'limit': limit
    }
    
    access_token = GetAccessToken()
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(url, headers=headers, params=params)
    search_results = response.json()
    return search_results

def SearchByYear(year_range: str = '2010-2020' , limit: int = 5) :
    """
    Dado un rango de años devuelve una lista de track ids que coincidan

    :param year_range: rango de años para la búsqueda. Es '2010-2020' por default
    :type year_range: str
    :param limit: limite de resultados de la búqueda. 5 por default. Máximo 50
    :type limit: int
    :returns: lista de tracks ids que coincidan con el rango de años
    :rtype: list
    """

    track_ids = []
    url = 'https://api.spotify.com/v1/search'
    params = {
        'q': f'year: {year_range}',
        'type': 'track',
        'limit': limit
    }
    
    access_token = GetAccessToken()    
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(url, headers=headers, params=params)
    search_results = response.json()

    for track in search_results['tracks']['items']:
        track_ids.append(track['id'])

    return track_ids

def GetAudioFeatures(track_id: str) :
    """
    Dado un track id devuelve las audio features para esa track

    :param track_id: track id
    :type track_id: str
    :returns: audio feature
    :rtype: json
    """
    
    url = f'https://api.spotify.com/v1/audio-features/{track_id}'
    
    access_token = GetAccessToken()
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(url, headers=headers)
    audio_features = response.json()
    return audio_features


def GetAudiosFeatures(track_ids: list) :
    """
    Dada una lista de track ids devuelve una lista con las audio features para esas tracks

    :param track_id: lista track ids
    :type track_id: list
    :returns: lista de audio feature
    :rtype: list
    """

    url = f"https://api.spotify.com/v1/audio-features?ids={','.join(track_ids)}"
    
    access_token = GetAccessToken()
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(url, headers=headers)
    audio_features = response.json()
    return audio_features

def JoinTrackInfo(track_info, audio_features) :
    """
    Devuelve un json con la track info y audio feature

    :param track_info: track info
    :type track_info: json
    :param audio_features: audio feature
    :type audio_features: json
    :returns: track info y audio features
    :rtype: json
    """

    new_data = {
        "track_id": track_info['id'],
        "track_name": track_info['name'],
        "track_artist": track_info['artists'][0]['name'],
        "track_artist_id": track_info['artists'][0]['id'],
        "track_popularity": track_info['popularity'],
        "track_album_id": track_info['album']['id'],
        "track_album_name": track_info['album']['name'],
        "album_release_date": track_info['album']['release_date'],
        "danceability": audio_features['danceability'],
        "energy": audio_features["energy"],
        "key": audio_features["key"],
        "loudness": audio_features["loudness"],
        "mode": audio_features["mode"],
        "speechiness": audio_features["speechiness"],
        "acousticness": audio_features["acousticness"],
        "instrumentalness": audio_features["instrumentalness"],
        "liveness": audio_features["liveness"],
        "valence": audio_features["valence"],
        "tempo": audio_features["tempo"],
        "duration_ms": track_info['duration_ms'],
    }

    return new_data