import streamlit as st
import pandas as pd
import numpy as np
import utils

# https://docs.streamlit.io/library/api-reference/widgets
# https://docs.streamlit.io/library/api-reference/layout


st.title('An App to Create Playlists')

data = pd.read_csv("part_1/new_song2vec/raw_data/spotify_playlists.tsv", sep="\t")
songs = data["track_name"].unique()

song_name = st.selectbox("Select a song", songs)

is_unique = utils.is_unique_song_name(data, song_name)
if is_unique:
    artist = data[data["track_name"] == song_name]["artist_name"].iloc[0]
    st.write(f"Artist: {artist}")
else:
    possible_artists = data[data["track_name"] == song_name]["artist_name"].unique()
    artist = st.selectbox("Select an artist", possible_artists)

playlist = utils.get_most_similar_tracks(song_name, artist)

dance_range = (0.0, 1.0)
bpm_range = (60, 180)
loudness_range = (-60, 0)
playlist_length = 10
max_length = float("inf")



dance_cont = st.container(border=True)
with dance_cont:
    dance_filter_bool = st.checkbox("Filter by Danceability")
    if dance_filter_bool:
        dance_range = st.slider("Danceability Range:", 0.0, 1.0, (0.0, 1.0))

bpm_cont = st.container(border=True)
with bpm_cont:
    bpm_filter_bool = st.checkbox("Filter by BPM")
    if bpm_filter_bool:
        bpm_range = st.slider("BPM Range:", 60, 180, (60, 180))

loudness_cont = st.container(border=True)
with loudness_cont:
    loudness_filter_bool = st.checkbox("Filter by Loudness")
    if loudness_filter_bool:
        loudness_range = st.slider("Loudness Range:", -60, 0, (-60, 0))
        
length_cont = st.container(border=True)
with length_cont:
    length_filter_bool = st.checkbox("Filter by Length")
    if length_filter_bool:
        playlist_length = st.slider("Playlist Length:", 5, 50, 10)

time_cont = st.container(border=True)
with time_cont:
    time_filter_bool = st.checkbox("Filter by Time")
    if time_filter_bool:
        max_length = st.slider("Max Length (minutes):", 10, 120, 30)

artist_cont = st.container(border=True)
with artist_cont:
    artist_filter_bool = st.checkbox("Filter by artist")
    if artist_filter_bool:
        artist_to_filter = st.selectbox("Artist Filter:", playlist["artist_name"].unique())
        num_artist_songs = len(playlist[playlist["artist_name"] == artist_to_filter])
        artist_cat_filter = st.selectbox("Songs by artist:", ["At most k", "At Least k", "None of"])
        if artist_cat_filter != "None of":
            k = st.slider("k:", 1, min(num_artist_songs, playlist_length), min(5, num_artist_songs//2, playlist_length//2))
        else:
            k = 0
        
    
st.write("## Playlist")
playlist = playlist[(playlist["danceability"] >= dance_range[0]) & (playlist["danceability"] <= dance_range[1])]
playlist = playlist[(playlist["tempo"] >= bpm_range[0]) & (playlist["tempo"] <= bpm_range[1])]
playlist = playlist[(playlist["loudness"] >= loudness_range[0]) & (playlist["loudness"] <= loudness_range[1])]
playlist = playlist.sort_values("similarity", ascending=False)
if artist_filter_bool:
    if artist_cat_filter == "None of":
        playlist = playlist[playlist["artist_name"] != artist_to_filter]
    else:
        non_artist_songs = playlist[playlist["artist_name"] != artist_to_filter]
        artist_songs = playlist[playlist["artist_name"] == artist_to_filter].sort_values("similarity", ascending=False)
        if artist_cat_filter == "At most k":
            artist_songs = artist_songs[:k]
            playlist = pd.concat([non_artist_songs, artist_songs]).sort_values("similarity", ascending=False)
        else:
            temp_playlist = playlist.head(playlist_length)
            if temp_playlist["artist_name"].isin([artist_to_filter]).sum() < k:
                artist_songs = artist_songs[:k]
                playlist = pd.concat([non_artist_songs, artist_songs]).sort_values("similarity", ascending=False)

playlist = playlist.head(playlist_length)

if time_filter_bool:
    empty_playlist = pd.DataFrame(columns=playlist.columns)
    running_time = 0
    for i, row in playlist.iterrows():
        if running_time + row["duration_ms"] / 60000 <= max_length:
            running_time += row["duration_ms"] / 60000
            empty_playlist = empty_playlist.append(row)
        else: 
            break
    playlist = empty_playlist

display_playlist = playlist.reset_index(drop=True)[["track_name", "artist_name", "similarity"]]
st.dataframe(display_playlist, width = 1000)