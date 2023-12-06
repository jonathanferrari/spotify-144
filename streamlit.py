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
    artist = data[data["track_name"] == song_name]["artist_name"][0]
else:
    possible_artists = data[data["track_name"] == song_name]["artist_name"].unique()
    artist = st.selectbox("Select an artist", possible_artists)

playlist = utils.get_most_similar_tracks(song_name, artist)






cont = st.container(border=True)
with cont:
    artist_filter_bool = st.checkbox("Filter by artist")
    if artist_filter_bool:
        artist_to_filter = st.selectbox("Artist Filter:", playlist["artist_name"].unique())
        artist_cat_filter = st.selectbox("Songs by artist:", ["At most k", "At Least k", "None of"])
        if artist_cat_filter != "None of":
            k = st.slider("k:", 1, 10, 5)
        else:
            k = 0
    
    

