import streamlit as st
import pandas as pd
import numpy as np

# https://docs.streamlit.io/library/api-reference/widgets
# https://docs.streamlit.io/library/api-reference/layout


st.title('An App to Create Playlists')

data = pd.read_csv("part_1/new_song2vec/raw_data/spotify_playlists.tsv",
                   sep="\t")

st.write(data.head())

