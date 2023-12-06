# imports
import numpy as np
import random
import pickle
import pandas as pd
import os


def get_track_name(df, uri):
    df = df.copy()
    df["uri"] = df["uri"].str.lower()
    return df.set_index("uri").loc[uri]['track_name']

def is_unique_song_name(df, track_name):
    return len(df[df["track_name"] == track_name]) == 1

def get_most_similar_tracks(track_name, artist, n=1000, exp_name="test_10"):
    tokenizer = pickle.load(open(f"part_1/new_song2vec/experiments/{exp_name}/data/tokenizer.pkl", "rb"))
    embedding_weights = pickle.load(open(f"part_1/new_song2vec/experiments/{exp_name}/embeddings.pkl", "rb"))
    df = pd.read_csv('part_1/new_song2vec/raw_data/spotify_playlists.tsv', sep='\t', index_col=0)
    def get_track_uri(name, artist):
        filtered = df[(df["artist_name"] == artist) & (df["track_name"] == name)]
        return filtered["uri"].iloc[0]
    track_uri = get_track_uri(track_name, artist).lower()
    track_idx = tokenizer.word_index[track_uri]
    track_vector = embedding_weights[track_idx, :].reshape(1, -1)
    similarities = np.dot(track_vector, embedding_weights.T) / (np.linalg.norm(track_vector) * np.linalg.norm(embedding_weights, axis=1))
    similarities = similarities.reshape(-1)
    most_similar_idxs = np.argpartition(similarities, -(n+1))[-(n+1):]
    most_similar_idxs = most_similar_idxs[np.argsort(similarities[most_similar_idxs])][::-1][1:]
    uris = [tokenizer.index_word[idx] for idx in most_similar_idxs]
    df["uri"] = df["uri"].str.lower()
    similarities = similarities[most_similar_idxs]
    df = df.set_index("uri").loc[uris]
    df["similarity"] = similarities
    return df