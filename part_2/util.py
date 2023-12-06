import numpy as np
import random
import pickle
import pandas as pd

tokenizer = pickle.load(open("./test_09" + "/data/tokenizer.pkl", "rb"))
embedding_weights = pickle.load(open("./test_09" + "/embeddings.pkl", "rb"))

df = pd.read_csv('spotify_playlists.tsv', sep='\t', index_col=0)

# function to get top-n most similar tracks
def get_most_similar_tracks(track_name, n=10, tokenizer=tokenizer, embedding_weights=embedding_weights):

    track_idx = tokenizer.word_index[track_name]
    track_vector = embedding_weights[track_idx, :].reshape(1, -1)
    similarities = np.dot(track_vector, embedding_weights.T) / (np.linalg.norm(track_vector) * np.linalg.norm(embedding_weights, axis=1))
    similarities = similarities.reshape(-1)
    most_similar_idxs = np.argpartition(similarities, -(n+1))[-(n+1):]
    most_similar_idxs = most_similar_idxs[np.argsort(similarities[most_similar_idxs])][::-1][1:]

    return [tokenizer.index_word[idx] for idx in most_similar_idxs]

def get_model_songs(track_name, n):
    song_list = get_most_similar_tracks(track_name, n=n)

    sorting_key = {song: index for index, song in enumerate(song_list)}

    # Filter the DataFrame based on the song_list
    filtered_df = df[df['track_name'].str.lower().isin(song_list)]

    # Create a new column 'sorting_key' using the custom sorting key
    filtered_df['sorting_key'] = filtered_df['track_name'].str.lower().map(sorting_key)

    # Sort the DataFrame based on the custom sorting key
    sorted_df = filtered_df.sort_values('sorting_key').drop('sorting_key', axis=1)
    return sorted_df