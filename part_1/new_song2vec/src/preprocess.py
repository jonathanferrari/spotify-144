# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
import pickle

# preprocess function
def preprocess_data(experiment_dir_path, window_size, random_seed=42):

    # load dataframe
    filename = experiment_dir_path + "/data/combined_dataset.csv"
    dataframe = pd.read_csv(filename, index_col=0, sep='\t')

    # separate track names by playlist
    print('COLS')
    print(dataframe.head(3))
    # print(type(dataframe.all_data))
    # print(type(dataframe.all_data[0]))
    # print(dataframe.all_data[0][0])
    unique_playlists = dataframe["Unnamed: 1"].unique() # changing playlist_name to artist_name
    print("# unique playlists: {count}".format(count=len(unique_playlists)))
    tracks_by_playlist = []
    for playlist in unique_playlists:
        tracks = dataframe[dataframe["Unnamed: 1"] == playlist]['Unnamed: 3'].tolist() #Unnamed: 1 is artist, Unnamed: 2 is song
        tracks_by_playlist.append(tracks)

    # tokenize track names
    tokenizer = Tokenizer(split=None)
    tokenizer.fit_on_texts(tracks_by_playlist)
    sequences = tokenizer.texts_to_sequences(tracks_by_playlist)

    # extract vocabulary
    vocabulary_size = len(tokenizer.word_index) + 1
    print("vocabulary size (# tracks): {size}".format(size=vocabulary_size))

    # store tokenizer
    pickle.dump(tokenizer, open(experiment_dir_path + "/data/tokenizer.pkl", "wb"))

    # generate skip-grams
    skip_grams = [skipgrams(sequence, vocabulary_size, window_size=window_size, seed=random_seed) for sequence in sequences]

    # prepare training data
    targets = np.array([], dtype="int32")
    contexts = np.array([], dtype="int32")
    y = np.array([], dtype="int32")

    for i, skip_gram in enumerate(skip_grams):
        pairs = list(zip(*skip_gram[0]))
        if len(pairs) > 0:
            target_idxs = np.array(pairs[0], dtype="int32")
            context_idxs = np.array(pairs[1], dtype="int32")
            labels = np.array(skip_gram[1], dtype="int32")
            targets = np.concatenate((targets, target_idxs), axis=0)
            contexts = np.concatenate((contexts, context_idxs), axis=0)
            y = np.concatenate((y, labels), axis=0)

    X = [targets, contexts]

    # store dataset
    num_datapoints = y.shape[0]
    pickle.dump(X, open(experiment_dir_path + "/data/X.pkl", "wb"))
    pickle.dump(y, open(experiment_dir_path + "/data/y.pkl", "wb"))
    print("exported {count} skip-gram pairs to {path}.".format(count=num_datapoints, path=experiment_dir_path + "/data"))