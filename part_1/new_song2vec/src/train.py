# -*- coding: utf-8 -*-
import numpy as np
import tensorflow
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler

import pickle

# training function
def build_and_train_model(experiment_dir_path, embedding_dim, learning_rate, num_epochs, random_seed=42):

    # load tokenizer
    tokenizer = pickle.load(open(experiment_dir_path + "/data/tokenizer.pkl", "rb"))
    track2idx = tokenizer.word_index
    vocabulary_size = len(track2idx) + 1
    dropout_rate = 0.2
    regularization_rate = 0.01

    # load training data
    X = pickle.load(open(experiment_dir_path + "/data/X.pkl", "rb"))
    y = pickle.load(open(experiment_dir_path + "/data/y.pkl", "rb"))

    # set seed
    tensorflow.random.set_seed(random_seed)

    # Define two inputs
    target_inp = Input(shape=(1,))
    context_inp = Input(shape=(1,))

    # Shared embedding layer
    # To use pre-trained embeddings, load them here and set weights=pretrained_weights, trainable=False
    embedding = Embedding(vocabulary_size, embedding_dim)

    # Target and context branches
    target_emb = embedding(target_inp)
    target_emb = Flatten()(target_emb)

    context_emb = embedding(context_inp)
    context_emb = Flatten()(context_emb)

    # Combine the outputs of the two branches
    combined = concatenate([target_emb, context_emb])

    # Add dense and output layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(regularization_rate))(combined)  # Increased neurons and added L2 regularization
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs=[target_inp, context_inp], outputs=x)

    # Compile the model with additional metrics
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Learning rate scheduler function
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    # Callback for learning rate adjustment
    callback = LearningRateScheduler(scheduler)

    # Model summary
    print(model.summary())

    # use gpu if available
    device_config = "/CPU:0"
    physical_devices = tensorflow.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        device_config = "/GPU:0"

    # fit model
    with tensorflow.device(device_config):
        r = model.fit(X, y, epochs=num_epochs, callbacks=[callback])

    # store model
    filepath = experiment_dir_path + "/model"
    model.save(filepath)
    print("exported trained model to {path}.".format(path=filepath))

    # store embedding weights
    target_embedding_layer = model.layers[2]
    embedding_weights = target_embedding_layer.get_weights()[0]
    filepath = experiment_dir_path + "/embeddings.pkl"
    pickle.dump(embedding_weights, open(filepath, "wb"))
    print("exported track embeddings to {path}.".format(path=filepath))