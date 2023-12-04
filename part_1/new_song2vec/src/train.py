# -*- coding: utf-8 -*-
import numpy as np
import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
import pickle

# training function
def build_and_train_model(experiment_dir_path, embedding_dim, learning_rate, num_epochs, random_seed=42):

    # load tokenizer
    tokenizer = pickle.load(open(experiment_dir_path + "/data/tokenizer.pkl", "rb"))
    track2idx = tokenizer.word_index
    vocabulary_size = len(track2idx) + 1

    # load training data
    X = pickle.load(open(experiment_dir_path + "/data/X.pkl", "rb"))
    y = pickle.load(open(experiment_dir_path + "/data/y.pkl", "rb"))

    # set seed
    tensorflow.random.set_seed(random_seed)

    # build model architecture
    target_inp = Input(shape=(1,)) 
    target_emb = Embedding(vocabulary_size, embedding_dim)(target_inp)
    target_emb = Flatten()(target_emb)

    context_inp = Input(shape=(1,))
    context_emb = Embedding(vocabulary_size, embedding_dim)(context_inp)
    context_emb = Flatten()(context_emb)

    extra_context_inp = Input(shape=(1,))
    extra_context_emb = Embedding(vocabulary_size, embedding_dim)(extra_context_inp)
    extra_context_emb = Flatten()(context_emb)

    extra_context2_inp = Input(shape=(1,))
    extra_context2_emb = Embedding(vocabulary_size, embedding_dim)(extra_context2_inp)
    extra_context2_emb = Flatten()(context_emb)

    x = Dot(axes=1)([target_emb, context_emb])
    x = Dense(1, activation="sigmoid")(x)

    new = Dot(axes=1)([context_emb, extra_context2_emb])
    # new = Dot(axes=1)([new, extra_context_emb])
    new = Dense(1, activation="sigmoid")(new)

    # model = Model([target_inp, context_inp], x) #original
    # model.add([extra_context_inp, extra_context2_inp], new)

    # model = Sequential()
    # modela = Model([target_inp, context_inp], x)
    # modelb = Model([extra_context_inp, extra_context2_inp], new)
    # model.add([target_inp, context_inp], x)
    # model.add(modela)
    # model.add(modelb)
    # model = Model(inputs=[target_inp, context_inp], outputs=[x, x]) # this worked but idk what its doing
    # model = Model(inputs=[context_inp, extra_context2_inp, extra_context_inp], outputs=new)
    model = Model(inputs=[target_inp, context_inp, extra_context2_inp], outputs=[x, new])


    print(model.summary())

    # compile model
    optimizer = Adam(learning_rate=learning_rate)
    loss = "binary_crossentropy"
    model.compile(optimizer=optimizer, loss=loss)

    # use gpu if available
    device_config = "/CPU:0"
    physical_devices = tensorflow.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        device_config = "/GPU:0"

    # fit model
    with tensorflow.device(device_config):
        r = model.fit(X, y, epochs=num_epochs)

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