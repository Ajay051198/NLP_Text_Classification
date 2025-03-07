import os
import time
import tensorflow as tf
import numpy as np

import build_model
import vectorize_data
import explore_data
import matplotlib.pyplot as plt

TOP_K = 20000

def get_embedding_matrix(word_index, embedding_data_dir, embedding_dim):
    """Gets embedding matrix from the embedding index data.
    # Arguments
        word_index: dict, word to index map that was generated from the data.
        embedding_data_dir: string, path to the pre-training embeddings.
        embedding_dim: int, dimension of the embedding vectors.
    # Returns
        dict, word vectors for words in word_index from pre-trained embedding.
    # References:
        https://nlp.stanford.edu/projects/glove/
        Download and uncompress archive from:
        http://nlp.stanford.edu/data/glove.6B.zip
    """

    # Read the pre-trained embedding file and get word to word vector mappings.
    embedding_matrix_all = {}

    # We are using 200d GloVe embeddings.
    fname = os.path.join(embedding_data_dir, 'glove.6B.200d.txt')
    with open(fname, encoding="utf8") as f:
        for line in f:  # Every line contains word followed by the vector value
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix_all[word] = coefs

    # Prepare embedding matrix with just the words in our word_index dictionary
    num_words = min(len(word_index) + 1, TOP_K)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        if i >= TOP_K:
            continue
        embedding_vector = embedding_matrix_all.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def train_sequence_model(data,
                        embedding_data_dir,
                        learning_rate=1e-3,
                        epochs=1000,
                        batch_size=128,
                        blocks=2,
                        filters=64,
                        dropout_rate=0.2,
                        embedding_dim=200,
                        kernel_size=3,
                        pool_size=3,
                        save_as='NA'):
    """Trains sequence model on the given dataset.
    # Arguments
        data: tuples of training and test texts and labels.
        embedding_data_dir: string, path to the pre-training embeddings.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    (train_texts, train_labels), (val_texts, val_labels) = data

    num_classes = explore_data.get_num_classes(train_labels)

    # Vectorize texts (word2vec)
    tokenizer, word_index, max_length = vectorize_data.train_vectorizer(train_texts)
    x_train = vectorize_data.convert_to_sequence(train_texts, tokenizer, max_length)
    x_val = vectorize_data.convert_to_sequence(val_texts, tokenizer, max_length)

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    embedding_matrix = get_embedding_matrix(
        word_index, embedding_data_dir, embedding_dim)

    # Create model instance. First time we will train rest of network while
    # keeping embedding layer weights frozen. So, we set
    # is_embedding_trainable as False.
    model = build_model.sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features,
                                     use_pretrained_embedding=True,
                                     is_embedding_trainable=False,
                                     embedding_matrix=embedding_matrix)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    model.fit(x_train,
              train_labels,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=(x_val, val_labels),
              verbose=1,  # Logs once per epoch.
              batch_size=batch_size)

    # Save the model.
    model.save_weights('sequence_model_with_pre_trained_embedding.h5')

    model = build_model.sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=x_train.shape[1:],
                                     num_classes=num_classes,
                                     num_features=num_features,
                                     use_pretrained_embedding=True,
                                     is_embedding_trainable=True,
                                     embedding_matrix=embedding_matrix)

    # Compile model with learning parameters.
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Load the weights that we had saved into this new model.
    model.load_weights('sequence_model_with_pre_trained_embedding.h5')

    # Train and validate model.
    history = model.fit(x_train,
                        train_labels,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(x_val, val_labels),
                        verbose=1,  # Logs once per epoch.
                        batch_size=batch_size)

    history = history.history

    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save_weights('{}_tuned_model_weights.h5'.format(save_as))
    model.save('{}_tuned_model.h5'.format(save_as))
    return history['val_acc'][-1], history['val_loss'][-1]
