import tensorflow as tf
from tensorflow.keras.preprocessing import text
import vectorize_data
import os
import load_data
import pickle
import numpy as np

def get_tokenizer_and_max_lenght():
    MAX_SEQUENCE_LENGTH = 500
    (train_texts, train_labels), (val_texts, val_labels) = load_data.load_tweet_weather_topic_classification_dataset('./data')

    os.chdir("assets\TW")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    train_texts = tokenizer.texts_to_sequences(train_texts)
    # get and set max sequence length
    max_length =  len(max(train_texts, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    return tokenizer, max_length

def get_model():
    model = tf.keras.models.load_model('TW_tuned_model.h5')
    model.load_weights('TW_tuned_model_weights.h5')
    return model

def predict(text, model, tokenizer, max_length):

    decode_dict_sentiment = { 0: "Negative", 1: "Postive"}

    decode_dict_weather = {
    1 : "clouds",
    2 : "cold",
    3 : "dry",
    4 : "hot",
    5 : "humid",
    6 : "hurricane",
    7 : "I can't tell",
    8 : "ice",
    9 : "other",
    10 : "rain",
    11 : "snow",
    12 : "storms",
    13 : "sun",
    14 : "tornado",
    15 : "wind",
    }
    entry = [text]
    entry = vectorize_data.convert_to_sequence(entry, tokenizer, max_length)
    classes = model.predict(entry)
    return decode_dict_weather[np.argmax(np.array(classes))+1]
