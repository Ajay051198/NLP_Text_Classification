from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
import pickle

# Vectorization parameters
TOP_K = 20000
TOKEN_MODE = 'word'
MAX_SEQUENCE_LENGTH = 500

def train_vectorizer (train_texts):
    """trains the vectorizer object so it can be used to transform training,
        validation and testing data
    # Arguments
        train_texts: list, training text strings
    # Returns
        word_index, tokenizer object
    """

    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    train_texts = tokenizer.texts_to_sequences(train_texts)
    # get and set max sequence length
    max_length =  len(max(train_texts, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer, tokenizer.word_index, max_length

def convert_to_sequence(texts, tokenizer, max_length):
    """ Vectorizes the texts as sequence vectors with the pre-fitted Tokenizer
    # Arguments
        text: list, texts to be vectorized
        tokenizer: Tokenizer object which has been fit on training data
        max_length: int, limit for padding
    # Returns:
        vectorized texts
    """
    fitted_tokenizer = tokenizer
    vectorized_text = fitted_tokenizer.texts_to_sequences(texts)

    # pad sequences
    vectorized_text = sequence.pad_sequences(vectorized_text, maxlen=max_length)

    return vectorized_text
