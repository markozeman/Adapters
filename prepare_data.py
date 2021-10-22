import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub
import math
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K


def generator(X, step, elmo, start, end):
    """
    Generator function for ELMo embeddings.

    :param X: inputs
    :param step: int - how many samples to save at once
    :param elmo: pre-trained ELMo embedding
    :param start: starting index for embedding of X
    :param end: starting index for embedding of X
    :return: embeddings
    """
    for i in range(math.ceil(len(X) / step)):
        print('i, start: ', i, start)
        embeddings = np.array(elmo(tf.convert_to_tensor(np.array(X[start:end]))))
        start += step
        end += step
        yield embeddings


def save_embeddings(elmo, X, y, step, train_test, path):
    """
    Use ELMo embeddings to extract features of the data and save them.
    For large datasets this function will take a while to process.

    :param elmo: pre-trained ELMo embedding
    :param X: inputs
    :param y: labels (one-hot-encoded)
    :param step: int - how many samples to save at once
    :param train_test: str - 'train' or 'test'
    :param: path: str - relative path to the folder
    :return: None
    """
    np.save('%s%s_labels.npy' % (path, train_test), y)   # save labels

    start = 0
    end = start + step

    g = generator(X, step, elmo, start, end)

    for embeddings in g:
    # for i in range(math.ceil(len(X) / step)):
    #     print('i: ', i)
    #
    #     embeddings = np.array(elmo(tf.convert_to_tensor(np.array(X[start:end]))))
    #
        # print('\nEmbeddings shape: (batch_size, hidden_size): ', embeddings.shape)

        np.save('%s%s_embeddings_%s_%s.npy' % (path, train_test, str(start), str(end)), embeddings)   # save embeddings

        start += step
        end += step


def load_embeddings(step, train_test, path):
    """
    Load saved extracted features of the data and reconstruct them.

    :param step: int - how many samples to load at once
    :param step: str - 'train' or 'test'
    :param: path: str - relative path to the folder
    :return: X, y
    """
    y = np.load('%s%s_labels.npy' % (path, train_test))   # load labels

    start = 0
    end = start + step

    Xs = []
    for i in range(math.ceil(len(y) / step)):
        X_part = np.load('%s%s_embeddings_%s_%s.npy' % (path, train_test, str(start), str(end)))   # load embeddings
        Xs.append(X_part)

        start += step
        end += step

    X = np.vstack(tuple(Xs))

    return X, y


def preprocess_hate_speech(filepath, test_split, nn_cnn):
    """
    Preprocess hate speech data from CSV and split them to train and test dataset.
    1 - hate speech, 0 – no hate speech.

    :param filepath: string path to the file with data
    :param test_split: share of the data to be test set
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :return: data in the shape of (X_train, y_train, X_test, y_test)
    """
    '''
    data = pd.read_csv(filepath)

    X = data["tweet"]
    y = data["class"]

    # make target label binary
    X = list(map(lambda s: str(s), X))
    y = list(map(lambda c: 0 if c == 2 else 1, y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=10, shuffle=True)

    # from strings to vector embeddings with ELMo
    # elmo = tensorflow_hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    # X = elmo(X_train + X_test, signature="default", as_dict=True)["elmo"]
    # elmo = tensorflow_hub.KerasLayer("https://tfhub.dev/google/elmo/3", trainable=False, signature="tokens", output_key="elmo")
    # X = elmo(X_train + X_test)["elmo"]

    # one-hot encode labels
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    '''

    step = 100

    '''
    # save ELMo embeddings to disk
    elmo = tensorflow_hub.KerasLayer("https://tfhub.dev/google/elmo/3", signature="default")
    save_embeddings(elmo, X_train, y_train, step, 'train', 'ELMo_embeddings/hate_speech/')
    save_embeddings(elmo, X_test, y_test, step, 'test', 'ELMo_embeddings/hate_speech/')
    '''

    # load ELMo embeddings from disk
    X_train, y_train = load_embeddings(step, 'train', 'ELMo_embeddings/hate_speech/')
    X_test, y_test = load_embeddings(step, 'test', 'ELMo_embeddings/hate_speech/')

    if nn_cnn == 'cnn':     # reshape 1024-dimensional vectors to squares 32x32
        X_train = np.reshape(X_train, (X_train.shape[0], 32, 32, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], 32, 32, 1))

    return X_train, y_train, X_test, y_test


def preprocess_IMDB_reviews(filepath, test_split, nn_cnn):
    """
    Preprocess hate speech data from CSV and split them to train and test dataset.
    1 - positive sentiment, 0 – negative sentiment.

    :param filepath: string path to the file with data
    :param test_split: share of the data to be test set
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :return: data in the shape of (X_train, y_train, X_test, y_test)
    """
    '''
    data = pd.read_csv(filepath)

    X = data["review"]
    y = data["sentiment"]

    # make target label binary integer
    X = list(map(lambda s: str(s), X))
    y = list(map(lambda sentiment: 0 if sentiment == 'negative' else 1, y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=10, shuffle=True)

    # # from strings to vector embeddings with ELMo
    # elmo = tensorflow_hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    # X = elmo(X_train + X_test, signature="default", as_dict=True)["elmo"]
    # X_train_len = len(X_train)
    # X_train = X[:X_train_len, :, :]
    # X_test = X[X_train_len:, :, :]

    # one-hot encode labels
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    '''

    step = 2

    '''
    # save ELMo embeddings to disk
    elmo = tensorflow_hub.KerasLayer("https://tfhub.dev/google/elmo/3", signature="default")
    save_embeddings(elmo, X_train, y_train, step, 'train', 'ELMo_embeddings/sentiment_analysis/')
    save_embeddings(elmo, X_test, y_test, step, 'test', 'ELMo_embeddings/sentiment_analysis/')
    '''

    # load ELMo embeddings from disk
    X_train, y_train = load_embeddings(step, 'train', 'ELMo_embeddings/sentiment_analysis/')
    X_test, y_test = load_embeddings(step, 'test', 'ELMo_embeddings/sentiment_analysis/')

    if nn_cnn == 'cnn':     # reshape 1024-dimensional vectors to squares 32x32
        X_train = np.reshape(X_train, (X_train.shape[0], 32, 32, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], 32, 32, 1))

    return X_train, y_train, X_test, y_test


def preprocess_SMS_spam(filepath, test_split, nn_cnn):
    """
    Preprocess SMS spam data from CSV and split them to train and test dataset.
    1 - spam, 0 – not spam.

    :param filepath: string path to the file with data
    :param test_split: share of the data to be test set
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :return: data in the shape of (X_train, y_train, X_test, y_test)
    """
    '''
    data = pd.read_csv(filepath, encoding='latin-1')

    X = data["v2"]
    y = data["v1"]

    # make target label binary integer
    X = list(map(lambda s: str(s), X))
    y = list(map(lambda c: 0 if c == 'ham' else 1, y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=10, shuffle=True)

    # one-hot encode labels
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    '''

    step = 20

    '''
    # save ELMo embeddings to disk
    elmo = tensorflow_hub.KerasLayer("https://tfhub.dev/google/elmo/3", signature="default")
    save_embeddings(elmo, X_train, y_train, step, 'train', 'ELMo_embeddings/sms_spam/')
    save_embeddings(elmo, X_test, y_test, step, 'test', 'ELMo_embeddings/sms_spam/')
    '''

    # load ELMo embeddings from disk
    X_train, y_train = load_embeddings(step, 'train', 'ELMo_embeddings/sms_spam/')
    X_test, y_test = load_embeddings(step, 'test', 'ELMo_embeddings/sms_spam/')

    if nn_cnn == 'cnn':     # reshape 1024-dimensional vectors to squares 32x32
        X_train = np.reshape(X_train, (X_train.shape[0], 32, 32, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], 32, 32, 1))

    return X_train, y_train, X_test, y_test




