import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.activations import get as get_keras_activation


def nn(input_size, num_of_units, num_of_classes):
    """
    Create simple NN model with two hidden layers, each has 'num_of_units' neurons.

    :param input_size: input size
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_size))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def cnn(input_size, num_of_units, num_of_classes):
    """
    Create simple CNN model.

    :param input_size: input size
    :param num_of_units: number of neurons in FC layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_size))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def random_nn_adapter(input_size, num_of_units, num_of_classes):
    """
    Create simple NN model with two frozen dense hidden layers and adapter layers.

    :param input_size: input size
    :param num_of_units: number of neurons in FC layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_size))
    model.add(CustomAdapterLayer(input_size, activation='linear'))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(CustomAdapterLayer(num_of_units, activation='linear'))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(CustomAdapterLayer(num_of_units, activation='linear'))
    model.add(Dense(num_of_classes, activation='softmax'))

    # freeze dense layers
    model.layers[2].trainable = False
    model.layers[4].trainable = False
    model.layers[6].trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def random_cnn_adapter(input_size, num_of_units, num_of_classes):
    """
    Create simple CNN model with frozen convolutional and frozen two dense hidden layers and trainable adapter layers.

    :param input_size: input size
    :param num_of_units: number of neurons in FC layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_size))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(CustomAdapterLayer(6272, activation='linear'))  # 6272 is the size of flattened vector
    model.add(Dense(num_of_units, activation='relu'))
    model.add(CustomAdapterLayer(num_of_units, activation='linear'))
    model.add(Dense(num_of_classes, activation='softmax'))

    # freeze conv. and dense layers
    model.layers[0].trainable = False
    model.layers[1].trainable = False
    model.layers[5].trainable = False
    model.layers[7].trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def adapter_from_FC_model(FC_model, input_size, num_of_units, num_of_classes):
    """
    Change pretrained FC network to network with adapters, while freezing pretrained dense layers.

    :param FC_model: trained fully connected network model
    :param input_size: input size
    :param num_of_units: number of neurons in FC layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    curr_w = {
        '0': FC_model.layers[1].get_weights().copy(),
        '1': FC_model.layers[2].get_weights().copy(),
        '2': FC_model.layers[3].get_weights().copy()
    }

    new_model = Sequential()
    new_model.add(Flatten(input_shape=input_size))
    new_model.add(CustomAdapterLayer(input_size, activation='linear'))
    new_model.add(Dense(num_of_units, activation='relu'))
    new_model.add(CustomAdapterLayer(num_of_units, activation='linear'))
    new_model.add(Dense(num_of_units, activation='relu'))
    new_model.add(CustomAdapterLayer(num_of_units, activation='linear'))
    new_model.add(Dense(num_of_classes, activation='softmax'))

    # set parameters from pretrained network in dense layers
    new_model.layers[2].set_weights(curr_w['0'])
    new_model.layers[4].set_weights(curr_w['1'])
    new_model.layers[6].set_weights(curr_w['2'])

    # freeze dense layers
    new_model.layers[2].trainable = False
    new_model.layers[4].trainable = False
    new_model.layers[6].trainable = False

    new_model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
    new_model.summary()
    return new_model


def adapter_from_CNN_model(CNN_model, input_size, num_of_units, num_of_classes):
    """
    Change pretrained CNN_model network to network with adapters, while freezing pretrained dense layers.

    :param FC_model: trained fully connected network model
    :param input_size: input size
    :param num_of_units: number of neurons in FC layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    curr_w = {
        '0': CNN_model.layers[0].get_weights().copy(),
        '1': CNN_model.layers[1].get_weights().copy(),
        '2': CNN_model.layers[4].get_weights().copy(),
        '3': CNN_model.layers[5].get_weights().copy()
    }

    new_model = Sequential()
    new_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_size))
    new_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    new_model.add(MaxPooling2D(pool_size=(2, 2)))
    new_model.add(Flatten())
    new_model.add(CustomAdapterLayer(6272, activation='linear'))  # 6272 is the size of flattened vector
    new_model.add(Dense(num_of_units, activation='relu'))
    new_model.add(CustomAdapterLayer(num_of_units, activation='linear'))
    new_model.add(Dense(num_of_classes, activation='softmax'))

    # set parameters from pretrained network in conv. and dense layers
    new_model.layers[0].set_weights(curr_w['0'])
    new_model.layers[1].set_weights(curr_w['1'])
    new_model.layers[5].set_weights(curr_w['2'])
    new_model.layers[7].set_weights(curr_w['3'])

    # freeze conv. and dense layers
    new_model.layers[0].trainable = False
    new_model.layers[1].trainable = False
    new_model.layers[5].trainable = False
    new_model.layers[7].trainable = False

    new_model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
    new_model.summary()
    return new_model


class CustomAdapterLayer(Layer):
    """
    Custom Keras Layer for training adapters.
    """
    def __init__(self, output_dimension, activation, **kwargs):
        """
        Initialize custom layer instance.

        :param output_dimension: output size of this custom layer
        :param activation: activation function that is used on the output of this custom layer
        :param kwargs: possible additional arguments
        """
        super(CustomAdapterLayer, self).__init__(**kwargs)
        self.output_dimension = output_dimension
        self.activation = get_keras_activation(activation)

    def build(self, input_shape):
        """
        Definition of the layer's weights/parameters.

        :param input_shape: input dimension of this custom layer
        :return: None
        """
        self.W = self.add_weight(
            shape=(input_shape[-1], ),
            initializer="glorot_uniform",
            trainable=True
        )

        super(CustomAdapterLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Behaviour of this layer - feed-forward pass.

        :param inputs: the input tensor from the previous layer
        :return: output of this custom layer after activation function
        """
        o = tensorflow.math.multiply(inputs, self.W)
        return self.activation(o)

    def compute_output_shape(self, input_shape):
        """
        Specify the change in shape of the input when it passes through the layer.

        :param input_shape: input dimension of this custom layer
        :return: output shape of this custom layer
        """
        return (input_shape[0], self.output_dimension)

    def get_config(self):
        """
        Method for serialization of the custom layer.

        :return: dict with needed keys and values
        """
        config = {
            'output_dimension': self.output_dimension,
            'activation': self.activation
        }
        return config

