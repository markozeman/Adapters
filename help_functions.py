import numpy as np
import tensorflow as tf


seeds = [(i * 7) + 1 for i in range(5)]


def truncate_pad_inputs(X_train, X_test, num_word_embeddings):
    """
    Truncate or pad vector embeddings to unique size.

    :param X_train: train data
    :param X_test: test data
    :param num_word_embeddings: integer, the number of truncated or padded word embeddings
    :return: X_train, X_test
    """
    if X_train.shape[1] >= num_word_embeddings:     # truncate
        return X_train[:, :num_word_embeddings, :], X_test[:, :num_word_embeddings, :]
    else:    # pad
        difference = num_word_embeddings - X_train.shape[1]
        padded_zeros = tf.zeros(shape=(X_train.shape[0], difference, 1024))  # 1024 is the size of the word embedding in ELMo
        X_train = tf.concat([X_train, padded_zeros], axis=1)

        padded_zeros = tf.zeros(shape=(X_test.shape[0], difference, 1024))  # 1024 is the size of the word embedding in ELMo
        X_test = tf.concat([X_test, padded_zeros], axis=1)

        return X_train, X_test


def random_binary_array(size, task_index, layer_index):
    """
    Create an array of 'size' length consisting only of numbers -1 and 1 (approximately 50% each).

    :param size: shape of the created array
    :param task_index: index of a task (in reality task_index=0 means the second task since the first does not have context)
    :param layer_index: index of the layer (1 for the input layer etc.)
    :return: binary numpy array with values -1 or 1
    """
    # to make sure that each task in each layer has a different seed (but seeds are the same for different runs)
    global seeds
    seed = seeds[task_index] + layer_index
    np.random.seed(seed)

    # np.random.seed(1)   # set fixed seed to have always the same random vectors
    vec = np.random.uniform(-1, 1, size)
    vec[vec < 0] = -1
    vec[vec >= 0] = 1

    return vec


def get_context_matrices(input_size, num_of_units, num_of_tasks):
    """
    Get random context vectors for simple neural network that uses binary superposition as a context.

    :param input_size: input size
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_tasks: number of different tasks (permutations of original images)
    :return: multidimensional numpy array with random context (binary superposition)
    """
    context_matrices = []
    for i in range(num_of_tasks):
        C1 = random_binary_array(input_size[0], i, 1)
        C2 = random_binary_array(num_of_units, i, 2)
        C3 = random_binary_array(num_of_units, i, 3)
        context_matrices.append([C1, C2, C3])
    return context_matrices


def get_context_matrices_CNN(model, num_of_tasks):
    """
    Get random context matrices for simple convolutional neural network that uses binary superposition as a context.

    :param model: Keras model instance
    :param num_of_tasks: number of different tasks
    :return: multidimensional numpy array with random context (binary superposition)
    """
    context_shapes = []
    for i, layer in enumerate(model.layers):
        if i < 2 or i > 3:   # conv layer or dense layer
            context_shapes.append(layer.get_weights()[0].shape)

    context_matrices = []
    for i in range(num_of_tasks):
        _, kernel_size, tensor_width, num_of_conv_layers = context_shapes[0]
        C1 = random_binary_array(kernel_size * kernel_size * tensor_width * num_of_conv_layers, i, 1)   # conv layer
        _, kernel_size, tensor_width, num_of_conv_layers = context_shapes[1]
        C2 = random_binary_array(kernel_size * kernel_size * tensor_width * num_of_conv_layers, i, 2)   # conv layer
        C3 = random_binary_array(context_shapes[2][0], i, 3)  # dense layer
        C4 = random_binary_array(context_shapes[3][0], i, 4)  # dense layer
        context_matrices.append([C1, C2, C3, C4])
    return context_matrices


def context_multiplication(model, context_matrices, task_index):
    """
    Multiply current model weights with context matrices in each layer (without changing weights from bias node).

    :param model: Keras model instance
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param task_index: index of a task to know which context_matrices row to use
    :return: None (but model weights are changed)
    """
    for i, layer in enumerate(model.layers[1:]):  # first layer is Flatten so we skip it
        curr_w = layer.get_weights()[0]
        curr_w_bias = layer.get_weights()[1]

        new_w = np.diag(context_matrices[task_index][i]) @ curr_w
        layer.set_weights([new_w, curr_w_bias])


def context_multiplication_CNN(model, context_matrices, task_index):
    """
    Multiply current model weights in CNN with context matrices in each layer (without changing weights from bias node).

    :param model: Keras model instance
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param task_index: index of a task to know which context_matrices row to use
    :return: None (but model weights are changed)
    """
    for i, layer in enumerate(model.layers):
        if i < 2 or i > 3:  # conv or dense layer
            curr_w = layer.get_weights()[0]
            curr_w_bias = layer.get_weights()[1]

            if i < 2:   # conv layer
                new_w = np.reshape(np.multiply(curr_w.flatten(), context_matrices[task_index][i]), curr_w.shape)
            else:    # dense layer
                new_w = np.diag(context_matrices[task_index][i - 2]) @ curr_w  # -2 because of Flatten and MaxPooling layers

            layer.set_weights([new_w, curr_w_bias])

