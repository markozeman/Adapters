import numpy as np
import copy
from tensorflow.keras.callbacks import Callback


class TestPerformanceCallback(Callback):
    """
    Callback class for testing normal model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, model):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.model = model  # this is only a reference, not a deep copy
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.accuracies.append(accuracy * 100)


class TestSuperpositionPerformanceCallback(Callback):
    """
    Callback class for testing superposition NN model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, context_matrices, model, task_index):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.context_matrices = context_matrices
        self.model = model  # this is only a reference, not a deep copy
        self.task_index = task_index
        self.accuracies = []

    def get_diagonal_matrix_inverse(self, diag_vector):
        """
        Return the reversed vector from the diagonal matrix.

        :param diag_vector: vector on the diagonal of the matrix
        :return: reversed vector
        """
        return np.array(list(map(lambda x: 1/x, diag_vector)))

    def on_epoch_begin(self, epoch, logs=None):
        if self.task_index == 0:    # first task - we did not use context yet
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
            self.accuracies.append(accuracy * 100)
            return

        # save current model weights (without bias node)
        curr_w_matrices = []
        curr_bias_vectors = []
        for layer in self.model.layers[1:]:  # first layer is Flatten so we skip it
            curr_w_matrices.append(layer.get_weights()[0])
            curr_bias_vectors.append(layer.get_weights()[1])

        # temporarily change model weights to be suitable for first task (without bias node)
        for i, layer in enumerate(self.model.layers[1:]):  # first layer is Flatten so we skip it
            # not multiplying with inverse because inverse is the same in binary superposition with {-1, 1} on the diagonal
            # using only element-wise multiplication on diagonal vectors for speed-up
            context_inverse_multiplied = copy.deepcopy(self.context_matrices[self.task_index][i])
            for task_i in range(self.task_index - 1, 0, -1):
                context_inverse_multiplied = np.multiply(context_inverse_multiplied, self.context_matrices[task_i][i])

            context_inverse_multiplied = np.diag(context_inverse_multiplied)    # vector to diagonal matrix

            layer.set_weights([context_inverse_multiplied @ curr_w_matrices[i], curr_bias_vectors[i]])

        # evaluate accuracy
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.accuracies.append(accuracy * 100)

        # change model weights back (without bias node)
        for i, layer in enumerate(self.model.layers[1:]):  # first layer is Flatten so we skip it
            layer.set_weights([curr_w_matrices[i], curr_bias_vectors[i]])


class TestSuperpositionPerformanceCallback_CNN(Callback):
    """
    Callback class for testing superposition CNN model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, context_matrices, model, task_index):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.context_matrices = context_matrices
        self.model = model  # this is only a reference, not a deep copy
        self.task_index = task_index
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        if self.task_index == 0:    # first task - we did not use context yet
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
            self.accuracies.append(accuracy * 100)
            return

        # save current model weights (without bias node)
        curr_w_matrices = []
        curr_bias_vectors = []
        for i, layer in enumerate(self.model.layers):
            if i < 2 or i > 3:  # conv or dense layer
                curr_w_matrices.append(layer.get_weights()[0])
                curr_bias_vectors.append(layer.get_weights()[1])

        # temporarily change model weights to be suitable for first task (without bias node)
        for i, layer in enumerate(self.model.layers):
            if i < 2 or i > 3:  # conv or dense layer
                # not multiplying with inverse because inverse is the same in binary superposition with {-1, 1} on the diagonal
                # using only element-wise multiplication on diagonal vectors for speed-up

                if i < 2:  # conv layer
                    # flatten
                    context_vector = self.context_matrices[self.task_index][i]
                    for task_i in range(self.task_index - 1, 0, -1):
                        context_vector = np.multiply(context_vector, self.context_matrices[task_i][i])

                    new_w = np.reshape(np.multiply(curr_w_matrices[i].flatten(), context_vector), curr_w_matrices[i].shape)
                    layer.set_weights([new_w, curr_bias_vectors[i]])
                else:  # dense layer
                    context_inverse_multiplied = self.context_matrices[self.task_index][i - 2]
                    for task_i in range(self.task_index - 1, 0, -1):
                        context_inverse_multiplied = np.multiply(context_inverse_multiplied, self.context_matrices[task_i][i - 2])
                    context_inverse_multiplied = np.diag(context_inverse_multiplied)    # vector to diagonal matrix

                    layer.set_weights([context_inverse_multiplied @ curr_w_matrices[i - 2], curr_bias_vectors[i - 2]])

        # evaluate accuracy
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.accuracies.append(accuracy * 100)

        # change model weights back (without bias node)
        for i, layer in enumerate(self.model.layers):
            if i < 2 or i > 3:  # conv or dense layer
                if i < 2:  # conv layer
                    layer.set_weights([curr_w_matrices[i], curr_bias_vectors[i]])
                else:  # dense layer
                    layer.set_weights([curr_w_matrices[i - 2], curr_bias_vectors[i - 2]])


