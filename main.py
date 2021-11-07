from prepare_data import *
from dataset_preparation import *
from networks import *
from callbacks import *
from help_functions import *
from plots import *
import numpy as np


def train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share,
                mode='normal', context_matrices=None, task_index=None):
    """
    Train and evaluate Keras model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update
    :param validation_share: share of examples to be used for validation
    :param mode: string for learning mode, important for callbacks - possible values: 'normal', 'superposition'
    :param context_matrices: multidimensional numpy array with random context (binary superposition), only used when mode = 'superposition'
    :param task_index: index of current task, only used when mode = 'superposition'
    :return: History object and 2 lists of test accuracies for every training epoch (normal and superposition)
    """
    test_callback = TestPerformanceCallback(X_test, y_test, model)
    if nn_cnn == 'nn':
        test_superposition_callback = TestSuperpositionPerformanceCallback(X_test, y_test, context_matrices, model, task_index)
    elif nn_cnn == 'cnn':
        test_superposition_callback = TestSuperpositionPerformanceCallback_CNN(X_test, y_test, context_matrices, model, task_index)

    callbacks = []
    if mode == 'normal':
        callbacks.append(test_callback)
    elif mode == 'superposition':
        callbacks.append(test_superposition_callback)

    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2,
                        validation_split=validation_share, callbacks=callbacks)      # validation_split removed because we use Tensors

    return history, test_callback.accuracies, test_superposition_callback.accuracies


def baseline_training(model, data, num_of_epochs, num_of_tasks, nn_cnn, batch_size, validation_split):
    """
    Train model for 'num_of_tasks' tasks.
    Check how accuracy of the original task is changing through tasks using normal training.

    :param model: Keras model instance
    :param data: 3x (X_train, y_train, X_test, y_test)
    :param num_of_epochs: number of epochs to train each task
    :param num_of_tasks: number of different tasks
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update
    :param validation_split: share of the training set to be used as validation set
    :return: list of test accuracies on the first task for 'num_of_epochs' epochs for each task
    """
    original_accuracies = []

    # training of the first task
    X_train, y_train, X_test, y_test = data[0]
    _, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=validation_split)
    original_accuracies.extend(accuracies)

    # other training tasks
    for i in range(num_of_tasks - 1):
        print("\n\n Task: %d \n" % (i + 1))

        X_train, y_train, _, _ = data[i + 1]       # do not load X_test, y_test since we want to monitor test accuracy on the first task
        _, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=validation_split)
        original_accuracies.extend(accuracies)

    return original_accuracies


def superposition_training(model, data, num_of_epochs, num_of_tasks, nn_cnn, batch_size, validation_split, context_matrices):
    """
    Train model for 'num_of_tasks' tasks.
    Check how accuracy for original images is changing through tasks using superposition training.

    :param model: Keras model instance
    :param data: 3x (X_train, y_train, X_test, y_test)
    :param num_of_epochs: number of epochs to train each task
    :param num_of_tasks: number of different tasks
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update
    :param validation_split: share of the training set to be used as validation set
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :return: list of test accuracies on the first task for 'num_of_epochs' epochs for each task
    """
    original_accuracies = []

    # training of the first task
    X_train, y_train, X_test, y_test = data[0]
    _, _, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size,
                                   validation_share=validation_split, mode='superposition',
                                   context_matrices=context_matrices, task_index=0)
    original_accuracies.extend(accuracies)

    # other training tasks
    for i in range(num_of_tasks - 1):
        print("\n\n Task: %d \n" % (i + 1))

        # multiply current weights with context matrices for each layer (without changing weights from bias node)
        if nn_cnn == 'nn':
            context_multiplication(model, context_matrices, i + 1)
        elif nn_cnn == 'cnn':
            context_multiplication_CNN(model, context_matrices, i + 1)

        X_train, y_train, _, _ = data[i + 1]       # do not load X_test, y_test since we want to monitor test accuracy on the first task
        history, _, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn,
                                             batch_size, validation_share=validation_split, mode='superposition',
                                             context_matrices=context_matrices, task_index=i + 1)
        original_accuracies.extend(accuracies)

    return original_accuracies


def adapter_training(model, data, num_of_epochs, num_of_tasks, nn_cnn, batch_size, validation_split, training_type):
    """
    Train model for 'num_of_tasks' tasks.
    Check how accuracy for original images is changing through tasks using superposition training.

    :param model: Keras model instance
    :param data: 3x (X_train, y_train, X_test, y_test)
    :param num_of_epochs: number of epochs to train each task
    :param num_of_tasks: number of different tasks
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update
    :param validation_split: share of the training set to be used as validation set
    :param training_type: str - 'random' (train adapter layers from starting random fixed weights)
                                or 'after first' (train adapter layers from learned parameters for the first task)
    :return: list of test accuracies on each subsequent task for 'num_of_epochs' epochs for each task
    """
    original_accuracies = []
    final_accuracies_per_task = []

    if training_type == 'random':
        # training of the first task
        X_train, y_train, X_test, y_test = data[0]
        _, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size,
                                       validation_share=validation_split)
        original_accuracies.extend(accuracies)
        final_accuracies_per_task.append(accuracies[-1])

        # other training tasks
        for i in range(num_of_tasks - 1):
            print("\n\n Task: %d \n" % (i + 1))

            X_train, y_train, X_test, y_test = data[i + 1]
            _, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size,
                                           validation_share=validation_split)
            original_accuracies.extend(accuracies)
            final_accuracies_per_task.append(accuracies[-1])

    elif training_type == 'after first':
        # training of the first task
        X_train, y_train, X_test, y_test = data[0]
        _, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size,
                                       validation_share=validation_split)
        original_accuracies.extend(accuracies)
        final_accuracies_per_task.append(accuracies[-1])

        # create new model with added adapter layers and freeze other layers
        if nn_cnn == 'nn':
            model = adapter_from_FC_model(model, input_size, num_of_units, num_of_classes)
        elif nn_cnn == 'cnn':
            model = adapter_from_CNN_model(model, input_size, num_of_units, num_of_classes)

        # other training tasks
        for i in range(num_of_tasks - 1):
            print("\n\n Task: %d \n" % (i + 1))

            X_train, y_train, X_test, y_test = data[i + 1]
            _, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size,
                                           validation_share=validation_split)
            original_accuracies.extend(accuracies)
            final_accuracies_per_task.append(accuracies[-1])

    return original_accuracies, final_accuracies_per_task


if __name__ == '__main__':
    domain = 'nlp'    # 'nlp' or 'cv'
    runs = 3
    train_normal = False
    train_superposition = False
    train_adapters = True
    adapters_type = 'random'    # 'random' or 'after first'
    overlap_tasks = True
    accuracies_normal = []
    accuracies_superposition = []
    accuracies_adapters = []
    accuracies_adapters_final = []
    for r in range(runs):
        print('RUN ', r)

        num_of_tasks = 3 if domain == 'nlp' else 2
        num_of_epochs = 1000     # epochs per task
        nn_cnn = 'nn'
        batch_size = 64
        validation_split = 0.1
        test_split = 0.1
        if domain == 'nlp':
            input_size = (1024, ) if nn_cnn == 'nn' else (32, 32, 1)
        elif domain == 'cv':
            input_size = (3072, ) if nn_cnn == 'nn' else (32, 32, 3)
        num_of_classes = 2 if domain == 'nlp' else 10
        num_of_units = 100

        if domain == 'nlp':
            data = []

            X_train, y_train, X_test, y_test = preprocess_hate_speech('datasets/hate_speech.csv', test_split, nn_cnn)
            # X_train, X_test = truncate_pad_inputs(X_train, X_test, num_word_embeddings)
            data.append((X_train, y_train, X_test, y_test))
            print('Hate speech:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            X_train, y_train, X_test, y_test = preprocess_IMDB_reviews('datasets/IMDB_sentiment_analysis.csv', test_split, nn_cnn)
            # X_train, X_test = truncate_pad_inputs(X_train, X_test, num_word_embeddings)
            data.append((X_train, y_train, X_test, y_test))
            print('Sentiment analysis:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

            X_train, y_train, X_test, y_test = preprocess_SMS_spam('datasets/sms_spam.csv', test_split, nn_cnn)
            # X_train, X_test = truncate_pad_inputs(X_train, X_test, num_word_embeddings)
            data.append((X_train, y_train, X_test, y_test))
            print('SMS spam:', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        elif domain == 'cv':
            data = get_dataset('cifar', nn_cnn, input_size, num_of_classes)
            data = data[:2]   # take only first two tasks

            # if you want overlapping tasks: Task 1: [0, 9], Task 2: [5, 14]
            if overlap_tasks:
                data = overlapping_tasks(data)

        if train_normal:
            if nn_cnn == 'nn':
                model = nn(input_size, num_of_units, num_of_classes)
            elif nn_cnn == 'cnn':
                model = cnn(input_size, num_of_units, num_of_classes)
            else:
                raise ValueError("'nn_cnn' variable must have value 'nn' or 'cnn'")

            acc_normal = baseline_training(model, data, num_of_epochs, num_of_tasks, nn_cnn, batch_size, validation_split)
            # print(acc_normal)
            accuracies_normal.append(acc_normal)

        if train_superposition:
            if nn_cnn == 'nn':
                model = nn(input_size, num_of_units, num_of_classes)
                context_matrices = get_context_matrices(input_size, num_of_units, num_of_tasks)
            elif nn_cnn == 'cnn':
                model = cnn(input_size, num_of_units, num_of_classes)
                context_matrices = get_context_matrices_CNN(model, num_of_tasks)
            else:
                raise ValueError("nn_cnn variable must have value 'nn' or 'cnn'")

            acc_superposition = superposition_training(model, data, num_of_epochs, num_of_tasks, nn_cnn, batch_size,
                                                       validation_split, context_matrices)
            # print(acc_superposition)
            accuracies_superposition.append(acc_superposition)

        '''
        if train_normal and train_superposition:
            plot_general(acc_superposition, acc_normal, ['Superposition model', 'Baseline model'],
                         'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)',
                         [(i + 1) * num_of_epochs for i in range(num_of_tasks - 1)], 0, 100)
        elif train_normal:
            plot_general([], acc_normal, ['Superposition model', 'Baseline model'],
                         'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)',
                         [(i + 1) * num_of_epochs for i in range(num_of_tasks - 1)], 0, 100)
        elif train_superposition:
            plot_general(acc_superposition, [], ['Superposition model', 'Baseline model'],
                         'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)',
                         [(i + 1) * num_of_epochs for i in range(num_of_tasks - 1)], 0, 100)
        elif train_adapters:
            plot_general(acc_adapters, [], ['Adapter model', ''],
                                 'Adapter model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)',
                                 [(i + 1) * num_of_epochs for i in range(num_of_tasks - 1)], 0, 100)
        '''

        if train_adapters:
            if adapters_type == 'random':
                if nn_cnn == 'nn':
                    model = random_nn_adapter(input_size, num_of_units, num_of_classes)
                elif nn_cnn == 'cnn':
                    model = random_cnn_adapter(input_size, num_of_units, num_of_classes)
            elif adapters_type == 'after first':
                if nn_cnn == 'nn':
                    model = nn(input_size, num_of_units, num_of_classes)
                elif nn_cnn == 'cnn':
                    model = cnn(input_size, num_of_units, num_of_classes)
            else:
                raise ValueError("adapters_type variable must have value 'random' or 'after first'")

            acc_adapters, final_accuracies = adapter_training(model, data, num_of_epochs, num_of_tasks, nn_cnn, batch_size, validation_split, adapters_type)
            accuracies_adapters.append(acc_adapters)
            accuracies_adapters_final.append(final_accuracies)

    if train_normal and train_superposition:
        all_results = [accuracies_superposition, accuracies_normal]
        plot_multiple_results(all_results, ['Superposition model', 'Baseline model'], 'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model',
                              ['tab:blue', 'tab:orange'], 'Epoch', 'Accuracy (%)',
                              [(i + 1) * num_of_epochs for i in range(num_of_tasks - 1)], 0, 100, show_CI=True)

    if train_adapters:
        accuracies_adapters_final = np.array(accuracies_adapters_final)
        means = np.mean(accuracies_adapters_final, axis=0)
        stds = np.std(accuracies_adapters_final, axis=0)
        if domain == 'nlp':
            print('Task 1 accuracy: %.1f +/- %.1f\nTask 2 accuracy: %.1f +/- %.1f\nTask 3 accuracy: %.1f +/- %.1f' %
                  (means[0], stds[0], means[1], stds[1], means[2], stds[2]))
        elif domain == 'cv':
            print('Task 1 accuracy: %.1f +/- %.1f\nTask 2 accuracy: %.1f +/- %.1f\n' %
                  (means[0], stds[0], means[1], stds[1]))
        plot_multiple_results([accuracies_adapters], ['Adapter model'],
                              'Adapter model with ' + nn_cnn.upper() + ' model', ['tab:blue'], 'Epoch', 'Accuracy (%)',
                              [(i + 1) * num_of_epochs for i in range(num_of_tasks - 1)], 0, 100, show_CI=True)


    ### Default accuracies
    # Hate speech: [4163, 20620] - 83.2%
    # Sentiment analysis: [25000, 25000] - 50%
    # SMS spam: [4825, 747] - 86.6%

    ### Trained accuracies
    # NN:
    # Hate speech: 30 units - 90.6%, 100 units - 90.4%
    # Sentiment analysis: 30 units - 86.5%, 100 units - 85.6%
    # SMS spam: 30 units - 99.2%, 100 units - 98.2%

    # CNN:
    # Hate speech: 30 units - 88.3%, 100 units - 89.1%
    # Sentiment analysis: 30 units - 83.1%, 100 units - 80.8%
    # SMS spam: 30 units - 98.2%, 100 units - 98.4%

    ### Trained adapter accuracies
    # NN - random:
    # Hate speech: 100 units - 89.2%
    # Sentiment analysis: 100 units - 85.9%
    # SMS spam: 100 units - 99.3%

    # NN - after first:
    # Hate speech: 100 units - 90.0% (on full network)
    # Sentiment analysis: 100 units - 86.3% (adapters)
    # SMS spam: 100 units - 99.0% (adapters)

    ### Number of parameters
    # NN, 30 units: 31,742
    # CNN, 30 units: 197,820
    # NN, 100 units: 112,802
    # CNN, 100 units: 637,070



