import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_general(line_1, line_2, legend_lst, title, x_label, y_label, vertical_lines_x, vl_min, vl_max, text_strings=None):
    """
    Plot two lines on the same plot with additional general information.

    :param line_1: y values of the first line
    :param line_2: y values of the second line
    :param legend_lst: list of two values -> [first line label, second line label]
    :param title: plot title (string)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    font = {'size': 18}
    plt.rc('font', **font)

    plt.plot(line_1, linewidth=3)
    plt.plot(line_2, linewidth=3)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', alpha=0.5, linestyles='dotted', linewidth=3)
    plt.legend(legend_lst)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.25, vl_min, text_strings[i], colors='k', alpha=0.5)
    plt.show()


def plot_multiple_results(data, legend_lst, title, colors, x_label, y_label, vertical_lines_x, vl_min, vl_max, show_CI=True, text_strings=None):
    """
    Plot more lines from the saved results on the same plot with additional information.

    :param data: list of 2D matrices, each matrix has more samples of the same experiment (number of experiments x length of experiment)
    :param legend_lst: list of label values (length of data)
    :param title: plot title (string)
    :param colors: list of colors used for lines (length of data)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param show_CI: show confidence interval range (boolean)
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    font = {'size': 18}
    plt.rc('font', **font)

    # plot lines with confidence intervals
    for i, data in enumerate(data):
        matrix = np.array(data)
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)

        # take only every n-th element of the array
        n = 1
        mean = mean[0::n]
        std = std[0::n]

        # plot the shaded range of the confidence intervals (mean +/- std)
        if show_CI:
            up_limit = mean + std
            up_limit[up_limit > 100] = 100  # cut accuracies above 100
            down_limit = mean - std
            plt.fill_between(range(0, mean.shape[0] * n, n), up_limit, down_limit, color=colors[i], alpha=0.25)

        # plot the mean on top (every other line is dashed)
        if i % 2 == 0:
            plt.plot(range(0, mean.shape[0] * n, n), mean, colors[i], linewidth=3)
        else:
            plt.plot(range(0, mean.shape[0] * n, n), mean, colors[i], linewidth=3, linestyle='--')

    if legend_lst:
        plt.legend(legend_lst)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', linestyles='dashed', linewidth=2, alpha=0.5)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.5, vl_min, text_strings[i], color='k', alpha=0.5)
    plt.show()



