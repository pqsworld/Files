import os
import re
from io import BytesIO

import flask
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline, make_interp_spline


def extract_g_l1_sm_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            if "G_L1_sm" not in line:
                continue
            str_num = line.split("G_L1_sm:")[-1].split("G_L1_kong")[0].replace(" ", "")

            number = float(str_num)
            numbers.append(number)
    return numbers


def traverse_directory_and_extract_numbers(folder_path):
    all_numbers = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'loss_log.txt':
                file_path = os.path.join(root, file)
                name = file_path.split('/')[-2]
                numbers = extract_g_l1_sm_numbers_from_file(file_path)
                all_numbers[name] = numbers
    return all_numbers


def plot_experiments_loss(experiments):
    """
    Plots the loss values over epochs for different experiments.

    Args:
        experiments (dict): A dictionary with experiment names as keys and
                            lists of loss values as values.
    """

    def str2int(parameter_list):
        return int(parameter_list.split('-')[0])

    # Initialize a new figure
    plt.figure(figsize=(25, 14))

    sorted_experiments_names = sorted(experiments.items(), key=lambda item: str2int(item[0]))
    # Loop through each experiment and plot its loss values
    for experiment_name, loss_values in sorted_experiments_names:
        if not loss_values:
            continue

        # epochs = list(range(1, len(loss_values) + 1))
        epochs = np.array(range(1, len(loss_values) + 1))
        loss_values = np.array(loss_values)
        spl = UnivariateSpline(epochs, loss_values, s=500)
        smooth_epochs = np.linspace(epochs.min(), epochs.max(), 2000)
        smooth_loss_values = spl(smooth_epochs)

        # plt.plot(epochs, loss_values, label=experiment_name)
        plt.plot(smooth_epochs, smooth_loss_values, label=experiment_name)

        plt.annotate(
            experiment_name,
            (epochs[-2], loss_values[-2]),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize='8',
            ha='left',
        )

    # Add labels and legend
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title('Loss over Epochs for Different Experiments', fontsize=20)
    plt.legend(fontsize=14, loc='upper right')

    # plt.xticks(ticks=np.arange(0, len(epochs) + 1, step=20))
    # plt.yticks(
    #     ticks=np.linspace(
    #         min(min(values) for values in experiments.values()), max(max(values) for values in experiments.values()), num=10
    #     )
    # )

    plt.xticks(ticks=np.arange(0, 1700, step=20))
    plt.ylim(5, 16)
    plt.yticks(ticks=list(np.arange(0, 10, 1)) + list(np.arange(12, 15, 5)))

    # Show grid

    plt.grid(True)

    # Show the plot
    plt.show()

    # Save the plot as an image file
    plt.savefig('experiments_loss_plot.svg', format='svg', bbox_inches='tight')


def home():
    folder_path = '/hdd/share/quality/checkpoints'  # replace with the path to your folder
    numbers_by_file = traverse_directory_and_extract_numbers(folder_path)

    plot_experiments_loss(numbers_by_file)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return


if __name__ == '__main__':
    home()
