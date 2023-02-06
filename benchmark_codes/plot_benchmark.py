import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_loss_accuracy(path_to_csv: str = "../benchmark/resnet-101-history.csv"):
    """
    Plot the graph of accuracy and loss
    :param path_to_csv: path to the csv file
    :return: graph
    """
    # read csv file
    loss = pd.read_csv(path_to_csv, usecols=["loss"])
    accuracy = pd.read_csv(path_to_csv, usecols=["accuracy"])

    # plot loss
    plt.plot(loss, label="loss")
    plt.plot(accuracy, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.savefig("../benchmark/resnet-152-epoch-20-non-enhanced-loss-accuracy.png")
    plt.show()


if __name__ == "__main__":
    plot_loss_accuracy("E:\\Pycharm\\ResNetEvaluations\\benchmark\\resnet-152-epoch-20-history-non-enhanced.csv")
