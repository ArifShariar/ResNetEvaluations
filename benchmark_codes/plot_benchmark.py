import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_graph(path_to_csv: str = "../benchmark/resnet-101-history.csv"):
    """
    Plot the graph of accuracy and loss
    :param path_to_csv: path to the csv file
    :return: graph
    """
    print("Plotting the graph... epoch: {}". format(path_to_csv))


if __name__ == "__main__":
    plot_graph()
