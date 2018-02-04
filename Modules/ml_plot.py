# !/usr/bin/env python3

import matplotlib.pyplot as plt


# generate scatter plottype

def plot_scatter(X, Y, label, title):
    plt.scatter(X, Y, c='r')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title(title)
    plt.show()
