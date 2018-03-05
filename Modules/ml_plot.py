# !/usr/bin/env python3

import matplotlib.pyplot as plt


FIGURE_COUNT = 0

# generate scatter plottype

def plot_scatter(X, Y, label, title):
    
    global FIGURE_COUNT
    FIGURE_COUNT += 1
    plt.figure(FIGURE_COUNT)
    plt.scatter(X, Y, c='r')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.title(title)


def plot_loss(loss):
    
    global FIGURE_COUNT
    FIGURE_COUNT += 1

    plt.figure(FIGURE_COUNT)
    plt.plot(loss)
