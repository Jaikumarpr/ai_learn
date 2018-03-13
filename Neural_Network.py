import numpy as np  
import pandas as pd 
import Modules.ml_data as dh 
import Modules.Neural as nn 
import matplotlib.pyplot as plt

import matplotlib.animation as animation


# iris data ====================

def run_iris_learning():
    
    train_data, test_data = dh.load_iris_data()

    train_X = train_data.iloc[:, 1:5].T
    train_Y = train_data.iloc[:, 6:9].T

    test_X = test_data.iloc[:, 1:5].T
    test_Y = test_data.iloc[:, 6:9].T

    print(train_X)
    print(train_Y)

    # train

    tanhh = nn.activate_tanh()
    RELU = nn.activate_RELU()
    LRELU = nn.activate_LRELU()

    NN = nn.NeuralNetwork([4, 3, 3], tanhh)

    w, b = NN.train(train_X.values, train_Y.values, 0.01, 500)

    # calculate training accuracy

    print('Training accuracy is {0:.2f} %'. format(NN.training_accuracy(train_X.values, train_Y.values)))
    print('Testing accuracy is {0:.2f} %'. format(NN.training_accuracy(
        test_X.values, test_Y.values)))

    print(NN.predict(np.array([5.0, 2.3, 3.3, 1.0]).T))


if __name__ == "__main__":
    run_iris_learning()

    
