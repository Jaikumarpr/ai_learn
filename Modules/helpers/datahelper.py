# usr/bin/env python3
import pandas as pd


def import_data(file_path, ip_idx, op_idx):
    """
    ip_idx = [column header] input column headers
    op_idx = [column headers] output column headers

    """
    df = pd.read_csv(file_path)
    return df[ip_idx].values, df[op_idx].values


def load_food_truck_data():

    return import_data("dataset/ml_ex1/food_truck_data.txt",
                       ["population"], ["profit"])


def load_iris_data():

    df = pd.read_csv("dataset/iris/iris.csv")
    df.insert(6, 'setosa', 0)
    df.insert(7, 'versicolor', 0)
    df.insert(8, 'virginica', 0)
    df.loc[df.Species == 'Iris-setosa', 'setosa'] = 1
    df.loc[df.Species == 'Iris-versicolor', 'versicolor'] = 1
    df.loc[df.Species == 'Iris-virginica', 'virginica'] = 1

    train_data = [df.iloc[0:40, :], df.iloc[50: 90, :], df.iloc[100:140, :]]
    test_data = [df.iloc[40:50, :], df.iloc[90: 100, :], df.iloc[140:150, :]]
    # pieces = [df.iloc[0:9, :], df.iloc[50: 59, :], df.iloc[100:139, :]]
    return pd.concat(train_data), pd.concat(test_data)


    # return import_data("dataset/iris/iris.csv", ["SepalLengthCm",
    #                    "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    #                    ["Species"])
