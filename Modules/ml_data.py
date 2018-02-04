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
