# usr/bin/env python3
import pandas as pd
import numpy as np


def import_data(file_path, ip_idx, op_idx):
    """
    ip_idx = [column header] input column headers
    op_idx = [column headers] output column headers
    """
    df = pd.read_csv(file_path)
    return df[ip_idx].values, df[op_idx].values
