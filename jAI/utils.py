
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np


def save_weights_temp(w, b):
    weight = np.array(w)
    bias = np.array(b)
    np.save('temp_out_weights', weight)
    np.save('temp_out_bias', bias)


def load_weights_temp():
    w = np.load('temp_out_weights.npy')
    b = np.load('temp_out_bias.npy')

    return w.tolist(), b.tolist()


def save_weights(w, b, filepath):
    data = json.dumps({"weight": w, "bias": b})
    with open(filepath, 'w') as f:
        f.write(data)


def load_weights(filepath):
    with open(filepath, 'r') as f:
        s = f.read()
        data = json.loads(s)
        return data["weights"], data["bias"]
