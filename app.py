# usr/bin/env python3

import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.insert(a, 0, [9, 8], axis=1)

print(b)
