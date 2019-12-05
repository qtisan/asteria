import sys
from pathlib import Path
sys.path.append(str(Path(__file__, '../../').resolve()))

import numpy as np
from sklearn import random_projection

# 数据降维
# rng = np.random.RandomState(0)
# X = rng.rand(10, 2000)
# X = np.array(X, dtype='float32')

# print(X)
# print(X.shape)
# tfmr = random_projection.GaussianRandomProjection()
# X1 = tfmr.fit_transform(X)
# print(X1)
# print(X1.shape)