import sys
from pathlib import Path
sys.path.append(str(Path(__file__, '../../').resolve()))

import os
import numpy as np
import pandas as pd
# import requests
# import pickle
# import moment

# from sklearn import svm, datasets, random_projection, cluster, covariance, manifold
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.datasets import fetch_openml
# from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
# from sklearn.gaussian_process.kernels \
#     import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel as C
# from sklearn.base import is_classifier, is_regressor

l1 = [12, 3, 4, 5]
l2 = [43, 1, 221]

print(l1 + l2)
