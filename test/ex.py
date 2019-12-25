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
import warnings
warnings.filterwarnings("ignore")

from tpot import TPOTClassifier, TPOTRegressor
from tpot.base import TPOTBase
tp = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)

print(isinstance(tp, TPOTBase))
