import sys
from pathlib import Path
sys.path.append(str(Path(__file__, '../../').resolve()))

# import os
# import numpy as np
# import pandas as pd
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

# from matplotlib import pyplot as plt, cm
# from matplotlib.collections import LineCollection

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

# 分类名

# iris = datasets.load_iris()
# clfr = svm.SVC()
# X, y, cns = iris.data, iris.target, iris.target_names[iris.target]
# clfr.fit(X, cns)
# print(list(clfr.predict(X[-4:])))

# 分类二元化
# 多标签
# X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
# y = [1, 2, 3, 2, 1]
# clfr = OneVsRestClassifier(estimator=svm.SVC(random_state=0))
# pred = clfr.fit(X, y).predict(X)
# print(pred, clfr.classes_)
# y = LabelBinarizer().fit_transform(y)
# pred = clfr.fit(X, y).predict(X)
# print(pred)
# y = [[1, 2], [3, 2, 5], [1, 2, 3, 4, 5], [3, 4], [1]]
# y = MultiLabelBinarizer().fit_transform(y)
# pred = clfr.fit(X, y).predict(X)
# print(pred)

# # ============================================================================
# # Define a pipeline to search for the best combination of PCA truncation
# # and classifier regularization.
# pca = PCA()
# # set the tolerance to a large value to make the example faster
# logistic = LogisticRegression(max_iter=200, tol=0.1)
# pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# X_digits, y_digits = datasets.load_digits(return_X_y=True)

# # Parameters of pipelines can be set using ‘__’ separated parameter names:
# param_grid = {
#     'pca__n_components': [5, 15, 30, 45, 47, 64],
#     'logistic__C': np.logspace(-4, 4, 4),
# }
# search = GridSearchCV(pipe, param_grid, n_jobs=-1)
# search.fit(X_digits, y_digits)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)

# # Plot the PCA spectrum
# pca.fit(X_digits)

# dd = {'a': 1, 'b': 2}

# ks = list(dd.keys())

# df = pd.DataFrame(
#     [[3, 0], [0.5, 6], [60, 0.002], [23, 2000], [1, 0], [3, 0], [2, 739939200002],
#      [0.000004, 0.0000000203920000], [0, 100020203039]],
#     columns=ks)
# bs = df.loc[:, ks].to_numpy()

# end
# input('press any key to end.')

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

from pathlib import Path
import pandas as pd
import numpy as np

p = Path(__file__).parent.parent / 'apps/stocking/data/trained/train-test-1/xy.csv'

df = pd.read_csv(p, index_col=0)
df = df.fillna(1)

l = len(df)
ra = df.iloc[:]['range'].to_numpy()
pr = df.iloc[:]['prediction'].to_numpy()
rr = pr <= ra
print('mean:{0:.2f}'.format(np.mean(rr) * 100),
      '{0}/{1}'.format(np.count_nonzero(rr), len(rr)))
# df = pd.DataFrame(np.concatenate((np.expand_dims(
#     [1, 2, 3, 4, 5, 6, 7], axis=1), np.expand_dims([4, 5, 6, 1, 2, 3, 4], axis=1)),
#                                  axis=1),
#                   index=[1, 2, 3, 4, 5, 6, 7],
#                   columns=['x', 'y'])
# df2 = df.iloc[-2:3]
# print(df2)
