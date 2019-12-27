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

from typing import TypeVar
from abc import ABCMeta, abstractmethod


class Grand(metaclass=ABCMeta):
    @abstractmethod
    def say(self):
        pass


class Father(Grand):
    def get_fn(self):
        return 'Zhang'

    def get_name(self):
        return 'Taoni'


class Child(Father):
    def say(self):
        return 'i am child.'

    def get_name(self):
        return 'Bu'


T = TypeVar('T')


def who(f: T) -> None:
    if isinstance(f, str):
        print(f)
    else:
        print(f.get_fn())


def create_grand():
    class GA(Grand):
        def say(self):
            print('i am grand.')

    return GA()


zb = Child()

who(zb)
who('Li')
ga = create_grand()
ga.say()