import numpy as np

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.base import is_classifier, is_regressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from apps.stocking.metainfo import logger
from apps.stocking.actions.preprocess import no_norm, normalization, pure_features, compose_preprocessor
from apps.stocking.actions.split import tt_split
from apps.stocking.actions.categorify import y_categorifier

# TODO: Ignore future warning, as https://github.com/EpistasisLab/tpot/issues/981
import warnings
warnings.filterwarnings("ignore")
from tpot import TPOTClassifier, TPOTRegressor
from tpot.base import TPOTBase


def is_clfr(estimator):
    if isinstance(estimator, TPOTBase):
        return estimator.classification
    else:
        return is_classifier(estimator)


estimators = {
    'TPOTClassifier': {
        'estimator':
            TPOTClassifier(generations=2,
                           offspring_size=5,
                           population_size=10,
                           verbosity=2,
                           random_state=42)
    },
    'TPOTRegressor': {
        'estimator':
            TPOTRegressor(generations=2,
                          offspring_size=5,
                          population_size=10,
                          verbosity=2,
                          random_state=42)
    },
    'MLPClassifier': {
        'estimator': MLPClassifier(alpha=0.01, max_iter=1000),
        'args': {
            'alpha': [0.02, 0.1],
            'max_iter': [1000, 2000],
        }
    },
    'SVC': {
        'estimator': SVC(kernel='rbf', C=10),
        'args': {
            'kernel': ['sigmoid', 'rbf'],
            'C': [1, 3, 5, 10],
            'gamma': [2, 3, 5]
        },
        'preproc': normalization
    },
    'GaussianProcess': {
        'estimator': GaussianProcessClassifier(1.0 * RBF(1.0)),
        'args': {
            'kernel': [1.0 * RBF(1.0), 1.0 * RationalQuadratic(0.5)],
            'optimizer': ['fmin_l_bfgs_b'],
            'max_iter_predict': [100, 200, 500]
        }
    },
    'RandomForestClassifier': {
        'estimator': RandomForestClassifier(max_depth=500, n_estimators=1000),
        'args': {
            'n_estimators': [100, 1000, 5000],
            'max_depth': [500, 1200]
        },
        'preproc': {
            'method': pure_features,
            'args': {
                'k': 20
            }
        }
    },
    'AdaBoostClassifier': {
        'estimator':
            AdaBoostClassifier(n_estimators=2000, learning_rate=1e-1,
                               random_state=29),
        'args': {
            'n_estimators': [100, 1000, 2000],
            'learning_rate': [1, 1e-1, 1e-2, 1e-3]
        },
        'preproc':
            no_norm
    },
    'GaussianNB': {
        'estimator': GaussianNB(var_smoothing=0.1),
        'args': {
            'var_smoothing': [1e-9, 1e-10]
        },
        'preproc': normalization
    },
    'RandomForestRegressor': {
        'estimator':
            RandomForestRegressor(n_estimators=2000, max_depth=500, random_state=39),
        'args': {
            'n_estimators': [100, 1000, 3000],
            'max_depth': [10, 100, 1000],
        },
        'preproc':
            normalization
    }
}
