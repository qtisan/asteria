import numpy as np

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from apps.stocking.metainfo import logger
from apps.stocking.actions.preprocess import no_norm, normalization, pure_features, compose_preprocessor
from apps.stocking.actions.split import tt_split
from apps.stocking.actions.categorify import y_categorifier

estimators = {
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
        'estimator':
            RandomForestClassifier(max_depth=500, n_estimators=1000,
                                   max_features=150),
        'args': {
            'n_estimators': [100, 1000, 5000],
            'max_depth': [500, 1200],
            'max_features': [10, 50, 150, 280]
        },
        'preproc': {
            'method': pure_features,
            'args': {
                'k': 270
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
