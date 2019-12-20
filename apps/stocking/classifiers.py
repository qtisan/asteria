import numpy as np

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, SelectFdr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from apps.stocking.metainfo import logger


def normalization(clf, x, y, x_latest, y_latest):
    x_all = np.concatenate((x_latest, x), axis=0)
    x_all_norm = StandardScaler().fit_transform(x_all)
    xl = len(x_latest)
    return clf, x_all_norm[xl:], y.ravel(), x_all_norm[:xl], y_latest.ravel()


def pure_features(clf, x, y, x_latest, y_latest):
    x_all = np.concatenate((x_latest, x), axis=0)
    y_all = np.concatenate((y_latest, y), axis=0)
    x_all_new = SelectKBest(mutual_info_classif,
                            k=66).fit_transform(x_all, y_all.ravel())

    logger.debug('X select k best result: from shape {0} to {1}'.format(
        x_all.shape, x_all_new.shape))
    xl = len(x_latest)
    return clf, x_all_new[xl:], y.ravel(), x_all_new[:xl], y_latest.ravel()


def no_norm(clf, x, y, x_latest, y_latest):
    return clf, x, y.ravel(), x_latest, y_latest.ravel()


def tt_split(x, y, test_size=0.2, *args, **kwargs):
    length = len(y)
    train_length = length - int(test_size * length)
    start = length - train_length
    x_train = x[start:]
    y_train = y[start:]
    x_test = x[:start]
    y_test = y[:start]
    return x_train, x_test, y_train, y_test


def y_categorifier(y_dict):
    '''
    y_dict sample:
    [{
        "name": "下跌",
        "threshold": -9999
    }, {
        "name": "上涨",
        "threshold": 0
    }]
    '''
    def classify_y(y_value):
        c = 0
        for i, yd in enumerate(y_dict):
            if y_value > yd['threshold']:
                c = i
        return c

    return classify_y


clfs = {
    'MLPClassifier': {
        'clf': MLPClassifier(alpha=0.01, max_iter=1000),
        'args': [{
            'alpha': [0.02, 0.1],
            'max_iter': [1000, 2000],
        }]
    },
    'SVC': {
        'clf':
            SVC(kernel='sigmoid', C=10),
        'args': [{
            'kernel': ['sigmoid', 'rbf'],
            'C': [1, 3, 5, 10],
            'gamma': [2, 3, 5]
        }]
    },
    'GaussianProcess': {
        'clf':
            GaussianProcessClassifier(1.0 * RBF(1.0)),
        'args': [{
            'kernel': [1.0 * RBF(1.0), 1.0 * RationalQuadratic(0.5)],
            'optimizer': ['fmin_l_bfgs_b'],
            'max_iter_predict': [100, 200, 500]
        }]
    },
    'RandomForestClassifier': {
        'clf':
            RandomForestClassifier(max_depth=200,
                                   n_estimators=3000,
                                   min_samples_split=15),
        'args': [{
            'n_estimators': [10, 100, 250],
            'max_depth': [5, 12],
            'max_features': [2, 15, 50]
        }],
        'norm':
            no_norm
    },
    'AdaBoostClassifier': {
        'clf':
            AdaBoostClassifier(n_estimators=2000, learning_rate=1e-1,
                               random_state=29),
        'args': [{
            'n_estimators': [100, 1000, 2000],
            'learning_rate': [1, 1e-1, 1e-2, 1e-3]
        }],
        'norm':
            no_norm
    },
    'GaussianNB': {
        'clf': GaussianNB(var_smoothing=1e-3),
        'args': [{
            'var_smoothing': [1e-9, 1e-10]
        }]
    }
}