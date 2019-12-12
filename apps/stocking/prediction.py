import sys
import os
import json
from pathlib import Path

from apps.stocking.actions import train, fetch, make, extends
from apps.stocking.meta import y_dict_name, categorify_y, TRAINED_ROOT
from apps.stocking import logger

import numpy as np
import pandas as pd
import pickle
import moment

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def normalization(clf, x, y, x_latest):
    x_all = np.concatenate((x_latest, x), axis=0)
    x_all_norm = StandardScaler().fit_transform(x_all)
    xl = len(x_latest)
    return clf, x_all_norm[xl:], y.ravel(), x_all_norm[:xl]


def pure_features(clf, x, y, x_latest):
    return clf, x, y.ravel(), x_latest


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
            SVC(kernel='sigmoid', C=1),
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
            RandomForestClassifier(max_depth=100,
                                   n_estimators=2000,
                                   min_samples_split=5),
        'args': [{
            'n_estimators': [10, 100, 250],
            'max_depth': [5, 12],
            'max_features': [2, 15, 50]
        }],
        'norm':
            pure_features
    },
    'GaussianNB': {
        'clf': GaussianNB(var_smoothing=1e-3),
        'args': [{
            'var_smoothing': [1e-9, 1e-10]
        }]
    }
}


def predict(infos,
            use_default=False,
            save_files=False,
            return_latest_prediction=False):

    info_dir = TRAINED_ROOT / infos['name']
    if not info_dir.exists():
        os.mkdir(info_dir)
    clf_path = info_dir / 'classifier.pkl'

    original_ds = fetch.fetch(**infos)
    ds = extends.extends_ds(original_ds, **infos)

    df_x, df_y, df_x_latest, df_y_latest = \
        make.make_xy(ds, return_Xy=True, to_numpy=False, categorify=categorify_y, **infos)
    x, y, x_latest, y_latest = \
        df_x.to_numpy(), df_y.to_numpy(), df_x_latest.to_numpy(), df_y_latest.to_numpy()
    xy_original = pd.concat([
        original_ds.iloc[:]['opendate'],
        pd.concat([df_x_latest, df_x]),
        pd.concat([df_y_latest, df_y])
    ],
                            axis=1)

    clf_info = clfs[infos['classifier']]
    classifier, param_grid, norm = clf_info['clf'], clf_info['args'], normalization
    if 'norm' in clf_info:
        norm = clf_info['norm']

    if use_default:
        param_grid = None

    clf, score, smaller_rate, y_valid, y_pred = \
        train.train(x, y, classifier, param_grid=param_grid, x_latest=x_latest, pre_process=normalization, **infos)

    xy = pd.concat([
        original_ds.iloc[:]['opendate'],
        pd.concat([df_x_latest, df_x]),
        pd.concat([df_y_latest, df_y]),
        pd.Series(np.concatenate((y_pred, y_valid)), name='prediction')
    ],
                   axis=1)

    infos['results'] = {
        'classifier': str(clf),
        'score': score,
        'smaller_rate': smaller_rate
    }

    val = infos

    if return_latest_prediction:
        y_pred_names = np.char.add('{0}d '.format(infos['future_days']),
                                   [y_dict_name[v] for v in y_pred])
        y_latest_names = np.char.add('{0}d '.format(infos['future_days']),
                                     [y_dict_name[v] for v in y_latest.ravel()])
        latest_prediction = pd.DataFrame(
            np.concatenate((np.expand_dims(
                y_pred_names, axis=1), np.expand_dims(y_latest_names, axis=1)),
                           axis=1),
            index=original_ds.iloc[:infos['future_days']]['opendate'].to_numpy(),
            columns=[
                'prediction_{0}'.format(infos['code']),
                'current_{0}'.format(infos['code'])
            ])
        infos['results']['latest_prediction'] = latest_prediction.to_json(
            orient='index', force_ascii=False)
        val = (val, latest_prediction)

    if save_files:
        infos['results']['trained_classifier'] = str(clf_path)
        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f)
        with open(info_dir / 'infos.json', 'w') as f:
            json.dump(infos, f)
        xy_original.to_csv(info_dir / 'xy_original.csv')
        xy.to_csv(info_dir / 'xy.csv')

    return val