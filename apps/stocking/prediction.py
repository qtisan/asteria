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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

clfs = {
    'MLPClassifier': {
        'clf': MLPClassifier(alpha=0.1, max_iter=200),
        'args': [{
            'alpha': [0.02, 0.1],
            'max_iter': [1000, 2000],
        }]
    },
    'SVC': {
        'clf':
            SVC(kernel='sigmoid', C=1),
        'args': [{
            'kernel': ['linear', 'rbf'],
            'C': [1, 3, 5, 10],
            'gamma': [2, 3, 5]
        }]
    }
}


def normalization(x, y, x_latest):
    x_all = np.concatenate((x_latest, x), axis=0)
    x_all_norm = StandardScaler().fit_transform(x_all)
    xl = len(x_latest)
    return x_all_norm[xl:], y.ravel(), x_all_norm[:xl]


def predict(infos, estimator=None):

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
    pd.concat([
        original_ds.iloc[:]['opendate'],
        pd.concat([df_x_latest, df_x]),
        pd.concat([df_y_latest, df_y])
    ],
              axis=1).to_csv(info_dir / 'xy_original.csv')

    clf_info = clfs[infos['classifier']]
    classifier, param_grid = clf_info['clf'], clf_info['args']

    if estimator:
        classifier, param_grid = estimator, None

    clf, score, smaller_rate, y_valid, y_pred = \
        train.train(x, y, classifier, param_grid=param_grid, x_latest=x_latest, pre_process=normalization, **infos)

    y_pred_names = np.char.add('{0}d '.format(infos['future_days']),
                               [y_dict_name[v] for v in y_pred])
    y_latest_names = np.char.add('{0}d '.format(infos['future_days']),
                                 [y_dict_name[v] for v in y_latest.ravel()])
    print('Predict last {0} days:'.format(infos['future_days']))
    print(
        pd.DataFrame(
            np.concatenate((np.expand_dims(
                y_pred_names, axis=1), np.expand_dims(y_latest_names, axis=1)),
                           axis=1),
            index=original_ds.iloc[:infos['future_days']]['opendate'].to_numpy(),
            columns=['prediction', 'current']))

    infos['results'] = {
        'classifier': str(clf),
        'trained_classifier': str(clf_path),
        'score': score,
        'smaller_rate': smaller_rate
    }

    with open(clf_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(info_dir / 'infos.json', 'w') as f:
        json.dump(infos, f)

    pd.concat([
        original_ds.iloc[:]['opendate'],
        pd.concat([df_x_latest, df_x]),
        pd.concat([df_y_latest, df_y]),
        pd.Series(np.concatenate((y_pred, y_valid)), name='prediction')
    ],
              axis=1).to_csv(info_dir / 'xy.csv')

    return infos