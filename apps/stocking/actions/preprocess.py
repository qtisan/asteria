from apps.stocking.metainfo import logger
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, SelectFdr
from sklearn.preprocessing import StandardScaler

import numpy as np


def rav(y: np.ndarray):
    s = len(y.shape)
    if s == 1:
        return y.ravel()
    elif s == 0:
        return y
    else:
        raise 'y shape {0} not correct.'.format(y.shape)


def normalization(estimator, x, y, x_latest, y_latest, **kwargs):
    x_all = np.concatenate((x_latest, x), axis=0)
    logger.debug('Start normalizing features...')
    x_all_norm = StandardScaler().fit_transform(x_all)
    xl = len(x_latest)
    return estimator, x_all_norm[xl:], rav(y), x_all_norm[:xl], rav(y_latest)


def pure_features(estimator, x, y, x_latest, y_latest, k=10, **kwargs):
    x_all = np.concatenate((x_latest, x), axis=0)
    y_all = np.concatenate((y_latest, y), axis=0)
    y_all = np.nan_to_num(y_all)
    x_all_new = SelectKBest(mutual_info_classif, k=k).fit_transform(x_all, rav(y_all))

    logger.debug('X select k best result: from shape {0} to {1}'.format(
        x_all.shape, x_all_new.shape))
    xl = len(x_latest)
    return estimator, x_all_new[xl:], rav(y), x_all_new[:xl], rav(y_latest)


def no_norm(estimator, x, y, x_latest, y_latest, **kwargs):
    return estimator, x, rav(y), x_latest, rav(y_latest)


def compose_preprocessor(*procs, **kwargs):
    def preprocess(*args):
        res_tup = args
        for p in [_p for _p in procs if callable(_p)]:
            res_tup = p(*res_tup, **kwargs)
        return res_tup

    return preprocess
