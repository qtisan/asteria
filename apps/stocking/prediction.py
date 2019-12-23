import sys
import os
import json
import re

from apps.stocking.actions import train, fetch, make, extends, split, preprocess
from apps.stocking.estimators import estimators
from apps.stocking.metainfo import logger

from sklearn.base import is_classifier

import numpy as np
import pandas as pd
import pickle
import moment


def hooked(infos, hooks=None):
    if hooks is not None:
        if isinstance(hooks, (list, tuple)):
            for hook in [h for h in hooks if callable(h)]:
                infos = hook(infos)
        elif callable(hooks):
            infos = hooks(infos)
        else:
            pass
    return infos


def prepare_data(infos,
                 ds,
                 categorify_y,
                 feature_keys,
                 df_opendate,
                 features_all=False):
    # making data x y
    if features_all:
        f_all = []
        for pow_i in infos['feature_pows']:
            powed_fs = np.char.add(feature_keys, '__{0}'.format(pow_i))
            f_all = np.concatenate((f_all, powed_fs))
        for chg_i in infos['feature_chgs']:
            chged_fs = np.char.add(feature_keys, '_chg_{0}'.format(chg_i))
            f_all = np.concatenate((f_all, chged_fs))

        infos['features'] = f_all.tolist()
    logger.debug('All features listed: \n')
    logger.debug(str(infos['features']))

    df_x, df_ys, df_x_latest, df_ys_latest = \
        make.make_xy(ds, return_Xy=True, to_numpy=False, categorify=categorify_y, **infos)
    x, ys, x_latest, ys_latest = \
        df_x.to_numpy(), df_ys.to_numpy(), df_x_latest.to_numpy(), df_ys_latest.to_numpy()
    xy_original = pd.concat([
        df_opendate,
        pd.concat([df_x_latest, df_x]),
        pd.concat([df_ys_latest, df_ys])
    ],
                            axis=1)
    return x, ys, x_latest, ys_latest, xy_original


def make_latest(future_days, index, y_latest, y_pred, y_dict, is_clf):
    fds = '未来{0}天'.format(future_days)
    vls = lambda y: [(y_dict[v]['name'] if is_clf else '{0:.2f}'.format(v))
                     for v in y]
    y_pred_names = np.char.add(fds, vls(preprocess.rav(y_pred)))
    y_latest_names = np.char.add(fds, vls(preprocess.rav(y_latest)))
    latest_prediction = pd.DataFrame(np.concatenate((np.expand_dims(
        y_pred_names, axis=1), np.expand_dims(y_latest_names, axis=1)),
                                                    axis=1),
                                     index=index,
                                     columns=['prediction', 'current'])
    return latest_prediction.to_dict('index')


def gen_feature_names(features, x_dict, x_dict_extra):
    # append feature names.
    feature_names = []
    fs_reg = '|'.join(x_dict.keys())
    patterns = [(re.compile(ent['suffix'].replace('feature', fs_reg)), ent['name'])
                for ent in x_dict_extra]
    for f in features:
        if f in x_dict.keys():
            feature_names.append(x_dict[f])
        else:
            for suff, name in patterns:
                ob = suff.search(f)
                if ob is not None:
                    _f, _d = ob.groups()
                    feature_names.append(name.format(x_dict[_f], _d))
    return feature_names


def predict(infos,
            info_hooks=None,
            use_default=False,
            save_files=False,
            return_latest_prediction=False,
            categorify_y=lambda yv: 0 if yv < 0 else 1,
            y_dict=[{
                'name': '下跌'
            }, {
                'name': '上涨'
            }],
            x_dict={},
            x_dict_extra={},
            features_all=False,
            *args,
            **kwargs):

    infos = hooked(infos, info_hooks)

    # fetch and extends data
    original_ds = fetch.fetch(**infos)
    ds = extends.extends_ds(original_ds, x_dict=x_dict, **infos)

    x, ys, x_latest, ys_latest, xy_original = \
        prepare_data(infos,
                     ds=ds,
                     categorify_y=categorify_y,
                     feature_keys=list(x_dict.keys()),
                     df_opendate=original_ds.iloc[:]['opendate'],
                     features_all=features_all)

    # prepare estimator
    est_info = estimators[infos['estimator']]
    estimator, param_grid, preproc, preproc_args = est_info['estimator'], est_info[
        'args'], None, {}
    if 'preproc' in est_info:
        pprc = est_info['preproc']
        if isinstance(pprc, dict):
            preproc = pprc['method']
            preproc_args = pprc['args']
        elif callable(pprc):
            preproc = pprc
    if use_default:
        param_grid = None

    # training.
    best_estimator, score, smaller_rate, smaller_rate_all, equal_rate_all, y_all, y_pred, y_latest = \
        train.fit(x,
                  ys,
                  estimator,
                  param_grid=param_grid,
                  x_latest=x_latest,
                  ys_latest=ys_latest,
                  pre_process=preproc,
                  pre_process_args=preproc_args,
                  test_size=infos['test_size'],
                  random_state=infos['random_state'],
                  t_t_split=split.tt_split)

    xy = pd.concat(
        [xy_original,
         pd.Series(np.concatenate((y_pred, y_all)), name='prediction')],
        axis=1)

    infos['results'] = {
        'best_estimator':
            str(best_estimator).replace('\n', '').replace(' ', ''),
        'score':
            score,
        'smaller_rate':
            smaller_rate,
        'smaller_rate_all':
            smaller_rate_all,
        'equal_rate_all':
            equal_rate_all,
        'sample_num':
            len(xy),
        'test_size':
            int(len(ys) * infos['test_size']),
        'feature_names':
            gen_feature_names(features=infos['features'],
                              x_dict=x_dict,
                              x_dict_extra=x_dict_extra)
    }

    val = (infos, best_estimator, xy_original, xy)

    # latest predictions
    if return_latest_prediction:
        lp = make_latest(
            future_days=infos['future_days'],
            index=original_ds.iloc[:infos['future_days']]['opendate'].to_numpy(),
            y_latest=y_latest,
            y_pred=y_pred,
            y_dict=y_dict,
            is_clf=is_classifier(best_estimator))
        infos['results']['latest_prediction'] = lp
        val = (*val, lp)

    return val
