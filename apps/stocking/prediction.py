import sys
import os
import json
import re

from apps.stocking.actions import train, fetch, make, extends
from apps.stocking.classifiers import clfs, normalization, tt_split

from apps.stocking.metainfo import logger, TRAINED_ROOT

import numpy as np
import pandas as pd
import pickle
import moment


def predict(infos,
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
            features_all=True):

    original_ds = fetch.fetch(**infos)
    ds = extends.extends_ds(original_ds, x_dict=x_dict, **infos)

    if features_all:
        f_all = []
        fss = list(x_dict.keys())
        for pow_i in infos['feature_pows']:
            powed_fs = np.char.add(fss, '__{0}'.format(pow_i))
            f_all = np.concatenate((f_all, powed_fs))
        for chg_i in infos['feature_chgs']:
            chged_fs = np.char.add(fss, '_chg_{0}'.format(chg_i))
            f_all = np.concatenate((f_all, chged_fs))

        infos['features'] = f_all.tolist()
    logger.debug('All features listed: \n')
    logger.debug(str(infos['features']))

    df_x, df_y, df_x_latest, df_y_latest, df_y_vc, df_y_vc_latest = \
        make.make_xy(ds, return_Xy=True, to_numpy=False, categorify=categorify_y, **infos)
    x, y, x_latest, y_latest = \
        df_x.to_numpy(), df_y.to_numpy(), df_x_latest.to_numpy(), df_y_latest.to_numpy()
    xy_original = pd.concat([
        original_ds.iloc[:]['opendate'],
        pd.concat([df_x_latest, df_x]),
        pd.concat([df_y_latest, df_y]),
        pd.concat([df_y_vc_latest, df_y_vc])
    ],
                            axis=1)

    clf_info = clfs[infos['classifier']]
    classifier, param_grid, norm = clf_info['clf'], clf_info['args'], normalization
    if 'norm' in clf_info:
        norm = clf_info['norm']

    if use_default:
        param_grid = None

    clf, score, smaller_rate, smaller_rate_all, equal_rate_all, y_all, y_pred = \
        train.train(x, y, classifier, param_grid=param_grid, t_t_split=tt_split,
                     x_latest=x_latest, y_latest=y_latest, pre_process=norm, **infos)

    xy = pd.concat(
        [xy_original,
         pd.Series(np.concatenate((y_pred, y_all)), name='prediction')],
        axis=1)

    infos['results'] = {
        'classifier': str(clf).replace('\n', '').replace(' ', ''),
        'score': score,
        'smaller_rate': smaller_rate,
        'smaller_rate_all': smaller_rate_all,
        'equal_rate_all': equal_rate_all,
        'sample_num': len(xy),
        'test_size': int(len(y) * infos['test_size'])
    }
    feature_names = []
    fs_reg = '|'.join(x_dict.keys())
    patterns = [(re.compile(ent['suffix'].replace('feature', fs_reg)), ent['name'])
                for ent in x_dict_extra]
    for f in infos['features']:
        if f in x_dict.keys():
            feature_names.append(x_dict[f])
        else:
            for suff, name in patterns:
                ob = suff.search(f)
                if ob is not None:
                    _f, _d = ob.groups()
                    feature_names.append(name.format(x_dict[_f], _d))
    infos['results']['feature_names'] = feature_names

    val = infos

    if return_latest_prediction:
        y_pred_names = np.char.add('未来{0}天'.format(infos['future_days']),
                                   [y_dict[v]['name'] for v in y_pred])
        y_latest_names = np.char.add('未来{0}天'.format(infos['future_days']),
                                     [y_dict[v]['name'] for v in y_latest.ravel()])
        latest_prediction = pd.DataFrame(
            np.concatenate((np.expand_dims(
                y_pred_names, axis=1), np.expand_dims(y_latest_names, axis=1)),
                           axis=1),
            index=original_ds.iloc[:infos['future_days']]['opendate'].to_numpy(),
            columns=[
                'prediction_{0}'.format(infos['code']),
                'current_{0}'.format(infos['code'])
            ])
        infos['results']['latest_prediction'] = latest_prediction.to_dict('index')
        val = (val, latest_prediction)

    if save_files:
        info_dir = TRAINED_ROOT / infos['name']
        if not info_dir.exists():
            os.mkdir(info_dir)
        clf_path = info_dir / 'classifier.pkl'

        infos['results']['trained_classifier'] = str(clf_path)
        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f)
        with open(info_dir / 'infos.json', 'w') as f:
            json.dump(infos, f, ensure_ascii=False, sort_keys=True, indent=4)
        xy_original.to_csv(info_dir / 'xy_original.csv')
        xy.to_csv(info_dir / 'xy.csv')

    return val
