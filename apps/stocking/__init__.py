# Stocking.  Dec 4, 2019

import os
import moment
import json
import time
import pandas as pd

from apps.stocking.prediction import predict
from apps.stocking.classifiers import y_categorifier, clfs
from apps.stocking.metainfo import *


def get_settings():
    settings = None
    with open(appdir / 'settings.json') as f:
        settings = json.load(f)
    return settings


def predict_batch(settings):
    if isinstance(settings, dict):
        stocks = settings['stocks']
        features_all = settings['features_all']

        infos = {}
        for k in [
                'features', 'past_days', 'future_days', 'test_size', 'random_state',
                'feature_pows', 'feature_chgs', 'zeros_copy_days', 'classifier'
        ]:
            infos[k] = settings[k]

        y_dict = sorted(settings['y_dict'], key=lambda _yi: _yi['threshold'])
        x_dict = settings['x_dict']
        x_dict_extra = settings['x_dict_extra']

        classify_y = y_categorifier(y_dict)

        for code in stocks:
            infos['name'] = 'infos-{0}'.format(code)
            infos['code'] = code
            infos['time'] = moment.now().format('YYYY-MM-DD HH:mm:ss')

            inf, lp = predict(infos,
                              categorify_y=classify_y,
                              use_default=True,
                              y_dict=y_dict,
                              x_dict=x_dict,
                              x_dict_extra=x_dict_extra,
                              return_latest_prediction=True,
                              save_files=True,
                              features_all=features_all)

            print('Predict last {0} days:'.format(inf['future_days']))
            print(lp)

            time.sleep(1.0)
    else:
        raise 'Settings not found!'


def run_one(code, classifier=None):
    settings = get_settings()
    settings['stocks'] = [code]
    if (classifier is not None) and (classifier in clfs):
        settings['classifier'] = classifier
    predict_batch(settings)


def run_batch(codes=None):
    settings = get_settings()
    if isinstance(codes, list):
        settings['stocks'] = codes
    predict_batch(settings)


def read_infos(code):
    infos = None
    with open(datadir / 'trained/infos-{0}/infos.json'.format(code)) as f:
        infos = json.load(f)
    return infos


def read_original_data(code):
    od = pd.read_csv(datadir / 'original/{0}.csv'.format(code), index_col=0)
    return od.to_dict('index')


def read_xyy(code):
    od = pd.read_csv(datadir / 'trained/infos-{0}/xy.csv'.format(code), index_col=0)
    return od.to_dict('index')
