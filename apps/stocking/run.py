from apps.stocking.actions.categorify import y_categorifier
from apps.stocking.prediction import predict
from apps.stocking.metainfo import appdir, TRAINED_ROOT
from apps.stocking.estimators import estimators

from sklearn.base import is_classifier

import time
import json
import moment
import os
import pickle


def get_settings():
    settings = None
    with open(appdir / 'settings.json') as f:
        settings = json.load(f)
    return settings


def save_infos(infos, estimator, ds, xy, dirname='latest'):
    info_dir = TRAINED_ROOT / infos['name'] / dirname
    if not info_dir.exists():
        os.makedirs(info_dir)
    est_path = info_dir / 'estimator.pkl'
    infos['type'] = 'classify' if is_classifier(estimator) else 'regress'

    with open(est_path, 'wb') as f:
        pickle.dump(estimator, f)
    with open(info_dir / 'infos.json', 'w') as f:
        json.dump(infos, f, ensure_ascii=False, sort_keys=True, indent=2)
    ds.to_csv(info_dir / 'ds.csv')
    xy.to_csv(info_dir / 'xy.csv')


def predict_batch(settings):
    if isinstance(settings, dict):
        stocks = settings['stocks']
        features_all = settings['features_all']

        infos = {}
        for k in [
                'features', 'past_days', 'future_days', 'test_size', 'random_state',
                'feature_pows', 'feature_chgs', 'zeros_copy_days', 'estimator',
                'estimator_default'
        ]:
            infos[k] = settings[k]

        y_dict = sorted(settings['y_dict'], key=lambda _yi: _yi['threshold'])
        x_dict = settings['x_dict']
        x_dict_extra = settings['x_dict_extra']
        use_default = settings['estimator_default']

        classify_y = y_categorifier(y_dict)

        for code in stocks:
            infos['name'] = 'infos-{0}'.format(code)
            infos['code'] = code
            infos['time'] = moment.now().format('YYYY-MM-DD HH:mm:ss')

            inf, est, ds, xy, lp = predict(infos,
                                           categorify_y=classify_y,
                                           use_default=use_default,
                                           y_dict=y_dict,
                                           x_dict=x_dict,
                                           x_dict_extra=x_dict_extra,
                                           return_latest_prediction=True,
                                           save_files=True,
                                           features_all=features_all)

            print('Predict last {0} days:'.format(inf['future_days']))
            print(lp)

            save_infos(infos, est, ds, xy)
            time.sleep(1.0)
    else:
        raise 'Settings not found!'


def run_one(code, estimator=None):
    settings = get_settings()
    settings['stocks'] = [code]
    if (estimator is not None) and (estimator in estimators):
        settings['estimator'] = estimator
        settings['estimator_default'] = True
    predict_batch(settings)


def run_batch(codes=None):
    settings = get_settings()
    if isinstance(codes, list):
        settings['stocks'] = codes
    predict_batch(settings)
