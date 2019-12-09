import sys
from pathlib import Path
sys.path.append(str(Path(__file__, '../../').resolve()))

import moment

from apps.stocking.prediction import predict

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

infos = {
    'name': 'train-test-1',
    'code': 'sz002415',
    'timestamp': moment.now().format('YYYYMMDD_HHmmss'),
    'features': [
        'netamount_chg_1', 'ratioamount_chg_1', 'r0_net_chg_1', 'r0_ratio_chg_1',
        'r0x_ratio__4', 'close_chg_1', 'high_chg_1', 'high_chg_2', 'high_chg_3',
        'low_chg_1', 'low_chg_2', 'low_chg_3', 'open_chg_3', 'yest_chg_3',
        'turnover__3', 'volumn', 'amount', 'circul_chg_10', 'volumn__4', 'low_chg_1',
        'low_chg_2'
    ],
    'past_days': 5,
    'future_days': 10,
    'test_size': 0.05,
    'random_state': 39,
    'feature_pows': [2, 3, 4],
    'feature_chgs': [1, 2, 3, 5, 10],
    'zeros_copy_days': 10,
    'classifier': 'SVC'
}

if __name__ == "__main__":
    predict(infos, estimator=SVC(gamma=4, C=2))

# import os
# import json
# from apps.stocking.actions import train, fetch, make, extends
# from apps.stocking.classifiers import clfs
# from apps.stocking.meta import y_dict_name, categorify_y, TRAINED_ROOT, normalization
# from apps.stocking import logger

# import numpy as np
# import pandas as pd
# import sklearn
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.pipeline import Pipeline

# import pickle
# import moment

# infos = {
#     'name': 'train-test-1',
#     'code': 'sz002415',
#     'timestamp': moment.now().format('YYYYMMDD_HHmmss'),
#     'features': [
#         'netamount_chg_1', 'ratioamount_chg_1', 'r0_net_chg_1', 'r0_ratio_chg_1',
#         'r0x_ratio__4', 'close_chg_1', 'high_chg_1', 'high_chg_2', 'high_chg_3',
#         'low_chg_1', 'low_chg_2', 'low_chg_3', 'open_chg_3', 'yest_chg_3',
#         'turnover__3', 'volumn', 'amount', 'circul_chg_10', 'volumn__4', 'low_chg_1',
#         'low_chg_2'
#     ],
#     'past_days': 5,
#     'future_days': 10,
#     'test_size': 0.05,
#     'random_state': 39,
#     'feature_pows': [2, 3, 4],
#     'feature_chgs': [1, 2, 3, 5, 10],
#     'zeros_copy_days': 10,
#     'classifier': 'SVC'
# }

# info_dir = TRAINED_ROOT / infos['name']
# if not info_dir.exists():
#     os.mkdir(info_dir)
# clf_path = info_dir / 'classifier.pkl'

# original_ds = fetch.fetch(**infos)
# ds = extends.extends_ds(original_ds, **infos)

# df_x, df_y, df_x_latest, df_y_latest = \
#     make.make_xy(ds, return_Xy=True, to_numpy=False, categorify=categorify_y, **infos)
# x, y, x_latest, y_latest = \
#     df_x.to_numpy(), df_y.to_numpy(), df_x_latest.to_numpy(), df_y_latest.to_numpy()
# pd.concat([
#     original_ds.iloc[:]['opendate'],
#     pd.concat([df_x_latest, df_x]),
#     pd.concat([df_y_latest, df_y])
# ],
#           axis=1).to_csv(info_dir / 'xy_original.csv')

# clf_info = clfs[infos['classifier']]
# # classifier, param_grid = clf_info['clf'], clf_info['args']
# # classifier, param_grid = \
# #     sklearn.neural_network.MLPClassifier(alpha=0.02, max_iter=1000), None
# classifier, param_grid = \
#     sklearn.svm.SVC(gamma=4, C=0.1), None

# clf, score, smaller_rate, y_valid, y_pred = \
#     train.train(x, y, classifier, param_grid=param_grid, x_latest=x_latest, pre_process=normalization, **infos)

# y_pred_names = np.char.add('{0}d '.format(infos['future_days']),
#                            [y_dict_name[v] for v in y_pred])
# y_latest_names = np.char.add('{0}d '.format(infos['future_days']),
#                              [y_dict_name[v] for v in y_latest.ravel()])
# print('Predict last {0} days:'.format(infos['future_days']))
# print(
#     pd.DataFrame(np.concatenate((np.expand_dims(
#         y_pred_names, axis=1), np.expand_dims(y_latest_names, axis=1)),
#                                 axis=1),
#                  index=original_ds.iloc[:infos['future_days']]['opendate'].to_numpy(),
#                  columns=['prediction', 'current']))

# infos['results'] = {
#     'classifier': str(clf),
#     'trained_classifier': str(clf_path),
#     'score': score,
#     'smaller_rate': smaller_rate
# }

# with open(clf_path, 'wb') as f:
#     pickle.dump(clf, f)
# with open(info_dir / 'infos.json', 'w') as f:
#     json.dump(infos, f)

# pd.concat([
#     original_ds.iloc[:]['opendate'],
#     pd.concat([df_x_latest, df_x]),
#     pd.concat([df_y_latest, df_y]),
#     pd.Series(np.concatenate((y_pred, y_valid)), name='prediction')
# ],
#           axis=1).to_csv(info_dir / 'xy.csv')
