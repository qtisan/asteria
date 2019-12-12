import sys
from pathlib import Path
sys.path.append(str(Path(__file__, '../../').resolve()))

import moment
import json
import time
import pandas as pd

from apps.stocking.prediction import predict

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

stocks = [
    'sh600004',
    'sh600009',
    'sh600011',
    'sh600029',
    'sh600030',
    'sh600061',
    'sh600111',
    'sh600177',
    'sh600309',
    'sh600350',
    'sh600535',
    'sh600588',
    'sh600606',
    'sh600845',
    'sh600887',
    'sh601006',
    'sh601857',
    'sh601877',
    'sz000338',
    'sz000789',
    'sz002340',
    'sz002415',
    'sz002475',
    'sz002690',
]

infos = {
    'name': 'temp',
    'code': 'sh600004',
    'time': moment.now().format('YYYY-MM-DD HH:mm:ss'),
    'features': [
        'netamount_chg_1', 'ratioamount_chg_1', 'r0_net_chg_1', 'r0_ratio_chg_1',
        'r0x_ratio', 'r0x_ratio__4', 'close_chg_1', 'high_chg_1', 'high_chg_3',
        'high_chg_5', 'low_chg_1', 'low_chg_2', 'low_chg_5', 'open_chg_3',
        'yest_chg_3', 'turnover__3', 'circul_chg_10', 'volumn__4', 'low_chg_1',
        'low__3'
    ],
    'past_days': 5,
    'future_days': 10,
    'test_size': 0.05,
    'random_state': 29,
    'feature_pows': [2, 3, 4],
    'feature_chgs': [1, 2, 3, 5, 10],
    'zeros_copy_days': 10,
    'classifier': 'RandomForestClassifier'
}

if __name__ == "__main__":
    # predict(infos)
    # inf, lp = predict(infos, use_default=True, return_latest_prediction=True)

    # print('Predict last {0} days:'.format(inf['future_days']))
    # print(lp)

    temp_dir = Path(__file__).parent.parent / 'apps/stocking/data/trained/temp'
    df: pd.DataFrame = None

    for code in stocks:
        infos['code'] = code
        infos_file = temp_dir / 'infos-{0}.json'.format(code)
        inf, lp = predict(infos, use_default=True, return_latest_prediction=True)
        with open(infos_file, 'w') as f:
            json.dump(inf, f)
        if df is None:
            df = lp
        else:
            df = pd.concat([df, lp])
        time.sleep(1.0)

    df.to_csv(temp_dir / 'all_{0}.csv'.format(moment.now().format('YYYYMMDD_HHmmss')))
