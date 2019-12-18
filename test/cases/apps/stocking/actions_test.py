import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from apps.stocking.actions.fetch import fetch_data
from apps.stocking import logger

from apps.stocking.actions.make import make_x, make_y, make_xy
from apps.stocking.classifiers import y_categorifier

import pandas as pd

err_code = 'sh000789'
corr_code = 'sz000789'

ds = pd.read_csv(Path(__file__).parent.parent.parent.parent / 'fixtures/sz002415.csv',
                 index_col=0)


def test_fetch_data():
    df = fetch_data(err_code)
    assert df is None
    df = fetch_data(corr_code)
    logger.info(df)
    assert len(df) > 2000


def test_make_x():
    x = make_x(range(1, 5), ds, ['trade', 'turnover'], past_days=2)
    assert x.iloc[0, 3] == x.iloc[1, 1]
    assert len(x) == 4
    assert len(x.columns) == 4
    assert x.columns[0] == 'p1_trade'


cate_y = y_categorifier([{
    "name": "下跌",
    "threshold": -99999
}, {
    "name": "涨0-3%",
    "threshold": 0
}, {
    "name": "涨3-6%",
    "threshold": 0.03
}, {
    "name": "涨6-10%",
    "threshold": 0.06
}, {
    "name": "涨超10%",
    "threshold": 0.1
}])


def test_cate_y():
    assert cate_y(0.01) == 1
    assert cate_y(0.12) == 4
    assert cate_y(0.08) == 3
    assert cate_y(0.05) == 2
    assert cate_y(-0.6) == 0
    assert cate_y(-0.03) == 0


def test_make_y():
    y, yvc = make_y(range(3, 100),
                    ds,
                    cols=['high', 'close'],
                    future_days=10,
                    categorify=cate_y)
    assert y.iloc[0] == cate_y(float(yvc.iloc[0]['y_c'].replace('%', '')) / 100)
    assert y.iloc[49] == cate_y(float(yvc.iloc[49]['y_c'].replace('%', '')) / 100)
    assert y.iloc[65] == cate_y(float(yvc.iloc[65]['y_c'].replace('%', '')) / 100)
    assert y.iloc[73] == cate_y(float(yvc.iloc[73]['y_c'].replace('%', '')) / 100)
