from apps.stocking.actions.fetch import fetch_data
from apps.stocking import logger

from apps.stocking.actions.make import make_x, make_y, cate_y, make_xy

import pandas as pd

err_code = 'sh000789'
corr_code = 'sz000789'

ds = pd.DataFrame(
    [[1, 2, 3, 4, 7, 7.2], [4, 3, 2, 6, 8, 8.1], [3, 5, 7, 3, 7.7, 7.8],
     [0, 3, 0, 3, 7.2, 7.43], [1, 2, 7, 3, 7, 7.5], [8, 4, 7, 5, 7.8, 7.93],
     [4, 8, 5, 7, 4, 4.02], [4, 9, 8, 5, 7, 7.44], [3, 9, 7, 5, 8, 8.33],
     [4, 9, 5, 7, 9, 9.08], [3, 9, 4, 7, 7, 7.04]],
    columns=['c1', 'c2', 'c3', 'c4', 'close', 'high'])


def test_fetch_data():
    df = fetch_data(err_code)
    assert df is None
    df = fetch_data(corr_code)
    logger.info(df)
    assert len(df) > 2000


def test_make_x():
    x = make_x(range(1, 5), ds, ['c2', 'c4'], past_days=2)
    assert x.iloc[0, 3] == x.iloc[1, 1]
    assert len(x) == 4
    assert len(x.columns) == 4
    assert x.columns[0] == 'p1_c2'


def test_cate_y():
    assert cate_y(0.01) == 'y_in_0_3'
    assert cate_y(0.12) == 'y_gt_10'
    assert cate_y(0.05) == 'y_in_3_10'
    assert cate_y(-0.6) == 'y_lt__5'
    assert cate_y(-0.03) == 'y_in__5_0'


def test_make_y():
    y = make_y(range(3, 9), ds, cols=['high', 'close'], future_days=2)
    assert y.loc[5]['range'] == 'y_in__5_0'


def test_make_xy():
    xy = make_xy(ds, ['c2', 'c4'], past_days=2, future_days=2)
    assert xy.loc[2]['range'] == 'y_in_3_10'
    assert xy.loc[5]['p2_c2'] == 8
