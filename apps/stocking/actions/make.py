from pandas import DataFrame
import requests
from io import BytesIO
import time
import datetime
import moment
import pandas as pd
import numpy as np

from apps.stocking import logger


class Flat_x(object):
    def __init__(self, featured_ds, past_days, col_names):
        self.featured_ds = featured_ds
        self.past_days = past_days
        self.col_names = col_names

    def __call__(self, row):
        curr_i = row.name
        return pd.Series(
            self.featured_ds.iloc[curr_i:curr_i +
                                  self.past_days].to_numpy().flatten(),
            index=self.col_names)


def make_x(index, ds: pd.DataFrame, features, past_days=20):
    featured_ds = ds.loc[:, features]
    col_names = [
        'p' + str(d + 1) + '_' + c for d in range(past_days) for c in features
    ]

    flat = Flat_x(featured_ds, past_days, col_names)
    targ = featured_ds.iloc[index]
    x = None
    if isinstance(targ, pd.Series):
        x = flat(targ)
    else:
        x = targ.apply(flat, axis=1)
    return x


def cate_y(y_value):
    c = 'y_lt__5'
    if y_value > 0.1:
        c = 'y_gt_10'
    elif y_value > 0.03:
        c = 'y_in_3_10'
    elif y_value > 0:
        c = 'y_in_0_3'
    elif y_value > -0.05:
        c = 'y_in__5_0'
    return c


def make_y(index,
           ds: pd.DataFrame,
           cols,
           future_days=10,
           categorify=cate_y,
           colname='range'):
    high_ds = ds.loc[:, cols]

    def ran(row):
        curr_i = row.name
        h_max = high_ds.iloc[curr_i - future_days:curr_i].max().max()
        curr_close = row['close']
        return pd.Series([categorify((h_max - curr_close) / curr_close)],
                         index=[colname])

    y = high_ds.iloc[index].apply(ran, axis=1)
    return y


def make_xy(ds: pd.DataFrame,
            features,
            past_days=20,
            future_days=10,
            y_col_names=['high', 'close'],
            y_cate_name='range'):
    logger.debug(
        '========================================================================'
    )
    logger.debug('start making x and y...')
    logger.debug('The origin data:')
    logger.debug(ds)

    len_ds = len(ds)
    end = len_ds - past_days
    begin = future_days
    assert begin < end, 'begin:{0}, end:{1}'.format(begin, end)
    logger.debug('Indexing from {0} to {1} in featured_ds...'.format(
        begin, end))

    x = make_x(range(begin, end), ds, features=features, past_days=past_days)
    y = make_y(range(begin, end),
               ds,
               future_days=future_days,
               colname=y_cate_name,
               cols=y_col_names)

    logger.debug('x size: {0}'.format(x.shape))
    logger.debug('y size: {0}'.format(y.shape))
    logger.debug('composing...')
    return pd.concat([x, y], axis=1)
