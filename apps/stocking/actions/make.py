from pandas import DataFrame
import requests
from io import BytesIO
import time
import datetime
import moment
import pandas as pd
import numpy as np

from apps.stocking.metainfo import logger


class X_Editor(object):
    def __init__(self, ds, features, past_days):
        self.featured_ds = ds.loc[:, features]
        self.past_days = past_days
        self.col_names = [
            'p' + str(d + 1) + '_' + c for d in range(past_days) for c in features
        ]

    def __call__(self, row):
        curr_i = row.name
        return pd.Series(self.featured_ds.iloc[curr_i:curr_i +
                                               self.past_days].to_numpy().flatten(),
                         index=self.col_names)


def make_x(index,
           ds: pd.DataFrame,
           features,
           past_days=20,
           Editor=X_Editor,
           *args,
           **kwargs):
    edit = Editor(ds, features, past_days)

    featured_ds = edit.featured_ds
    targ = featured_ds.iloc[index]
    x = None
    if isinstance(targ, pd.Series):
        x = edit(targ)
    else:
        x = targ.apply(edit, axis=1)
    return x


def cate_y(y_value):
    c = 0
    if y_value > 0:
        c = 1
    return c


def make_y(index,
           ds: pd.DataFrame,
           cols,
           future_days=10,
           categorify=cate_y,
           colname='range',
           return_value_change=True,
           *args,
           **kwargs):
    high_ds = ds.loc[:, cols]

    def ran(row):
        curr_i = row.name
        curr_b = 0 if curr_i - future_days < 0 else curr_i - future_days
        h_max = high_ds.iloc[curr_b:curr_i].max().max()
        curr_close = row['close']
        yc = (h_max - curr_close) / curr_close if curr_close != 0 else -0.01
        return pd.Series([categorify(yc), h_max, '{0:.2f}%'.format(yc * 100)],
                         index=[colname, 'y_v', 'y_c'])

    ys = high_ds.iloc[index].apply(ran, axis=1)
    val = ys.iloc[:][colname]
    if return_value_change:
        val = (val, ys.iloc[:][['y_v', 'y_c']])
    return val


def make_xy(ds: pd.DataFrame,
            features,
            past_days=20,
            future_days=10,
            y_col_names=['high', 'close'],
            categorify=cate_y,
            y_cate_name='range',
            return_Xy=False,
            to_numpy=True,
            *args,
            **kwargs):
    logger.debug(
        '========================================================================')
    logger.debug('start making x and y...')
    logger.debug('The origin data:')
    logger.debug(ds)

    len_ds = len(ds)
    end = len_ds - past_days
    begin = future_days
    assert begin < end, 'begin:{0}, end:{1}'.format(begin, end)
    logger.debug('Indexing from {0} to {1} in featured_ds...'.format(begin, end))

    x = make_x(range(begin, end), ds, features=features, past_days=past_days)
    y, y_vc = make_y(range(begin, end),
                     ds,
                     cols=y_col_names,
                     future_days=future_days,
                     colname=y_cate_name,
                     categorify=categorify)
    x_latest = make_x(range(begin), ds, features=features, past_days=past_days)
    y_latest, y_vc_latest = make_y(range(begin),
                                   ds,
                                   cols=y_col_names,
                                   future_days=future_days,
                                   colname=y_cate_name,
                                   categorify=categorify)

    logger.debug('x size: {0}'.format(x.shape))
    logger.debug('y size: {0}'.format(y.shape))
    logger.debug('x_latest size: {0}'.format(x_latest.shape))
    logger.debug('y_latest size: {0}'.format(y_latest.shape))
    logger.debug('composing...')
    if return_Xy:
        if to_numpy:
            return x.to_numpy(), y.to_numpy(), x_latest.to_numpy(), y_latest.to_numpy(
            ), y_vc.to_numpy(), y_vc_latest.to_numpy()
        return x, y, x_latest, y_latest, y_vc, y_vc_latest
    if to_numpy:
        return pd.concat([x, y], axis=1).to_numpy(), x_latest.to_numpy(
        ), y_latest.to_numpy(), y_vc.to_numpy(), y_vc_latest.to_numpy()
    return pd.concat([x, y], axis=1), x_latest, y_latest, y_vc, y_vc_latest
