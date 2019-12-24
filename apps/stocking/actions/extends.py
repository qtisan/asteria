from utils.functional import move_down

import pandas as pd
import numpy as np


def features_pow(darr: np.ndarray, keys: list, pows: list = [2, 3]):
    _pow = lambda n: pd.DataFrame(darr**n,
                                  columns=np.char.add(keys, '__{0}'.format(str(n))))

    return pd.concat([_pow(n) for n in pows], axis=1)


def features_changed(darr: np.ndarray, keys: list, chgs: list = [1, 2, 3, 5]):
    def _changed(n):
        _for_cmp = np.concatenate(
            (darr[n:len(darr)], np.repeat([darr[len(darr) - 1]], n, axis=0)))

        # FIXME: divide by zero
        return pd.DataFrame((darr - _for_cmp) / _for_cmp,
                            columns=np.char.add(keys, '_chg_{0}'.format(str(n))))

    return pd.concat([_changed(n) for n in chgs], axis=1)


def fill_zeros(ds: np.ndarray, n: int, extra_value=1):
    _for_cpy = move_down(ds, n)
    _ds = ds.copy()
    np.copyto(_ds, _for_cpy, where=_ds == 0)
    np.copyto(_ds, extra_value, where=_ds == 0)
    return _ds


def calc_ema(closes: list, N: int):
    emas = []
    for i in range(len(closes)):
        if i == 0:
            emas.insert(i, closes[i])
        else:
            emas.insert(i, (2 * closes[i] + (N - 1) * emas[i - 1]) / (N + 1))
    return emas


def calc_macd(df: pd.DataFrame, n_short=12, n_long=26, M=9, close_name='close'):
    sorted_df = df.sort_index(ascending=False)
    closes = sorted_df.loc[:, close_name].to_list()
    sh = calc_ema(closes, n_short)
    lg = calc_ema(closes, n_long)
    diff = np.subtract(sh, lg)
    dea = []
    for i in range(len(diff)):
        if i == 0:
            dea.insert(i, diff[i])
        else:
            dea.insert(i, (2 * diff[i] + (M - 1) * diff[i - 1]) / (M + 1))
    macd = 2 * np.subtract(diff, dea)
    comp = pd.concat((pd.Series(diff), pd.Series(dea), pd.Series(macd)), axis=1)
    comp.set_axis(['diff', 'dea', 'macd'], axis=1, inplace=True)
    comp.set_axis(sorted_df.index, axis=0, inplace=True)
    return comp.sort_index(ascending=True)


def extends_ds(ds: pd.DataFrame,
               feature_pows=[2, 3, 4],
               feature_chgs=[1, 2, 3, 5, 10],
               zeros_copy_days=10,
               x_dict={},
               *args,
               **kwargs):
    x_keys_base = list(x_dict.keys())
    macd = calc_macd(ds)
    ds_with_macd = pd.concat([ds.loc[:, x_keys_base], macd], axis=1)
    x_keys_extra = x_keys_base + list(macd.keys().values)
    ds_base = ds_with_macd.to_numpy()
    ds_base = fill_zeros(ds_base, zeros_copy_days, extra_value=1)
    return pd.concat([
        ds_with_macd,
        features_pow(ds_base, x_keys_extra, pows=feature_pows),
        features_changed(ds_base, x_keys_extra, chgs=feature_chgs),
    ],
                     axis=1)
