from apps.stocking.meta import x_dict_name
from utils.common import move_down

import pandas as pd
import numpy as np

x_keys_base = list(x_dict_name.keys())


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


def extends_ds(ds: pd.DataFrame,
               feature_pows=[2, 3, 4],
               feature_chgs=[1, 2, 3, 5, 10],
               zeros_copy_days=10,
               *args,
               **kwargs):
    ds_base = ds.loc[:, x_keys_base].to_numpy()
    ds_base = fill_zeros(ds_base, zeros_copy_days, extra_value=1)
    return pd.concat([
        ds.loc[:, x_keys_base],
        features_pow(ds_base, x_keys_base, pows=feature_pows),
        features_changed(ds_base, x_keys_base, chgs=feature_chgs)
    ],
                     axis=1)
