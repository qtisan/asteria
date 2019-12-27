import os
import moment
import numpy as np


def is_expire(last_time, span=1, unit='day'):
    lt = None
    if isinstance(last_time, str):
        lt = moment.date(last_time)
    if isinstance(last_time, moment.Moment):
        lt = last_time
    if lt is None:
        raise 'last_time must be str or moment.Moment'
    now = moment.now()
    return now.subtract(unit, span) > lt


def get_file_modify_time(filepath):
    return moment.unix(int(os.stat(filepath).st_mtime * 1000))


def time_span(d1: moment.Moment, d2: moment.Moment, el='day'):
    d1e, d2e = d1.epoch(), d2.epoch()
    if el == 'second':
        return d2e - d1e
    if el == 'minute':
        return int(d2e / 60) - int(d1e / 60)
    if el == 'hour':
        return int(d2e / 3600) - int(d1e / 3600)
    if el == 'day':
        return int(d2e / 3600 / 24) - int(d1e / 3600 / 24)
    else:
        y1, y2, m1, m2 = d1.year, d2.year, d1.month, d2.month
        if el == 'month':
            return m2 - m1 + 12 * (y2 - y1)
        if el == 'year':
            return y2 - y1
    raise 'el error!'


def same_day(d1: moment.Moment, d2: moment.Moment):
    return time_span(d1, d2) == 0


def is_yesterday(today: moment.Moment, target_day: moment.Moment):
    return time_span(today, target_day) == -1


def move_up(arr: np.ndarray, n: int):
    return np.concatenate((arr[n:len(arr)], np.repeat([arr[len(arr) - 1]], n,
                                                      axis=0)))


def move_down(arr: np.ndarray, n: int):
    return np.concatenate((np.repeat([arr[0]], n, axis=0), arr[0:len(arr) - n]))


def hooked(data, hooks=None):
    if hooks is not None:
        if isinstance(hooks, (list, tuple)):
            for hook in [h for h in hooks if callable(h)]:
                data = hook(data)
        elif callable(hooks):
            data = hooks(data)
        else:
            pass
    return data
