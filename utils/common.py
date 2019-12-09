import os
import requests
import moment
import logging
import numpy as np
from pathlib import Path


def getLogger(app_name, level=logging.DEBUG):
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    logfile = logging.FileHandler(
        Path(__file__, '../../logs/{0}'.format(app_name + '-log.txt')).resolve())
    logfile.setLevel(level)
    logfile.setFormatter(
        logging.Formatter(
            '[%(name)s][%(levelname)s]%(asctime)s: %(message)s > %(pathname)s(func: [%(funcName)s] at line %(lineno)d)'
        ))
    logger.addHandler(logfile)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter('[%(levelname)s] - %(message)s - %(asctime)s'))
    logger.addHandler(console)

    return logger


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


def move_up(arr, n):
    return np.concatenate((arr[n:len(arr)], np.repeat([arr[len(arr) - 1]], n,
                                                      axis=0)))


def move_down(arr, n):
    return np.concatenate((np.repeat([arr[0]], n, axis=0), arr[0:len(arr) - n]))


default_logger = getLogger('default')


def debug(msg):
    default_logger.debug(msg)
