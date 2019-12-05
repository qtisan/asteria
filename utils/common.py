import requests
import moment
import logging
from pathlib import Path


def getLogger(app_name, level=logging.DEBUG):
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    logfile = logging.FileHandler(
        Path(__file__,
             '../../logs/{0}'.format(app_name + '-log.txt')).resolve())
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
    lt = moment.date(last_time)
    now = moment.now()
    return now.subtract(unit, span) > lt


default_logger = getLogger('default')


def debug(msg):
    default_logger.debug(msg)
