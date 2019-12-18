import logging
from pathlib import Path

root_dir = Path(__file__).parent.parent


def get_logger(app_name, level=logging.DEBUG):
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    logfile = logging.FileHandler(root_dir / 'logs/{0}'.format(app_name + '-log.txt'))
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


default_logger = get_logger('default')


def debug(msg):
    default_logger.debug(msg)


def path_finder(func='apps'):
    proot = root_dir / func

    def p_path(app_name=None, sub_path=None):
        if app_name is None:
            return proot
        if isinstance(app_name, str):
            if sub_path is not None:
                return proot / app_name / sub_path
            return proot / app_name

        raise 'Arguments error!'

    return p_path


data_path = path_finder('data')
app_path = path_finder('apps')


def add_sys_path():
    import sys
    srd = str(root_dir)
    if srd not in sys.path:
        sys.path.append(srd)
