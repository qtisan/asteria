import os
from utils.common import get_logger
from utils.common import data_path, app_path

app_name = 'stocking'

datadir = data_path(app_name)
appdir = app_path(app_name)

CACHED_ROOT = datadir / 'original'
TRAINED_ROOT = datadir / 'trained'

if not datadir.exists():
    os.mkdir(datadir)

if not CACHED_ROOT.exists():
    os.mkdir(CACHED_ROOT)

if not TRAINED_ROOT.exists():
    os.mkdir(TRAINED_ROOT)

logger = get_logger(app_name)
