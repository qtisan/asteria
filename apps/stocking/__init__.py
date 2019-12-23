# Stocking.  Dec 4, 2019

import os
import pandas as pd

from apps.stocking.metainfo import *
from apps.stocking.run import *


def read_infos(code):
    infos = None
    with open(datadir / 'trained/infos-{0}/latest/infos.json'.format(code)) as f:
        infos = json.load(f)
    return infos


def read_original_data(code):
    od = pd.read_csv(datadir / 'original/{0}.csv'.format(code), index_col=0)
    return od.to_dict('index')


def read_xyy(code):
    od = pd.read_csv(datadir / 'trained/infos-{0}/latest/xy.csv'.format(code),
                     index_col=0)
    return od.to_dict('index')
