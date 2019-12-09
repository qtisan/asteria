from pathlib import Path
import numpy as np

CACHED_ROOT = Path(__file__).parent / 'data/original'
TRAINED_ROOT = Path(__file__).parent / 'data/trained'

y_dict_name = {0: '下跌', 1: '涨0-3个点', 2: '涨3-10个点', 3: '涨超10个点'}


def categorify_y(y_value):
    c = 0
    if y_value > 0.1:
        c = 3
    elif y_value > 0.03:
        c = 2
    elif y_value > 0:
        c = 1
    return c


x_dict_name = {
    'netamount': '净流入',
    'ratioamount': '净流入率',
    'r0_net': '主力净流入',
    'r0_ratio': '主力净流入率',
    'r0x_ratio': '主力罗盘',
    'cnt_r0x_ratio': '主力朝向',
    'cate_ra': 'RA',
    'cate_na': 'NA',
    'close': '收盘价',
    'high': '最高值',
    'low': '最低值',
    'open': '开盘价',
    'yest': '昨收价',
    'turnover': '换手率',
    'volumn': '成交量',
    'amount': '成交额',
    'total': '总市值',
    'circul': '流通值'
}
x_dict_name_extra = {'__{0}': '{0}次方', '_chg_{0}': '前{0}天变化幅度'}
