from urllib.parse import urlencode
import execjs
import moment
import pandas as pd
import requests
from apps.stocking import logger
from utils.common import get_file_modify_time, is_expire
from pathlib import Path
import os
import io

from apps.stocking.meta import CACHED_ROOT

# sina
# =========================================
#                收盘价      涨跌幅     换手率      净流入/万      净流入率   主力净流入/万  主力净流入率  主力罗盘
#      opendate  trade  changeratio  turnover    netamount  ratioamount       r0_net  r0_ratio  r0x_ratio  cnt_r0x_ratio   cate_ra       cate_na
# 0  2019-11-28  11.94    -0.011589   59.0871   -663428.24    -0.006620   -480062.28 -0.004790  -157.5440             -1 -0.013478 -2.487583e+08
# 1  2019-11-27  12.03     0.001665  101.8510  17036254.05     0.097309  12704493.07  0.072567    88.6854              1  0.000918  1.550034e+07

# ne
# =========================================
#                   收     高     低      开    昨收     换手率    成交量        成交额          总市值        流通市值
#          date  close   high    low   open   yest  turnover    volumn        amount         total        circul
# 0  2019-11-28  11.93  12.19  11.85  12.00  12.08    0.5385   8470873  1.015543e+08  1.876724e+10  1.876724e+10
# 1  2019-11-27  12.08  12.33  11.96  12.28  12.01    0.9251  14553516  1.768222e+08  1.900320e+10  1.900320e+10


def path_of(code: str):
    return CACHED_ROOT / '{0}.csv'.format(code)


def read_cached_data(code: str):
    data = None
    filepath = path_of(code)
    if not filepath.exists():
        return data
    mtime = get_file_modify_time(filepath)
    logger.debug('Cached data last updated: {0}'.format(
        mtime.format('YYYY-MM-DD HH:mm:ss')))
    if not is_expire(mtime, span=0.5):
        logger.debug('Cached data not expired.({0})'.format(filepath))
        data = pd.read_csv(filepath, index_col=0)
    return data


def make_sina_url(code: str, page=1, num=4000):
    return '{0}?{1}'.format(
        'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_zjlrqs',
        urlencode({
            'page': page,
            'num': num,
            'sort': 'opendate',
            'asc': 0,
            'daima': code
        }))


def make_ne_url(
    code: str,
    start='20000101',
    end=moment.now().format('YYYYMMDD'),
    fields='TCLOSE;HIGH;LOW;TOPEN;LCLOSE;VOTURNOVER;VATURNOVER;TCAP;MCAP'):
    return '{0}?{1}'.format(
        'http://quotes.money.163.com/service/chddata.html',
        urlencode({
            'code': code,
            'start': start,
            'end': end,
            'fields': fields
        }))


def make_url(code: str):
    sina_url = make_sina_url(code)
    m = '0' if code.startswith('sh') else '1'
    ne_url = make_ne_url(m + code[2:])
    return (sina_url, ne_url)


def fetch_data(code: str):
    '''
    str: sh600765, sz000232, etc...
    '''
    logger.debug('Cached data expired or not exists, fetching from websites...')
    data_sina: pd.DataFrame = None
    data_ne: pd.DataFrame = None
    sina_url, ne_url = make_url(code)
    try:
        res = requests.get(sina_url)
        json_str = execjs.eval('JSON.stringify({0})'.format(res.text))
        data_sina = pd.read_json(json_str)
        data_sina.set_index('opendate')

        res = requests.get(ne_url)
        names = [
            'opendate', 'code', 'name', 'close', 'high', 'low', 'open', 'yest',
            'volumn', 'amount', 'total', 'circul'
        ]
        data_ne = pd.read_csv(io.BytesIO(res.text.encode('utf-8')),
                              header=0,
                              names=names,
                              usecols=[0, *range(3, 12)],
                              index_col='opendate')
    except Exception as ex:
        logger.error(ex)
        return None

    ds = date_indexed_compose(data_sina, data_ne)
    filepath = path_of(code)
    ds.to_csv(filepath)

    return ds


def date_indexed_compose(sina: pd.DataFrame, ne: pd.DataFrame):
    ds = pd.merge(sina, ne, on='opendate')
    ds = ds.fillna(method='bfill')
    ds = ds.fillna(method='ffill')
    ds = ds.replace(0., method='bfill')
    ds = ds.replace(0., method='ffill')

    return ds


def fetch(code: str, *args, **kwargs):
    data = read_cached_data(code)
    if data is not None:
        return data
    return fetch_data(code)
