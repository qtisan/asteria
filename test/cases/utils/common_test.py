import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.functional import is_expire, time_span
import moment

d1 = moment.now()
d2 = moment.now().subtract('day', 2)
d3 = moment.now().subtract('day', 0.4)
d4 = moment.now().subtract('hour', 48)


def test_is_expire():
    assert not is_expire(d1)
    assert is_expire(d2)
    assert not is_expire(d3, span=0.5)
    assert is_expire(d4, span=1.6)
    assert not is_expire(d3, span=10, unit='hour')


def test_time_span():
    d1 = moment.date('2019-12-12 12:12:12')
    d2 = moment.date('2020-05-12 01:12:00')
    d3 = moment.date('2020-05-12 02:05:00')
    assert time_span(d1, d2, 'year') == 1
    assert time_span(d1, d2, 'month') == 5
    assert time_span(d1, d2, 'day') == 152
    assert time_span(d3, d2, 'hour') == -1
    assert time_span(d3, d2, 'minute') == -53
    assert time_span(d3, d2, 'second') == -53 * 60
