from utils.common import is_expire
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
