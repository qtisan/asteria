from flask import Blueprint, g, render_template, request
from web.routes.auth import login_required

from apps.stocking import get_settings, read_infos, read_original_data, read_xyy

bp = Blueprint('stocking', __name__, url_prefix='/stocking')


@bp.route('/')
@login_required
def index():
    settings = get_settings()
    stocks = list(settings['stocks'])
    stocks.insert(0, 'all')
    return render_template('stocking/index.j2', title='Stocking', stocks=stocks)


@bp.route('/<string:code>')
@login_required
def infos(code: str):
    settings = get_settings()
    stocks = list(settings['stocks'])
    stocks.insert(0, 'all')
    infos = read_infos(code)

    od = read_original_data(code)
    xy = read_xyy(code)

    dates = []
    values = []
    volumes = []
    ranges = []
    predictions = []
    y_cs = []
    y_vs = []
    dss = sorted(od.items(), key=lambda o: o[1]['opendate'])
    len_dss = len(dss) - 1
    for k, v in dss:
        values.append([v['open'], v['close'], v['low'], v['high'], v['yest']])
        volumes.append(
            [len_dss - k, v['volume'], 1 if v['open'] < v['close'] else -1])
        dates.append(v['opendate'])
        ranges.append(xy[k]['range'])
        predictions.append(xy[k]['prediction'])
        y_cs.append(xy[k]['y_c'])
        y_vs.append(xy[k]['y_v'])
    last_index = len(y_vs) - 1
    y_vs[last_index] = values[last_index][1]
    y_cs[last_index] = '0%'

    return render_template('stocking/index.j2',
                           title='Stocking',
                           stocks=stocks,
                           code=code,
                           infos=infos,
                           kdata={
                               'dates': dates,
                               'values': values,
                               'volumes': volumes,
                               'ranges': ranges,
                               'predictions': predictions,
                               'y_vs': y_vs,
                               'y_cs': y_cs
                           },
                           y_dict=settings['y_dict'])
