def y_categorifier(y_dict):
    '''
    y_dict sample:
    [{
        "name": "下跌",
        "threshold": -9999
    }, {
        "name": "上涨",
        "threshold": 0
    }]
    '''
    def classify_y(y_value):
        c = 0
        for i, yd in enumerate(y_dict):
            if y_value > yd['threshold']:
                c = i
        return c

    return classify_y
