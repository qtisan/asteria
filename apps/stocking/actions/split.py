def tt_split(x, y, test_size=0.2, *args, **kwargs):
    length = len(y)
    train_length = length - int(test_size * length)
    start = length - train_length
    x_train = x[start:]
    y_train = y[start:]
    x_test = x[:start]
    y_test = y[:start]
    return x_train, x_test, y_train, y_test