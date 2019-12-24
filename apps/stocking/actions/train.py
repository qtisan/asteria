import numpy as np
import moment
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import is_classifier, is_regressor

from apps.stocking.metainfo import logger


def fit(x,
        ys,
        estimator,
        param_grid=None,
        x_latest=None,
        ys_latest=None,
        pre_process=lambda *p: p,
        pre_process_args={},
        test_size=0.1,
        random_state=29,
        regress_col_index=2,
        classify_col_index=0,
        t_t_split=train_test_split,
        *args,
        **kwargs):

    col_index = classify_col_index if is_classifier(estimator) else (
        regress_col_index if is_regressor(estimator) else None)
    if col_index is None:
        raise 'Estimator <{0}> error, not Classifier or Regressor!'.format(
            str(estimator))
    y, y_latest = ys[:, col_index], ys_latest[:, col_index]

    if callable(pre_process):
        logger.debug('Preprocessing data with {0}...'.format(str(pre_process)))
        estimator, x, y, x_latest, y_latest = pre_process(estimator, x, y, x_latest,
                                                          y_latest,
                                                          **pre_process_args)

    if param_grid is not None:
        estimator = GridSearchCV(estimator, param_grid)
    x_train, x_test, y_train, y_test = \
        t_t_split(x, y, test_size=test_size, random_state=random_state)
    logger.debug(
        'Train and Test split, train size: x{0} y{1}, test size: x{2} y{3}.'.format(
            np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)))
    logger.debug('Train with \n {0} \n'.format(str(estimator)))
    logger.debug('Fitting...')

    timestart = moment.now().epoch()
    estimator.fit(x_train, y_train)
    logger.debug('-- Fit time cost: {0} sec.'.format(moment.now().epoch() -
                                                     timestart))
    logger.debug('Computing score...')
    score = estimator.score(x_test, y_test)
    best_score = score
    best_estimator = estimator
    if param_grid is not None:
        best_score = estimator.best_score_
        best_estimator = estimator.best_estimator_

    score_form = 'Best score: {0:.2f}%'.format(
        best_score *
        100) if col_index == classify_col_index else 'Best score: {0:.2f}'.format(
            best_score)
    if param_grid is not None:
        logger.debug(score_form)
        logger.debug('Best params: {0}'.format(str(estimator.best_params_)))
    else:
        logger.debug(score_form)

    # Validation
    y_test_valid = best_estimator.predict(x_test)
    yvy = y_test_valid <= y_test
    smaller_rate = np.mean(yvy)
    logger.debug('Test smaller rate: {0:.2f}%({1}/{2})'.format(
        smaller_rate * 100, np.count_nonzero(yvy), len(yvy)))

    y_pred_all = best_estimator.predict(np.concatenate((x_latest, x)))
    y_pred = y_pred_all[len(y):]
    y_all = y_pred_all[:len(y)]
    yey = y_all == y if col_index == classify_col_index else np.abs(
        (y_all - y) / y) <= 0.05
    yay = y_all <= y
    equal_rate_all = np.mean(yey)
    smaller_rate_all = np.mean(yay)
    logger.debug(
        'All equal rate: {3:.2f}%({4}/{5}), smaller rate: {0:.2f}%({1}/{2})'.format(
            smaller_rate_all * 100, np.count_nonzero(yay), len(yay),
            equal_rate_all * 100, np.count_nonzero(yey), len(yey)))

    return best_estimator, best_score, smaller_rate, smaller_rate_all, equal_rate_all, y_all, y_pred, y_latest
