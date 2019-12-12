import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from apps.stocking import logger
from pathlib import Path


def train(x,
          y,
          clf,
          x_latest=None,
          param_grid=None,
          pre_process=lambda *p: p,
          test_size=0.1,
          random_state=33,
          *args,
          **kwargs):

    logger.debug('Preprocessing data with {0}...'.format(pre_process.__name__))
    clf, x, y, x_latest = pre_process(clf, x, y, x_latest)

    if param_grid is not None:
        clf = GridSearchCV(clf, param_grid)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=test_size, random_state=random_state)
    logger.debug(
        'Train and Test split, train size: x{0} y{1}, test size: x{2} y{3}.'.format(
            np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)))
    logger.debug('Train data fitting...')
    clf.fit(x_train, y_train)
    logger.debug('Computing score...')
    score = clf.score(x_test, y_test)
    best_score = score
    best_estimator = clf
    if param_grid is not None:
        best_score = clf.best_score_
        best_estimator = clf.best_estimator_
    logger.debug('Test score: {0:.2f}%'.format(score * 100))
    if param_grid is not None:
        logger.debug('Best score: {0:.2f}%'.format(best_score * 100))
        logger.debug('Best params: {0}'.format(str(clf.best_params_)))
    y_valid = best_estimator.predict(x)
    yvy = y_valid <= y
    smaller_rate = np.mean(yvy)
    y_pred = best_estimator.predict(x_latest)

    logger.debug('Valid smaller rate: {0:.2f}%({1}/{2})'.format(
        smaller_rate * 100, np.count_nonzero(yvy), len(yvy)))
    return best_estimator, best_score, smaller_rate, y_valid, y_pred
