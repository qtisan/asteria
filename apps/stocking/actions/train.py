import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

from apps.stocking.metainfo import logger


def train(x,
          y,
          clf,
          x_latest=None,
          y_latest=None,
          param_grid=None,
          pre_process=lambda *p: p,
          test_size=0.1,
          random_state=29,
          t_t_split=train_test_split,
          *args,
          **kwargs):

    logger.debug('Preprocessing data with {0}...'.format(pre_process.__name__))
    clf, x, y, x_latest, y_latest = pre_process(clf, x, y, x_latest, y_latest)

    if param_grid is not None:
        clf = GridSearchCV(clf, param_grid)
    x_train, x_test, y_train, y_test = \
        t_t_split(x, y, test_size=test_size, random_state=random_state)
    logger.debug(
        'Train and Test split, train size: x{0} y{1}, test size: x{2} y{3}.'.format(
            np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)))
    logger.debug('Train with \n {0} \n'.format(str(clf)))
    logger.debug('Fitting...')

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

    # Validation
    y_valid = best_estimator.predict(x_test)
    yvy = y_valid <= y_test
    smaller_rate = np.mean(yvy)
    logger.debug('Valid smaller rate: {0:.2f}%({1}/{2})'.format(
        smaller_rate * 100, np.count_nonzero(yvy), len(yvy)))

    y_all = best_estimator.predict(x)
    yay = y_all <= y
    smaller_rate_all = np.mean(yay)
    logger.debug('All smaller rate: {0:.2f}%({1}/{2})'.format(
        smaller_rate_all * 100, np.count_nonzero(yay), len(yay)))

    y_pred = best_estimator.predict(x_latest)

    return best_estimator, best_score, smaller_rate, smaller_rate_all, y_all, y_pred
