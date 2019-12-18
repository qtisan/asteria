from utils.common import add_sys_path
add_sys_path()

import sys

from apps.stocking import logger, run_one, run_batch

if __name__ == "__main__":

    args = sys.argv
    arglen = len(args)

    if arglen == 1:
        logger.error('The first parameter needed, it should be `all` or <stock_code>')
        exit()

    if arglen == 2:
        if args[1] == 'all':
            logger.info('- Stock Predicting ALL CODES with default classifier.')
            run_batch()
        else:
            logger.info('- Stock Predicting ({0}) with default classifier.'.format(
                args[1]))
            run_one(args[1])
    elif arglen == 3:
        if isinstance(args[2], str):
            logger.info('- Stock Predicting ({0}) with [{1}].'.format(
                args[1], args[2]))
            run_one(args[1], classifier=args[2])
        else:
            raise 'Error argument [classifier]!'

    else:
        logger.error('Arguments num not correct.')
