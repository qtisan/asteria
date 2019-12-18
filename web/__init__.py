# web module
import sys
from pathlib import Path

from .routes import bps

import os
from flask import Flask, request
import moment

static_dir = str(Path(__file__).parent / 'static')
templates_dir = str(Path(__file__).parent / 'templates')


def create_app(test_config=None):

    print('Instance [{0}] mounting...'.format(__name__))
    app = Flask(__name__,
                instance_relative_config=True,
                static_url_path=static_dir,
                template_folder=templates_dir)
    print('Instance path: {0}'.format(app.instance_path))
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'asteria.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    for bp in bps:
        print('[-] Register Blueprint [{0}].'.format(bp.name))
        app.register_blueprint(bp)

    from . import db
    db.init_app(app)

    @app.context_processor
    def inject_utilities():  # pylint: disable=unused-variable
        def timestamp():
            return moment.now().epoch(milliseconds=True)

        def cates():
            ps = request.path.split('/')
            return ps[1] if ps[0] == '' else ps[0]

        return dict(timestamp=timestamp, cates=cates)

    return app
