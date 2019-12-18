from flask import Blueprint, g, render_template, request
from web.routes.auth import login_required

bp = Blueprint('about', __name__, url_prefix='/about')


@bp.route('/')
@login_required
def index():
    return render_template('about/index.j2', title='About')
