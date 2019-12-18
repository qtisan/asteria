#!/bin/sh


cd $(pwd)/`dirname $0`
cd ..


if [ -z $VIRTUAL_ENV ] ; then
    . venv/bin/activate
fi


export FLASK_APP="web"

# python scripts/host.py
if [ 'x'$1 = "xdev" ]; then
    export FLASK_ENV="development"
else
    export FLASK_ENV="production"
fi
echo "Running server in ["${FLASK_ENV}"] mode."

flask run -h 0.0.0.0 -p 8000
