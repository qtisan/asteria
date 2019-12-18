#!/bin/sh


cd $(pwd)/`dirname $0`
cd ..

if [ -z $VIRTUAL_ENV ] ; then
    . venv/bin/activate
fi

venv/bin/python -m scripts.stocking $*
