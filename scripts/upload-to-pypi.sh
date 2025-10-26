#!/bin/bash

rm -rf dist
python setup.py sdist bdist_wheel
python setup.py sdist
pip install twine
twine upload dist/*
