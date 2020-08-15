#!/bin/bash

python setup.py sdist
pip install twine
twine upload dist/*
