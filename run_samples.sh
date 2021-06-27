#!/usr/bin/env bash
source ./venv/bin/activate
export PYTHONPATH="$PWD"
python3 samples/polynomial_fit/sample.py
python3 samples/curve_fit/sample.py
python3 samples/plane_fit/sample.py
python3 samples/icp_fit/sample.py