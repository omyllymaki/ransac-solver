#!/usr/bin/env bash
source ./venv/bin/activate
export PYTHONPATH="$PWD"
echo "Sample: polynomial fit"
python3 samples/polynomial_fit/sample.py
echo "Sample: curve fit"
python3 samples/curve_fit/sample.py
echo "Sample: plane fit"
python3 samples/plane_fit/sample.py
echo "Sample: ICP fit"
python3 samples/icp_fit/sample.py
echo "Sample: classification"
python3 samples/classification/sample.py