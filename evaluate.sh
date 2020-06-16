#!/bin/sh
source ~/pytorch_env/bin/activate

cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py

