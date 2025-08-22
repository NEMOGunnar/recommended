#!/bin/sh
git pull
conda activate recommended
python --version
python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt
