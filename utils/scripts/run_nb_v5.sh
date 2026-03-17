#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate snn_env
jupyter nbconvert --to notebook --execute --inplace 'analysis/pjm_hybrid/pjm_hybrid.ipynb' > /tmp/nbconvert_v5.log 2>&1
