#!/bin/bash
# file: start_tf_kernel.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensorflowintel
exec "$@"
