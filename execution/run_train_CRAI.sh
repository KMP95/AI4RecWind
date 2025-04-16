#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate crai

echo "------------ Training the model with the following hyperparameters  ------------"
cat ./ws_crai_training.inp
echo -e "--------------------------------------\n"

python -m climatereconstructionai.train \
	--data-root-dir ../input_data/ \
	--mask-dir ../input_data/masks/ \
	--log-dir ./logs/ \
	--snapshot-dir ./snapshots/  \
	--device cuda \
	--load-from-file ./ws_crai_training.inp 
