#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate crai

echo "------------ Evaluating the model with the following hyperparameters  ------------"
cat ./evaluation_spain.inp
echo -e "--------------------------------------\n"

python -m climatereconstructionai.evaluate \
	--data-root-dir ../input_data/ \
	--mask-dir ../input_data/masks/ \
	--log-dir ./logs/ \
	--model-dir ./snapshots/ckpt \
	--evaluation-dirs ./evaluation/ \
	--data-names test_data_m_s_2021_2021_pen.nc \
	--mask-names test_hourly_mask_era5SL_010_144x144_m_s_2021_2021_pen.nc \
	--eval-names eval_ERA5Land_Spain-WS_2021-2021 \
	--device cuda \
	--load-from-file ./evaluation_spain.inp
