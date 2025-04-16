#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate crai

python metrics.py \
	--exp_name "loss_1_with_normalization_CRAI  – 2M steps - bs = 64" \
	--root ../ \
	--data_root ../input_data/ \
	--mask_root ../input_data/masks/ \
	--split test \
	--save_pred_dir ../execution/evaluation/ \
	--dataset_name "test_data_m_s_2021_2021_pen.nc" \
	--reconstructed_file "eval_ERA5Land_Spain-WS_2021-2021_output.nc" \
	--device cuda \
	--metrics l1Error l2Error Bias_time Bias_space Corr_time Corr_space RMSE_time RMSE_space \