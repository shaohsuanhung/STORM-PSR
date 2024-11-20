#!/bin/bash
cd /hpc/home/shaohung/code/KeyStep/temporal_stream/pretrained_spatial
source activate /hpc/data/hpc-smc-internships/shaohung/environments/thesis
path_name=try_run_meccano_ep100_iter500
python train.py $path_name --run_path /shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage1 --epoch 100 --loss supcon --use_pretrained_weights --data_path  /shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/meccano/frames --psr_label_path /shared/nl011006/res_ds_ml_restricted/shaohung/MECCANO-PSR --n_synth 0
python compute_embedding_from_dataset.py --run_name $path_name --checkpoint best.pth --loss SupCon --log_path /shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage1 --data_path  /shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/meccano/frames --psr_label_path /shared/nl011006/res_ds_ml_restricted/shaohung/MECCANO-PSR
