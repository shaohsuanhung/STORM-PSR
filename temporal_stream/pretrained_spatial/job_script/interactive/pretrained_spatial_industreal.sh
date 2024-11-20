#!/bin/bash
cd /hpc/home/shaohung/code/KeyStep/temporal_stream/pretrained_spatial
source activate /hpc/data/hpc-smc-internships/shaohung/environments/thesis

path_name=try_run_industreal_ep100_iter500
python train.py $path_name --run_path /shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage1 --epoch 100 --loss supcon --use_pretrained_weights --data_path  /shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/IndustReal/recordings  --syn_path /shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/industreal_cont --psr_label_path /shared/nl011006/res_ds_ml_restricted/shaohung/IndustReal_corrected_PSR 
python compute_embedding_from_dataset.py --run_name $path_name --checkpoint best.pth --loss SupCon --log_path /shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage1 --data_path  /shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/IndustReal/recordings  --syn_path /shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/industreal_cont --psr_label_path /shared/nl011006/res_ds_ml_restricted/shaohung/IndustReal_corrected_PSR 
