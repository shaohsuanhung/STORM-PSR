#!/bin/bash
#--1. Pre-trained spatail encdoer
path_name_stage1=try_run_industreal_100
EPOCH_stage1=10
log_path_stage1=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage1
#--2. Train temporal encoder 
cd /hpc/home/shaohung/code/KeyStep/temporal_stream/train_spatial_temporal
path_name_stage2=try_run_industreal_stage2_ft
EPOCH_stage2=50
psr_label_path=/shared/nl011006/res_ds_ml_restricted/shaohung/IndustReal_corrected_PSR 
log_path=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage2
video_path=/shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/IndustReal/recordings
pretrained_file=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage2/try_run_industreal_stage2/checkpoints/weights_50.pth
spatial_pretrained=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage1/try_run_industreal_100/checkpoints/best.pth

python fine_tuning.py --run_name  $path_name_stage2 --config configs/IndustReal/SimStepNet_F65-dim128.yaml --pretrained_weight $pretrained_file --spatial_pretrained_weight $spatial_pretrained --dtype video --data_dir $video_path --psr_label_path $psr_label_path --log_path $log_path --epoch $EPOCH_stage2 --skip_factor 3 --sampling_strategy bimodal --lr 5e-3 --job_file_mode --sanity_check
# python test.py --run_name $path_name_stage2 --split test --checkpoint weights_$EPOCH_stage2 --dtype video --skip_factor 3 --data_dir $video_path --psr_label_path $psr_label_path --log_path $log_path
