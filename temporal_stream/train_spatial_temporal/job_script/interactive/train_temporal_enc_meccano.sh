#!/bin/bash
#--1. Pre-trained spatail encdoer
path_name_stage1=try_run_meccano_100
EPOCH_stage1=10
log_path_stage1=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage1
#--2. Train temporal encoder 
cd /hpc/home/shaohung/code/KeyStep/temporal_stream/train_spatial_temporal
path_name_stage2=try_run_meccano_stage2
EPOCH_stage2=50
psr_label_path=/shared/nl011006/res_ds_ml_restricted/shaohung/MECCANO-PSR
log_path=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage2
embedding_CSV_data_dir=$log_path_stage1/$path_name_stage1/embeddings/best.pth

# python train.py --run_name  $path_name_stage2 --config configs/MECCANO/temporal-F65-dim128.yaml --dtype embedding --data_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path --epoch $EPOCH_stage2 --skip_factor 3 --sampling_strategy bimodal --lr 5e-3
# python test.py --run_name $path_name_stage2 --split train --checkpoint weights_$EPOCH_stage2 --dtype embedding --skip_factor 3 --csv_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path
python test.py --run_name $path_name_stage2 --split test --checkpoint weights_$EPOCH_stage2 --dtype embedding --skip_factor 3 --csv_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path
