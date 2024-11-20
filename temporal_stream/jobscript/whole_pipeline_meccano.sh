#!/bin/bash

source activate /hpc/data/hpc-smc-internships/shaohung/environments/thesis

#-- 1. Pretrained the spatial encoder
cd /hpc/home/shaohung/code/KeyStep/temporal_stream/pretrained_spatial
path_name_stage1=try_run_meccano_ep100_iter500
EPOCH_stage1=100
log_path_stage1=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage1
video_data_path=/shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/meccano/frames
syn_path=None
psr_label_path=/shared/nl011006/res_ds_ml_restricted/shaohung/MECCANO-PSR

python train.py $path_name_stage1 --run_path $log_path_stage1 --epoch $EPOCH_stage1 --loss supcon --use_pretrained_weights --data_path $video_data_path --syn_path $syn_path --psr_label_path $psr_label_path
#-- 2. After training hte spatial encoder extract the spatial representation of video dataset 
python compute_embedding_from_dataset.py --run_name $path_name_stage1 --checkpoint best.pth --loss SupCon --log_path $log_path_stage1 --data_path  $video_data_path  --syn_path $syn_path --psr_label_path $psr_label_path


#-- 3. Training the temporal encoder 
cd /hpc/home/shaohung/code/KeyStep/temporal_stream/train_spatial_temporal
path_name_stage2=try_run_meccano_stage2_tmp_enc
EPOCH_stage2=50
skip_factor=3
psr_label_path=/shared/nl011006/res_ds_ml_restricted/shaohung/MECCANO-PSR
log_path_stage2=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage2
embedding_CSV_data_dir=$log_path_stage1/$path_name_stage1/embeddings/best.pth
tmp_enc_config=configs/MECCANO/temporal-F65-dim128.yaml

python train.py --run_name  $path_name_stage2 --config $tmp_enc_config --dtype embedding --data_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path_stage2 --epoch $EPOCH_stage2 --skip_factor $skip_factor --sampling_strategy bimodal --lr 5e-3 --job_file_mode


#-- 3.5 Test the temproal encoder model by only forwarding the embedding
python test.py --run_name $path_name_stage2 --split test --checkpoint weights_$EPOCH_stage2 --dtype embedding --skip_factor $skip_factor --csv_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path
#------ Evaluate model with only forwarding the embedding to the temproal encoder
cd /hpc/home/shaohung/code/KeyStep/PSR_evaluation
eval_path_stage2=$log_path_stage2/$path_name_stage2
rgb_video_path=/shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/meccano/MECCANO_RGB_Videos/MECCANO_RGB_Videos
proc_info=./utils/procedure_info_MECCANO.json
ckpt=weights_$EPOCH_stage2
conf_threshold=0.5
cum_conf_threshold=1.0
cum_decay=0.75
num_dig_psr=11
python evaluate_TemporalStream.py --run_path $eval_path_stage2 --rec_path $video_data_path --psr_label_path $psr_label_path --video_dir $rgb_video_path --procedure_info $proc_info --split test --checkpoint $ckpt --conf_threshold $conf_threshold --cum_conf_threshold $cum_conf_threshold --cum_decay $cum_decay --num_dig_psr $num_dig_psr --FPS 10 --width 1280 --height 720 --create_PSR_plot


#-- 4. Fine-tuning the whole temporal stream (spatial encoder + temporal encoder +MLP multi-head classification head) 
cd /hpc/home/shaohung/code/KeyStep/temporal_stream/train_spatial_temporal
path_name_ft=try_run_meccano_stage2_ft
EPOCH_ft=50
psr_label_path=/shared/nl011006/res_ds_ml_restricted/shaohung/MECCANO-PSR
log_path_ft=/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/handover/stage2
video_path=/shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/meccano/MECCANO_RGB_Videos/MECCANO_RGB_Videos
pretrained_file=$log_path_stage2/$path_name_stage1/checkpoints/weights_$EPOCH_stage2.pth
spatial_pretrained=$log_path_stage1/$path_name_stage1/checkpoints/best.pth
temporal_stream_config=configs/MECCCANO/temporal-F65-dim128.yaml
python fine_tuning.py --run_name  $path_name_ft --config $temporal_stream_config --pretrained_weight $pretrained_file --spatial_pretrained_weight $spatial_pretrained --dtype video --data_dir $video_path --psr_label_path $psr_label_path --log_path $log_path_ft --epoch $EPOCH_ft --skip_factor $skip_factor --sampling_strategy bimodal --lr 5e-3 --job_file_mode --sanity_check

#-- 4.5 Test the whole temporal stream model by forwarding the video
python test.py --run_name $path_name_ft --split test --checkpoint weights_$EPOCH_ft --dtype video --skip_factor $skip_factor --data_dir $video_path --psr_label_path $psr_label_path --log_path $log_path_ft
#------ Evaluate model with forwarding the video. 
cd /hpc/home/shaohung/code/KeyStep/PSR_evaluation
eval_path_ft=$log_path_ft/$path_name_ft
rgb_video_path=/shared/nl011006/res_ds_ml_restricted/TimSchoonbeek/meccano/MECCANO_RGB_Videos/MECCANO_RGB_Videos
proc_info=./utils/procedure_info_MECCANO.json
ckpt=weights_$EPOCH_ft
conf_threshold=0.5
cum_conf_threshold=1.0
cum_decay=0.75
num_dig_psr=11
python evaluate_TemporalStream.py --run_path $eval_path_ft --rec_path $video_data_path --psr_label_path $psr_label_path --video_dir $rgb_video_path --procedure_info $proc_info --split test --checkpoint $ckpt --conf_threshold $conf_threshold --cum_conf_threshold $cum_conf_threshold --cum_decay $cum_decay --num_dig_psr $num_dig_psr --FPS 12 --width 1920 --height 1080 --create_PSR_plot

