#!/bin/bash

source activate {Your environment name}

#-- 1.1 Pretrained the spatial encoder
cd STORM-PSR/temporal_stream/pretrain_spatial_encoder
path_name_stage1=your_run_name
EPOCH_stage1=100
log_path_stage1=your_log_path
video_data_path=your_dataset_path
syn_path=your_synthetic_data_path
psr_label_path=your_label_path
loss_type=supcon

python train.py $path_name_stage1 --run_path $log_path_stage1 --epoch $EPOCH_stage1 --loss supcon --use_pretrained_weights --data_path $video_data_path --syn_path $syn_path --psr_label_path $psr_label_path

#-- 1.2 After training hte spatial encoder extract the spatial representation of video dataset 
python compute_embedding_from_dataset.py --run_name $path_name_stage1 --checkpoint best.pth --loss $loss_type --log_path $log_path_stage1 --data_path  $video_data_path  --syn_path $syn_path --psr_label_path $psr_label_path


#-- 2. Train the temporal encoder 
cd STORM-PSR/temporal_stream/train_spatial_temporal
path_name_stage2=your_run_name
EPOCH_stage2=50
skip_factor=3
psr_label_path=your_psr_label_path
log_path_stage2=your_log_path_stage2    
embedding_CSV_data_dir=$log_path_stage1/$path_name_stage1/embeddings/best.pth

tmp_enc_config=configs/IndustReal/temporal-F65-dim128.yaml

python train.py --run_name  $path_name_stage2 --config $tmp_enc_config --dtype embedding --data_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path_stage2 --epoch $EPOCH_stage2 --skip_factor $skip_factor --sampling_strategy bimodal --lr 5e-3 --job_file_mode


#-- 2.2. Test the model with only forwarding the embedding to the temproal encoder
python test.py --run_name $path_name_stage2 --split test --checkpoint weights_$EPOCH_stage2 --dtype embedding --skip_factor $skip_factor --csv_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path

#-- 2.3. Evaluate model with only forwarding the embedding to the temproal encoder
cd STORM-PSR/evaluation
eval_path_stage2=$log_path_stage2/$path_name_stage2
rgb_video_path=your_rgb_video_path
proc_info=./utils/procedure_info_IndustReal.json
ckpt=weights_$EPOCH_stage2
conf_threshold=0.5
cum_conf_threshold=6.0
cum_decay=0.75
num_dig_psr=11

python evaluate_TemporalStream.py --run_path $eval_path_stage2 --rec_path $video_data_path --psr_label_path $psr_label_path --video_dir $rgb_video_path --procedure_info $proc_info --split test --checkpoint $ckpt --conf_threshold $conf_threshold --cum_conf_threshold $cum_conf_threshold --cum_decay $cum_decay --num_dig_psr $num_dig_psr --FPS 10 --width 1280 --height 720 --create_PSR_plot 


#-- 3.1 Fine-tuning the whole temporal stream (spatial encoder + temporal encoder +MLP multi-head classification head) 
cd STORM-PSR/temporal_stream/train_spatial_temporal
path_name_ft=you_run_name_for_fine_tuning
EPOCH_ft=50
psr_label_path=your_psr_label_path
log_path_ft=your_log_path_ft
video_path=your_video_path
pretrained_file=$log_path_stage2/$path_name_stage1/checkpoints/weights_$EPOCH_stage2.pth
spatial_pretrained=$log_path_stage1/$path_name_stage1/checkpoints/best.pth
temporal_stream_config=configs/IndustReal/temporal-F65-dim128.yaml
skip_factor=3

python fine_tuning.py --run_name  $path_name_ft --config $temporal_stream_config --pretrained_weight $pretrained_file --spatial_pretrained_weight $spatial_pretrained --dtype video --data_dir $video_path --psr_label_path $psr_label_path --log_path $log_path_ft --epoch $EPOCH_ft --skip_factor $skip_factor --sampling_strategy bimodal --lr 5e-3 --job_file_mode --sanity_check

#-- 3.2. Test and evaluate the model by forwarding the video to the whole temporal stream model.
python test.py --run_name $path_name_ft --split test --checkpoint weights_$EPOCH_ft --dtype video --skip_factor $skip_factor --data_dir $video_path --psr_label_path $psr_label_path --log_path $log_path_ft

#-- 3.3 Evaluate model with forwarding the video. 
cd STORM-PSR/PSR_evaluation
eval_path_ft=$log_path_ft/$path_name_ft
rgb_video_path=your_rgb_video_path
proc_info=./utils/procedure_info_IndustReal.json
ckpt=weights_$EPOCH_ft
conf_threshold=0.5
cum_conf_threshold=6.0
cum_decay=0.75
num_dig_psr=11
python evaluate_TemporalStream.py --run_path $eval_path_ft --rec_path $video_data_path --psr_label_path $psr_label_path --video_dir $rgb_video_path --procedure_info $proc_info --split test --checkpoint $ckpt --conf_threshold $conf_threshold --cum_conf_threshold $cum_conf_threshold --cum_decay $cum_decay --num_dig_psr $num_dig_psr --FPS 10 --width 1280 --height 720 --create_PSR_plot

