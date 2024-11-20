#!/bin/bash
# conda activate keystep

#-- 1. Training the temporal encoder 
cd C:/Users/franc/Desktop/master_thesis/src/temporal-video-understanding-feature-temporal-feature-study/src/KeyStep/temporal_stream/pretrained_spatial


name=test_run_industreal_stage1
run_path=C:/Users/franc/Desktop/master_thesis/src/temporal-video-understanding-feature-temporal-feature-study/src/data_log
ep=50
data_path=C:/Users/franc/Desktop/master_thesis/dataset/industreal_resized
syn_path=na
data_psr=C:/Users/franc/Desktop/master_thesis/dataset/industReal_PSR
log_path=C:/Users/franc/Desktop/master_thesis/src/temporal-video-understanding-feature-temporal-feature-study/src/data_log
psr_path=C:/Users/franc/Desktop/master_thesis/dataset/industReal_PSR

python train.py $name --run_path $run_path --epoch $ep --loss supcon --use_pretrained_weights --data_path $data_path --syn_path $syn_path --psr_label_path $data_psr --n_real 16 --n_synth 0 --n_bg 0

#-- 2. Extract embedding after the training
python compute_embedding_from_dataset.py --run_name $name --checkpoint best.pth --loss SupCon --log_path $log_path --data_path  $data_path  --syn_path $syn_path --psr_label_path $psr_path
