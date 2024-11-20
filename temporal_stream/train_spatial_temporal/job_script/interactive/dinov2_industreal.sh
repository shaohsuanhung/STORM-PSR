#!/bin/bash
# conda activate keystep

#-- 3. Training the temporal encoder 
cd C:/Users/franc/Desktop/master_thesis/src/temporal-video-understanding-feature-temporal-feature-study/src/KeyStep/temporal_stream/train_spatial_temporal
path_name_stage2=train_on_dinov2_embeddings
EPOCH_stage2=50
skip_factor=3
psr_label_path=C:/Users/franc/Desktop/master_thesis/dataset/industReal_PSR
log_path_stage2=C:/Users/franc/Desktop/master_thesis/src/temporal-video-understanding-feature-temporal-feature-study/src/data_log
embedding_CSV_data_dir=C:/Users/franc/Desktop/master_thesis/dataset/industreal_embedding_ImgNet_Norm/dinov2_vits_pretrained
tmp_enc_config=configs/IndustReal/temporal-F65-dim384.yaml

python train.py --run_name  $path_name_stage2 --config $tmp_enc_config --dtype embedding --data_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path_stage2 --epoch $EPOCH_stage2 --skip_factor $skip_factor --sampling_strategy bimodal --lr 5e-3 --job_file_mode