#!/bin/bash

source activate {Your environment name}

#-- 1.1 Pretrained the spatial encoder
cd STORM-PSR/temporal_stream/pretrain_spatial_encoder

data_path=your_dataset_path
psr_label_path=your_label_path
log_path=your_log_path

run_name=your_run_name
checkpoint=best.pth
config=configs/IndustReal/temporal-F65-dim128.yaml
loss=supcon
num_workers=4   


python compute_embedding_from_dataset.py --run_name $run_name --checkpoint $checkpoint --loss $loss --config $config --log_path $log_path --data_path  $data_path  --psr_label_path $psr_label_path --num_workers $num_workers

