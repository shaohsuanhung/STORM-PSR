#!/bin/bash
source activate {Your environment name}
cd STORM-PSR/temporal_stream/train_spatial_temporal

data_dir=your_data_dir
psr_label_path=your_psr_label_path
log_path=your_log_path


run_path=your_run_path
config=configs/IndustReal/temporal-F65-dim128.yaml


lr=5e-3
scheduler=consine_restart
T_0=5
lr_gamma=0.975
lr_step=5
warmup=2
weights_decay=1e-4
epochs=100
batch_size=8
workers=8
warmpup_rate=1e-3
sampling_strategy=bimodal
skip_factor=3

python fine_tuning.py --run_name $run_path --config $config --data_dir $data_dir --psr_label_path $psr_label_path --log_path $log_path --lr $lr --scheduler $scheduler --T_0 $T_0 --lr_gamma $lr_gamma --lr_step $lr_step --warmup $warmup --weights_decay $weights_decay --epochs $epochs --batch_size $batch_size --workers $workers --warmpup_rate $warmpup_rate --sampling_strategy $sampling_strategy --skip_factor $skip_factor --job_file_mode