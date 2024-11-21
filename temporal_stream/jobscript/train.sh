#-- 2. Train the temporal encoder 
cd STORM-PSR/temporal_stream/train_spatial_temporal
source activate {Your environment name}

# Data path setting
data_dir=your_data_dir
psr_label_path=your_psr_label_path
log_path=your_log_path
ckpt_path=your_ckpt_path

# Hyperparameters
path_name_stage2=your_run_name
run_path=your_run_path
epochs=100
model=vit_small_patch16_224.augreg_in21k_ft_in1k
loss=
EPOCH_stage2=50
skip_factor=3
psr_label_path=your_psr_label_path
log_path_stage2=your_log_path_stage2    
embedding_CSV_data_dir=$log_path_stage1/$path_name_stage1/embeddings/best.pth

tmp_enc_config=configs/IndustReal/temporal-F65-dim128.yaml
dtype=embedding

# hpyerparameters
lr=1e-3
scheduler=cosine_restart
T_0=5
lr_gamma=0.975
lr_steps=5
warmup=3
weight_decay=5e-4
batch_size=256
workers=16
warmup_rate=1e-3
sampling_strategy=bimodal
n_iter=160000
exe_mode=no_error



python train.py --data_dir $data_dir --run_name  $path_name_stage2 --config $tmp_enc_config --dtype $dtype --data_dir $embedding_CSV_data_dir --psr_label_path $psr_label_path --log_path $log_path_stage2 --ckpt_dir $ckpt_path --epoch $EPOCH_stage2 --skip_factor $skip_factor --lr $lr --job_file_mode --scheduler $scheduler --T_0 $T_0 --lr_gamma $lr_gamma --lr_step $lr_steps --warmup $warmup --weight_decay $weight_decay --batch_size $batch_size --workers $workers --warmup_rate $warmup_rate --n_iter $n_iter --exe_mode $exe_mode --sampling_streragy $sampling_strategy