#!/bin/bash

source activate {Your environment name}
cd STORM-PSR/temporal_stream/pretrain_spatial_encoder

#-- Run path
path_name_pt=your_run_name
log_path_pt=your_log_path


#-- Hyperparameters
EPOCH_stage1=100
model=vit_small_patch16_224.augreg_in21k_ft_in1k
loss_type=supcon
hidden=128
lr=1e-3
scheduler=cosine_restart
T_0=40
lr_gamma=0.975
lr_steps=10
warmup_epochs=10
n_iters=500
margin=0.01
weight_decay=1e-6
temperature=0.07
stop_after=40

#-- Data path
video_data_path=your_dataset_path
syn_path=your_synthetic_data_path
psr_label_path=your_label_path

#-- Contrastive learning setting 
n_classes=11
n_real=8
n_synth=8
n_bg=8
random_seed=1234
n_frames=20

# Data augmentation setting
img_w=224
img_h=224
channels=3
workers=4
batch_size=32
kernel_size=5
sigma_l=0.01
sigma_h=2.0
bright=0.1
sat=0.7
cont=0.1

python train.py $path_name_pt --run_path $log_path_pt --epoch $EPOCH_stage1 --model $model --loss $loss_type --use_pretrained_weights --hidden $hideen --lr $lr --scheduler $scheduler --T_0 $T_0 --lr_gamma $lr_gamma --lr_step $lr_steps --warmup $warmup_epochs --n_iters $n_iters --margin $margin --weight_decay $weight_decay --temperature $temperature --stop_after $stop_after --data_path $video_data_path --syn_path $syn_path --psr_label_path $psr_label_path --n_classes $n_classes --n_real $n_real --n_synth $n_synth --n_bg $n_bg --exclude_bg $exclude_bg --seed $random_seed --n_frames $n_frames --img_w $img_w --img_h $img_h --channels $channels --workers $workers --batch_size $batch_size --kernel_size $kernel_size --sigma_l $sigma_l --sigma_h $sigma_h --bright $bright --sat $sat --cont $cont 