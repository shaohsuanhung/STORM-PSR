#-- 2. Train the temporal encoder 
cd STORM-PSR/temporal_stream/train_spatial_temporal
source activate {Your environment name}

# Data path setting
data_dir=your_data_dir
psr_label_path=your_psr_label_path
log_path=your_log_path
ckpt_path=your_ckpt_path
csv_dir=your_csv_dir
asd_label=your_asd_label
dtype=embedding

skip_factor=3


# Run name
run_name=your_run_name
checkpoint=your_checkpoint
split=test


# Test using embeddings
python test.py --run_name $run_name --checkpoint $checkpoint --split $split --data_dir $data_dir --csv_dir $csv_dir --asd_label $asd_label --psr_label_path $psr_label_path --log_path $log_path  --dtype embedding --skip_factor $skip_factor

# # Test using videos
# python test.py --run_name $run_name --checkpoint $checkpoint --split $split --data_dir $data_dir --asd_label $asd_label --psr_label_path $psr_label_path --log_path $log_path  --dtype video --skip_factor $skip_factor