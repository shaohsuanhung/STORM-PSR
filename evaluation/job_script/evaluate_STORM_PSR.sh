#!/bin/bash

#- Data
run_path=PATH_to_the_run_folder
save_dir_base=PATH_to_the_save_folder
rec_path=PATH_to_the_rec_folder
psr_label_path=PATH_to_the_psr_label_folder
video_dir=PATH_to_the_video_folder
split=test
#- Give the prior procedure information either IndustReal dataset or MECANO dataset
# procedure_info=./utils/procedure_info_MECCANO.json
procedure_info=./utils/procedure_info_IndustReal.json


#- Evaluation parameters setting
conf_threshold=0.5
cum_conf_threshold=8.0
cum_decay_rate=0.75
num_dig_psr=11
temporal_window=256

#- Setting for generating resulted videos and plots
create_video=False
create_PSR_plot=False
FPS=10
width=1280
height=720
bbox=True


python evaluate_STORM_PSR.py --run_path $run_path --save_dir_base $save_dir_base --rec_path $rec_path --psr_label_path $psr_label_path  --procedure_info $procedure_info --split $split --conf_threshold $conf_threshold --cum_conf_threshold $cum_conf_threshold --cum_decay $cum_decay --num_dig_psr $num_dig_psr --temporal_win $temporal_window --video_dir $video_dir --create_video $create_video --create_PSR_plot $create_PSR_plot --FPS $FPS --width $width --height $height --bbox $bbox