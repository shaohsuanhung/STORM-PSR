#!/bin/bash

#- Data
run_path=C:\Users\franc\Desktop\master_thesis\Table1\IndustReal\YOLOv8-m
save_dir_base=C:\Users\franc\Desktop\master_thesis\Table1\IndustReal\YOLOv8-m
rec_path=C:\Users\franc\Desktop\master_thesis\dataset\industreal
psr_label_path=C:\Users\franc\Desktop\master_thesis\dataset\industreal
video_dir=C:\Users\franc\Desktop\master_thesis\dataset\industreal\all_rgb_videos
procedure_info=./utils/procedure_info_IndustReal.json
split=test

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


python evaluate_keystep.py --run_path $run_path --save_dir_base $save_dir_base --rec_path $rec_path --psr_label_path $psr_label_path  --procedure_info $procedure_info --split $split --conf_threshold $conf_threshold --cum_conf_threshold $cum_conf_threshold --cum_decay $cum_decay --num_dig_psr $num_dig_psr --temporal_win $temporal_window --video_dir $video_dir --create_video $create_video --create_PSR_plot $create_PSR_plot --FPS $FPS --width $width --height $height --bbox $bbox