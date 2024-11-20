## Pre-train the Spatail encoder in Weakly-supervised Contrastive Learning Setting
## Installation 

```
$ git clone [To be add, url of this repo in the public github]
$ cd KeyStep/temporal_stream/pretrained_spatial
$ conda create -n KeyStep python=3.9 -y
$ conda activate KeyStep
$ pip install -r KeyStep/temporal_stream/requirements.txt
```

## Usage
The scripts in this subfolder are mostly adpot from our previous work. ([paper](https://arxiv.org/abs/2408.11700), [repo](https://github.com/TimSchoonbeek/AssemblyStateRecognition)). The only different is that we train the spatial encoder in weakly-superivised contrastive learning setting. The weakly-supervised sampling method (also called Key-frame sampling) is implemented in the script `temporal_aware_datasets.py` at [here](./temporal_aware_CL/temporal_aware_datasets.py). 


### Pre-train the spatial encoder
The following command can be used to train the spatial model (default hyperparameters can be found in our paper):
```
python train.py  [--run_path RUN_PATH]\
                 [--epochs EPOCHS]\
                 [--model MODEL]\
                 [--loss LOSS]\
                 [--use_pretrained_weights]\
                 [--hidden HIDDEN]\
                 [--lr LR]\
                 [--scheduler SCHEDULER]\
                 [--T_0 T_0]\
                 [--lr_gamma LR_GAMMA]\
                 [--lr_step LR_STEP]\
                 [--warmup WARMUP]
                 [--n_iters N_ITERS]\
                 [--margin MARGIN]\
                 [--weight_decay WEIGHT_DECAY]\
                 [--temperature TEMPERATURE] \
                 [--stop_after STOP_AFTER]\
                 [--data_path DATA_PATH] \
                 [--syn_path SYN_PATH]\
                 [--psr_label_path PSR_LABEL_PATH]\
                 [--n_classes N_CLASSES]
                 [--n_real N_REAL]\
                 [--n_synth N_SYNTH] \
                 [--n_bg N_BG]\
                 [--img_w IMG_W]\
                 [--img_h IMG_H] \
                 [--exclude_bg]\
                 [--seed SEED]\
                 [--channels CHANNELS]\
                 [--workers WORKERS] \
                 [--batch_size BATCH_SIZE]\
                 [--kernel_size KERNEL_SIZE]\
                 [--sigma_l SIGMA_L]\
                 [--sigma_h SIGMA_H]\
                 [--bright BRIGHT]\
                 [--sat SAT] \
                 [--cont CONT]\
                 [--rotate]\
                 [--n_frames N_FRAMES]
```
### Extracting the spatial representations
The following command can be used to encod the video frames into embeddings (sptail representations in the lower dimensional space).
```
python compute_embedding_from_dataset.py [--data_path DATA_PATH]\
                                         [--psr_label_path PSR_LABEL_PATH]\
                                         [--log_path LOG_PATH]\
                                         [--run_name RUN_NAME]\
                                         [--checkpoint CHECKPOINT]\
                                         [--config CONFIG]\
                                         [--loss LOSS]\
                                         [--num_workers]\
```

### Job script
To automated the training pipeline, we provided the shell script to automated the training and testing process. Please refer to [KeyStep/temporal_stream/pretrained_spatial/job_script](./job_script/interactive/). Indicating the path of the config file used to train the model by calling:
```
sh job_script/interactive/pretrained_spatial_industreal.sh
```