# Learning to Recognize Correctly Completed Procedure Steps in Egocentric Assembly Videos through Spatio-Temporal Modeling
We propose Spatio-Temporal Occlusion-Resilient Modeling for Procedure Step Recognition (STORM-PSR), a dual-stream framework for Procedure step recognition (PSR) that leverages both spatial and temporal features. PSR task aims to identify all correctly completed steps and their sequential order in videos of procedural tasks. The existing state-of-the-art models rely solely on detecting assembly object states in individual video frames.

STORM-PSR is evaluated on the MECCANO and IndustReal datasets, reducing the average delay between actual and predicted assembly step completions by 11.2\% and 26.1\%, respectively, compared to [prior methods](https://openaccess.thecvf.com/content/WACV2024/papers/Schoonbeek_IndustReal_A_Dataset_for_Procedure_Step_Recognition_Handling_Execution_Errors_WACV_2024_paper.pdf).


## Getting Started
1. Clone the repository
```terminal
git clone https://github.com/shaohsuanhung/STORM-PSR.git
cd STORM-PSR
```
2. Setup and activate your conda environment, and install dependencies
```terminal
conda create --name storm-psr python=3.12.2
conda activate storm-psr
pip install -r STORM-PSR/requirements.txt
```
3. Dataset preparation
   * IndustReal:  Please refer to  [IndustReal github page](https://github.com/TimSchoonbeek/IndustReal) to download the datasets.
   * MECCANO: Please refer to [MECCANO github page](https://github.com/fpv-iplab/MECCANO) to download the datasets. <font color="red"> ADD link to download the MECCANO ASD & PSR annotation !!!</font>


## Object detection stream model
Please refer to [assembly state detection tutorial](https://github.com/TimSchoonbeek/IndustReal/tree/main/ASD) to train the object detection stream model.

## Temporal Stream model
### To train a temporal-stream model
```bash
cd STORM-PSR/temporal_stream
sh scripts/train.sh
```
### To test a temporal-stream model
```bash
cd STORM-PSR/temporal_stream
sh scripts/test.sh
```

## To evaluate a model
```bash
sh scripts/evaluate_STORM.sh  # To evaluate the STORM-PSR model
sh scripts/evaluate_ODStream.sh # To evaluate the Object Detection Stream
sh scripts/evaluate_TemporalStream.sh  # To evaluate the Temporal Stream model
```

### To visualize spatial embedding using UMAP / t-SNE
```bash
```