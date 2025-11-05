## Visualize Embedding of the trained model
## Installation 

```
$ git clone [Redacted, url of this repo in the public github]
$ cd STORM-PSR/temporal_stream/visualize_embedding
$ conda create -n storm-psr python=3.9 -y
$ conda activate storm-psr
$ pip install -r storm-psr/temporal_stream/visualize_embedding/requirements.txt
```

## Usage
The `visualized_utils.py` can be used to visualized the embedding by dimension reduction. We implement visualization in the t-SNE and UMAP. The `visualized_embedding.ipynb` help user to visualize the embedding in a more interactive way. 

## How to structure folders/embedding files for visualization?
We build 3 different load embedding functions since there are different ways that people store embeddings.
The loading function would return the dataframe in the exact same format. 
Please adapt to different function depence on how you file structure of your embedding csv files.

1.  Use `load_embedding_from_AllInOne_CSV` function in `visualize_util.py`, if the ebmeddings are stored in the following way:
    * All the embedding from different recordings are store in "**one** .csv file".
    * In that csv file, there are 4 columns: (1) filename, (2) frameID, (3) state label (4) embeddings: [N x 1], where N is the size of your embeddings.

2. Use `load_embedding_from_structural_folder` function in `visualize_util.py`, if ebmeddings are stored in the following way: 
```
{Embeddings folder}
|--- {recording name}
|       |--- embeddings.csv
|
|---  ....
|
|--- {recording name}
|       |--- embeddings.csv
```    
For this kind of structure, the labels are not included in the {run_name} folder.
Labels files are read from directory from differnet folder, structure in the same way as [IndustReal dataset](https://github.com/TimSchoonbeek/IndustReal).


3. Use `load_embedding_from_PlainCSV` in `visualize_util.py`, if the ebmeddings are stored in the following way:
    * The embeddings of the whole dataset are store in a csv in that format that each row record one [N x 1] embedding, they are all coming from images with bounding box.
    * To know the label of embeddings, there is also a labels.csv 
```
{Embedding folder}
|--- embeddings.csv
|--- labels.csv 
|--- error_embeddings.csv

# The 'labels.csv' file only have labels for embeddgins.csv, the error_embeddings.csv are all labeled as '23' (error state defined in the IndustReal, you have you adapt the label to your own error state).
```