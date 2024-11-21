'''
The script is for visualizing embedding extracted from spatial encoder.
Implement features:
    1. load, standarized and give statistics from the embedding csv file.
    2. Fit embeddings with PCA, t-SNE and UMAP, and return corresponding dataframe, which have 4 columns: {filename, frameID, state, embedding}. One row is one embedding data point. 
    3. Visualize 2D/3D scatter plot. 
    4. Convert frame-by-frame embedding trajectory in feature space to video.

'''
import pandas as pd
import os
import numpy as np
import random
from pathlib import Path
import glob
import cv2
import math 
import datatable as dt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from PIL import Image
from natsort import natsorted
from numpy.typing import ArrayLike

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#- Clustering metrics
from DBCV import DBCV
from dbcv import dbcv
from cdbw import CDbw

#- GLOBAL SETTING
#-- Plot setting
_new_black = '#373737'
sns.set_theme(style='ticks', font_scale=0.75, rc={
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'axes.labelpad': 2,
    'axes.linewidth': 0.5,
    'axes.titlepad': 4,
    'lines.linewidth': 0.5,
    'legend.fontsize': 9,
    'legend.title_fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.size': 2,
    'xtick.major.pad': 1,
    'xtick.major.width': 0.5,
    'ytick.major.size': 2,
    'ytick.major.pad': 1,
    'ytick.major.width': 0.5,
    'xtick.minor.size': 2,
    'xtick.minor.pad': 1,
    'xtick.minor.width': 0.5,
    'ytick.minor.size': 2,
    'ytick.minor.pad': 1,
    'ytick.minor.width': 0.5,

    # Avoid black unless necessary
    'text.color': _new_black,
    'patch.edgecolor': _new_black,
    'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
    'hatch.color': _new_black,
    'axes.edgecolor': _new_black,
    # 'axes.titlecolor': _new_black # should fallback to text.color
    'axes.labelcolor': _new_black,
    'xtick.color': _new_black,
    'ytick.color': _new_black

})
#-- Global variables targets, colors and markers
#   for functions: (1) plot_2D, (2)analyze_variation, (3) plot_all_states
targets = [str(idx) for idx in range(0, 24)]
colors = ['#db5f57', '#db8057', '#dba157', '#dbc257', '#d3db57', '#b2db57', \
          '#91db57', '#70db57', '#57db5f', '#57db80', '#57dba1', '#57dbc2', \
          '#57d3db', '#57b2db', '#5791db', '#5770db', '#5f57db', '#8057db', \
          '#a157db', '#c257db', '#db57d3', '#db57b2', '#db5791', '#db5770', \
          '#57b2db', '#5791db', '#5770db']

markers = ['.', '^', 'v', '<', '>', 's', \
            'D', 'd', 'p', 'h', 'H', '8', \
            '>', '*', '.', 'P', 'x', '+', \
            '1', '2', '3', '4', '|', 'X']

#- START of the loading function
'''
We build 3 different load embedding functions since there are different ways that people store embeddings.
The loading function would return the dataframe in the exact same format. 
Please adapt to different function depence on how you file structure of your embedding csv files.

(1). Use "load_embedding_from_AllInOne_CSV" function, if the ebmeddings are stored in the following way: # loading_embedding_df
    * All the embedding from different recordings are store in "one .csv file".
    * In the csv file, there are 4 columns: (1) filename, (2) frameID, (3) state label (4) embeddings: [N x 1], where N is the size of your embeddings.

(2). Use "load_embedding_from_structural_folder" function, if the ebmeddings are stored in the following way: # , load_embedding_and_label_df
    run_name
    |- {recording name}
        |- embeddings.csv
    |
    |- {recording name}
        |- embeddings.csv
    
    For this kind of structure, the labels are not included in the {run_name} folder.
    Labels files are read from directory from differnet folder.


(3). Use "load_embedding_from_PlainCSV", if the ebmeddings are stored in the following way: # , load_emedding_df_one_embeddingfile
    * The embeddings of the whole dataset are store in a csv in that format that each row record one [N x 1] embedding, they are all coming from images with bounding box.
    * To know the label of embeddings, there is also a labels.csv 
    run_name
    |- embeddings.csv
    |- labels.csv # Only have labels for embeddgins.csv, the error_embeddings.csv are all labeled as '23' 
    |- error_embeddings.csv
'''

def load_embedding_from_AllInOne_CSV(data_file: str, statics: bool = False, standardize: bool = True) -> pd.DataFrame:
    ''' Given the path of csv file, and flags, (1)load data, (2) show statics and (3) standardization the data value
    input :
        data_file : str,  path to the csv file that you store embeddings in the format there are 4 columns: (1) filename, (2) frameID, (3) state label (4) embeddings: [N x 1].
        statics   : bool, If true, print the statics of the data.
        standarize: bool, If true standarize the embedding value. 
    
    output:
        df        : pandas.DataFrame, in the format of  {'filename': str, frameID: int, 'state': str, 'embedding': float}
    '''
    # Load data from a csv file
    embedding_data = dt.fread(data_file).to_pandas()
    embeddings = embedding_data['embedding'].tolist()
    state_labels = embedding_data['state'].astype(str)
    filenames = embedding_data['filename'].astype(str)
    frameIDs = embedding_data['frameID'].astype(int)

    # Convert embeddings from str to float
    for i, embed in enumerate(embeddings): 
        embed = embed.split('[')[-1].split(']')[0]
        row_ele = [float(ele) for ele in embed.split(',')]
        embeddings[i] = row_ele

    # standardize embeddings
    if standardize:
        print('Normalizing...')
        standardized_data = StandardScaler().fit_transform(embeddings)
        embeddings = []
        for std_embedding in standardized_data:
           embeddings.append(std_embedding)

    d = {'filename': filenames, 'frameID': frameIDs, 'state': state_labels, 'embedding': embeddings}
    df = pd.DataFrame(data=d)
    if statics:
        print("Labels from data:{}\n".format(state_labels.unique()))
        # Count of recording that we collected
        print("Num of recorderings:{}".format(len(filenames.unique())))
        print("Num of frame of each recording:{}".format(filenames.value_counts()))
        # Count number of state of all recording
        print("State count:\n{}\n".format(state_labels.value_counts()))
        # Count change of state in each single video
        for recording in df['filename'].unique():
            states = df[df['filename'].isin([recording])]['state']
            print('{} state ({} samples) in recording: {}'.format(len(states.unique()), len(states), recording))
            print(states.value_counts())
            print('-' * 30)
    return df

def load_embedding_from_structural_folder(data_path: str, label_path: str, statics:bool = False, normalized: bool = True) -> pd.DataFrame:
    ''' Given the path to csv files, and flags, (1)load, (2) show statics and (3) standardization the data value
    input:
        data_path : str, "directory" to csv files
        label_path: str, "directory" to folder that generated by IndusTreal/ASD/create_dataset_real.py, 
                         we extract label from the file name, e.g.: 01_assy_1_1_000003_01.jpg.
        statics   : bool, If true, print the statics of the data.
        standarize: bool, If true standarize the embedding value. 

    output:
        df        : pandas.DataFrame, in the format of {'filename': str, frameID: int, 'state': str, 'embedding': float}s
    '''
    # Read csv file under the given directory
    # data_path  = Path(data_path)
    # label_path = Path(label_path)
    state_list = []
    frame_list = []
    name_list = []
    embeddings = []


    if label_path is not None:
        with os.scandir(label_path) as it:
            label_path_list = list(it)
        label_path_list.sort(key=lambda x: x.name)

        for file in label_path_list:
            state = file.name[:-4].split('_')[-1]
            frameID = int(file.name[:-4].split('_')[-2])
            state_list.append(str(int(state)))
            frame_list.append(int(frameID))

    with os.scandir(data_path) as it:
        data_path_list = list(it)
    data_path_list.sort(key=lambda x: x.name)

    

    for entry in data_path_list:
        for file in glob.glob(entry.path + '/*.csv'):
            embedding_data = dt.fread(file).to_pandas()
            # Extract features
            for idx, row in embedding_data.iterrows():
                embeddings.append(row.values)
                name_list.append(entry.name)
            
                if label_path is None:
                    frame_list.append(int(idx)+1) # For MECCANO, the frame are started from 1
                    # No state label 
                    state_list.append(-1)

    # Normalize the embedding
    if normalized:
        print('Normalizing...')
        standardized_data = StandardScaler().fit_transform(embeddings)
        embeddings = []
        for data in standardized_data:
             embeddings.append(data)

    df =  pd.DataFrame(data={'filename':name_list,'frameID':frame_list, 'state':state_list, 'embedding':embeddings})
    if statics:
        print("Labels from data:{}\n".format(df['state'].unique()))
        # Count of recording that we collected
        print("Num of recorderings:{}".format(len(df['filename'].unique())))
        print("Num of frame of each recording:{}".format(df['frameID'].value_counts()))
        # Count number of state of all recording
        print("State count:\n{}\n".format(df['state'].value_counts()))
        # Count change of state in each single video
        for recording in df['filename'].unique():
            states = df[df['filename'].isin([recording])]['state']
            print('{} state ({} samples) in recording: {}'.format(len(states.unique()), len(states), recording))
            print(states.value_counts(sort=False))
            print('-' * 30)

    return df

def load_embedding_from_PlainCSV(data_file: str, label_file: str, statics: bool = False, normalized: bool = True, error_embed_flag: bool = False) -> pd.DataFrame:
    ''' Given the path of csv file, and flags, (1)load, (2) show statics and (3) standardization 
    input:

        data_file: str, a csv file is in the format: embeddings with dimension [N x 1]
        label file is: str, a another csv file that record the state of the embedding in each row. 
        statics   : bool, If true, print the statics of the data.
        standarize: bool, If true standarize the embedding value. 

    output df:
         df        : pandas.DataFrame, in the format of {'state': str, 'embedding': float}
    '''
    # embedding_data = pd.read_csv(data_path,engine = 'python')
    embedding_data = dt.fread(data_file).to_pandas()
    label_data = dt.fread(label_file).to_pandas()
    embeddings = []
    state_list = []
    # Extract features
    for idx, row in embedding_data.iterrows():
        embeddings.append(row.values)
    
    if error_embed_flag:
        # If the the csv only contain error embeddings, then there is no labels (since all of them are labeled state 23)
        for _ in range(len(embeddings)):
            state_list.append(str(23))
    else:
        for _, row in label_data.iterrows():
            state_list.append(str(row.values[0]))

    if normalized:
        print('Normalizing...')
        standardized_data = StandardScaler().fit_transform(embeddings)
        embeddings = []
        for data in standardized_data:
            embeddings.append(data)
        
    df =  pd.DataFrame(data={'state':state_list, 'embedding':embeddings})
    if statics:
        print("Labels from data:{}\n".format(df['state'].unique()))
        # Count number of state of all recording
        print("State count:\n{}\n".format(df['state'].value_counts()))
    
    return df

def load_embedding_from_structural_folder_PSR(data_path: str, psr_label_path: str,statics:bool = False, normalized: bool = True,execution_mode = 'no_error', category = None) -> pd.DataFrame:
   
    state_list = []
    frame_list = []
    name_list = []
    embeddings = []
    lenght_list = dict()
    action_list = []
    #-- Embedding
    with os.scandir(data_path) as it:
        data_path_list = list(it)
    data_path_list.sort(key=lambda x: x.name)

    
    for entry in data_path_list:
        for file in glob.glob(entry.path + '/*.csv'):
            embedding_data = dt.fread(file).to_pandas()
            lenght_list.update({entry.name:len(embedding_data)})
            # Extract features
            for idx, row in embedding_data.iterrows():
                embeddings.append(row.values)
                name_list.append(entry.name)
                frame_list.append(int(idx)+1) # For MECCANO, the frame are started from 1


    #-- PSR label
    # print(lenght_list)
    with os.scandir(psr_label_path) as it:
        psr_label_path_list = list(it)

    psr_label_path_list.sort(key=lambda x: x.name)
    for file in psr_label_path_list:
        _PSR_labels = dict()
        psr_file = dt.fread(glob.glob(os.path.join(file,'PSR_labels_raw.csv'))).to_pandas()

        sub_action_list = [[str(0) for _ in range(len(psr_file.iloc[0][1:]))] for _ in range(lenght_list[file.name])]
        for idx in range(len(psr_file)):
            row = psr_file.iloc[idx]
            label = list()
            for idx in range(len(row[1:])):
                # Load the raw psr label
                # Turn the -1 (execution error) to 0, if not going to learn the error execution.
                if ((execution_mode == 'no_error') and (int(row.iloc[idx+1]) == -1)):
                    label.append(0)
                else:
                    # Here the label would contain -1
                    label.append(int(row.iloc[idx+1]))

            _PSR_labels.update({int(row.iloc[0][:-4]): label})

        diff_labels = dict()
        for idx, (frame, state) in enumerate(_PSR_labels.items()):
            if idx == 0:
                prev_state = state
                continue
            else:
                diff_labels.update({frame: [str(np.abs(cur-prev))
                                    for (prev, cur) in zip(prev_state, state)]})  # [1,0,0] , [1,0,1]
            prev_state = state

        for frame, action in diff_labels.items():
            # KFA assume the follow 10 frames are also labeled as same action state. 
            if action in category:
                for idx in range(10):
                    sub_action_list[frame+idx] = action
        # Assign labels to list 
        action_list.extend(sub_action_list)
    

    action_list = [list_to_state_string(ele) for ele in action_list]
    # Normalize the embedding
    if normalized:
        print('Normalizing...')
        standardized_data = StandardScaler().fit_transform(embeddings)
        embeddings = []
        for data in standardized_data:
             embeddings.append(data)

    df =  pd.DataFrame(data={'filename':name_list,'frameID':frame_list, 'state':action_list, 'embedding':embeddings})
    if statics:
        print("Labels from data:{}\n".format(df['state'].unique()))
        # Count of recording that we collected
        print("Num of recorderings:{}".format(len(df['filename'].unique())))
        print("Num of frame of each recording:{}".format(df['frameID'].value_counts()))
        # Count number of state of all recording
        print("State count:\n{}\n".format(df['state'].value_counts()))
        # Count change of state in each single video
        for recording in df['filename'].unique():
            states = df[df['filename'].isin([recording])]['state']
            print('{} state ({} samples) in recording: {}'.format(len(states.unique()), len(states), recording))
            print(states.value_counts(sort=False))
            print('-' * 30)

    return df

#- END of the loading function
#- START of the dimensional reduction function
def list_to_state_string(state_list: list) -> str:
    state_string = str()
    for ele in state_list:
        state_string += ele

    return state_string  # From ['1','0', '1','1'] -> '1011'

def state_string_to_list(state_string: str) -> list:
    state_list = []
    idx = 0
    while idx < len(state_string):
        s = state_string[idx]
        if s == '1':
            state_list.append(1)
        elif s == '0':
            state_list.append(0)
        idx += 1
    return state_list  # From '1011' -> [1,0,1,1]
def get_PCA_df(pca_setting, df : pd.DataFrame, dimension: int, normalized: bool = False) -> pd.DataFrame:
    ''' Execute the PCA, dimensional reduction and return a dataframe
    input:
        pca_setting: scikit learn PCA setting.
        df         : pandas dataframe that got from load_embedding_and_label_df function.
        dimension  : int, number of dimension of the embedded space.
        normalized : bool, if True, perform min max normalization.

    output:
        final_df   : pandas dataframe that contain the coordinate in emedded sapce. 
                     Column: {filename, frameID, state, pc1, pc2, ....{depends on number of dimension you have gave}}
    
    '''
    # Example of pca_setting: PCA(n_components=dimension, svd_solver='full')
    cols = []
    for n_comps in range(0, dimension):
        cols.append('PC' + str(n_comps + 1))
    # pca = PCA(n_components=dimension, svd_solver='full')
    low_dim_feature = pca_setting.fit_transform(df['embedding'].tolist())
    if normalized:
        low_dim_feature = MinMaxScaler().fit_transform(low_dim_feature)
    pca_df = pd.DataFrame(data=low_dim_feature, columns=cols)
    final_df = pd.concat([df['state'], df['filename'], df['frameID'], pca_df], axis=1)
    final_df.head()
    return final_df

def get_tsne_df_and_score(tsne_setting, df: pd.DataFrame, dimension: int, supervised: bool = False, normalized : bool = True, cluster_metrics : str = 'cdbw'):
    ''' Execute the t-sne, dimensional reduction and return a dataframe
    input:
        tsne_setting: scikit learn PCA setting.
        df         : pandas dataframe that got from load_embedding_and_label_df function.
        dimension  : int, number of dimension of the embedded space.
        supervised : boolean, if True, provide label for t-SNE training.
        normalized : boolean, if True, perform min max normalization.

    output:
        Trained_tsne: trained t-SNE model
        tsne_df: pandas dataframe that contain the coordinate in emedded sapce. 
        Column name: {filename, frameID, state, pc1, pc2, ....{depends on number of dimension you have gave}}
        sscore : clustering score
    '''
    # Recommand: TSNE(n_components=dimension, early_exaggeration=250,learning_rate='auto', init='pca', perplexity=300,n_iter=1000)
    cols = []
    for n_comps in range(0, dimension):
        cols.append('PC' + str(n_comps + 1)) # To plot the scatter plot easier, so we name the result also as 'PC1' ..

    if supervised:
        Trained_tsne = tsne_setting.fit(np.array(df['embedding'].tolist()),df['state'])
    else:
        Trained_tsne = tsne_setting.fit(np.array(df['embedding'].tolist()))

    tsne_result = Trained_tsne.fit_transform(np.array(df['embedding'].tolist()))

    if normalized:
        embedding = MinMaxScaler().fit_transform(tsne_result)

    tsne_df = pd.DataFrame(data=embedding, columns=cols)
    tsne_df = pd.concat([df['state'], df['filename'], df['frameID'], tsne_df], axis=1)
    sscore = evaluate_clusters(tsne_df[(tsne_df['state']!='23') & (tsne_df['state']!='0')], cluster_metrics, dim=2,label='state',)
    return Trained_tsne, tsne_df, sscore

def get_UMAP_df_and_score(umap_setting, df: pd.DataFrame, dimension: int, supervised: bool = False, normalized : bool = True, anonymous_mode: bool  = False, cluster_metrics: str = 'cdbw'):
    '''Execute the UAMP, dimensional reduction and return a dataframe
    Input:
        1. umap_setting: scikit learn UMAP setting 
        2. df: pandas dataframe that got from load_embedding_and_label_df function.
        3. supervised: bool, if Ture, provide label for t-SNE training.
        4. normalized: bool, variable, if True, perform min max normalization.
        5. anonymous mode: str, some of our input dataframe do not have filename (If load the embedding data using load_embedding_PlainCSV)
                           frameID but only have state and embeddings, set the anonymous_mode = True to read only state and embeddings from df

    output:
        Trained_tsne: trained UMAP model
        UMAP_df: pandas dataframe that contain the coordinate in emedded sapce. 
                 Column name: {filename, frameID, state, pc1, pc2, ....{depends on number of dimension you have gave}}
        sscore : float, CDbw score
    '''
    if supervised:
        Trained_umap = umap_setting.fit(df['embedding'].tolist(), df['state'])

    else:
        Trained_umap = umap_setting.fit(df['embedding'].tolist())

    embedding = Trained_umap.transform(df['embedding'].tolist())

    if normalized:
        embedding = MinMaxScaler().fit_transform(embedding)

    cols = []
    for n_comps in range(0, dimension):
        cols.append('PC' + str(n_comps + 1))
    
    umap_df = pd.DataFrame(data=embedding, columns=cols)

    if not anonymous_mode:
        final_df = pd.concat([df['state'], df['filename'], df['frameID'], umap_df], axis=1)

    else:
        final_df = pd.concat([df['state'],umap_df],axis=1)

    # Here we only calculate the clustering of "pre-defined assembly state". Shouldn't included error state and intermediate state. 
    try:
        sscore = evaluate_clusters(final_df[(final_df['state']!='23') & (final_df['state']!='0')], dim=2,label='state',metrics = cluster_metrics)
    # sscore = evaluate_clusters(final_df[(final_df['state']!='23')],dim=2,label='state')
    except:
        sscore = -1
        print("Skip clustering score")

    return Trained_umap, final_df, sscore

def evaluate_clusters(df : pd.DataFrame ,dim : int, label: str, metrics: str):
    '''Ex: evaluate_clusters(pca_df_blackout_with_head[pca_df_blackout_with_head['state']!=22])
    input: 
        df : pandas dataframe, that got from load_embedding_and_label_df function.
        dim: int, number of dimension of the embedded space.
        label: str, column name in the df, which is the label of data. 
    
    output:
        score: float, CDbw socre
    '''
    X = []
    if dim == 2:
        for idx in range(len(df)):
            X.append([df.iloc[idx]['PC1'],df.iloc[idx]['PC2']])
    elif dim== 3:
        for idx in range(len(df)):
            X.append([df.iloc[idx]['PC1'],df.iloc[idx]['PC2'],df.iloc[idx]['PC3']])
    else:
        raise NotImplementedError
    
    # Calculate the CDbw, 
    # Remapping the label since the "developer from cdbw" did not consdier str or non-consecutive integer label...
    label_list = []
    map = df[label].unique().tolist()

    # Replace the label with unique list index (to be consecutive interger number list )
    for idx, row in df.iterrows():
        label_list.append(int(map.index(row[label])))
    
    # Calculate the clustering score
    if metrics == 'cdbw':
        compact, cohesion, separation, score = CDbw(np.array(X),np.array(label_list),'euclidean',multipliers = True)
        print('Compact:{:.4f}, cohension:{:.4f}, separation:{:.4f}'.format(compact, cohesion, separation))
        print('cdbw score:{:.4f}'.format(score))
    
    elif metrics == 'DBCV':
        # DBCV is expensive to calcuated
        dbcv_score = dbcv(np.array(X),np.array(label_list))
        print('DBCV score:{:.4f}'.format(dbcv_score))

    else:
        raise NotImplementedError
    
    print('-'*50)
    return score

def fit_UMAP_transform(UMAP_model, df: pd.DataFrame, normalized :bool = True, anonymous_mode : bool = False, metrics : str = 'cdbw'):
    ''' Given the trained UMAP model, transform new data on that embedding space. 
    Input: 
        UMAP_model: the model that is already being trained.
        df        : pandas DataFrame, that store high dimensional embedding 
        normalized: bool, if true normalized the value of data after UMAP transformation
        anonymous_mode: bool, some of the df do not contain 'filename', (if load the data by "load_df_PainCSV")
        metrics   : str, either 'cdbw' or 'DWCV'
    Return:
        final_df  : pandas dataframe that contain the 2D coordinate
        sscore    : double, score of the clustering
    Return:
    '''
    embedding = UMAP_model.transform(df['embedding'].tolist())
    if normalized:
        embedding = MinMaxScaler().fit_transform(embedding)
    
    umap_df = pd.DataFrame(data=embedding, columns=['PC1','PC2'])
    if not anonymous_mode:
        final_df = pd.concat([df['state'], df['filename'], df['frameID'], umap_df], axis=1)
        sscore = evaluate_clusters(final_df[(final_df['state']!='23') & (final_df['state']!='0')],metrics = 'cdbw', dim=2,label='state',)

    else:
        final_df = pd.concat([df['state'], umap_df], axis=1)
        sscore = 0

    return final_df, sscore
#- END of the dimensional redunction function

#- START of the plotting function
def create_embeddings_video(temporal_feature: pd.DataFrame, output_path: str) -> None:
    """Generate a video from all images in the specified folder.

    input:
        temporal_feauture: dataframe, filtered by specific filename
        output_path      : output directory

    output: None
    """
    ## Example:
    ## pca_setting = PCA(n_components=dimension, svd_solver='full')
    ## df = get_PCA_df(embeddings, state_labels, frameIDs, filenames, 2, pca_setting)
    ## created_temporal_feature = df[df["filename"].isin(['17_assy_1_5'])].reset_index()
    ## create_embeddings_video(created_temporal_feature, './17_assy_1_5')

    # Plot setting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('2 component PCA for single recording', fontsize=20)
    ax.grid()
    ax.set_xlim([0, 1.0])
    ax.set_ylim([-0.1, 1.0])
    canvas = FigureCanvas(fig)

    # Build video writer
    video_filename = output_path + '.mp4'
    width, height = fig.get_size_inches() * fig.dpi
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(width), int(height)))

    # Plot
    for i in range(0, len(temporal_feature)):
        row = temporal_feature.loc[i]
        # Plot the dash line
        if i == 0:
            prev_x = row['PC1']
            prev_y = row['PC2']
        else:
            prev_x = temporal_feature.loc[i - 1]['PC1']
            prev_y = temporal_feature.loc[i - 1]['PC2']
        # Plot figure
   
        ax.scatter(row['PC1'], row['PC2']
                   , c=colors[int(row['state'])]
                   , s=50, label=row['state'], marker=markers[int(row['state'])], alpha=0.8, edgecolors='black',
                   linewidth=0.5)

        ax.set_title('FrameID:{}, State{}'.format(temporal_feature.loc[i]['frameID'], temporal_feature.loc[i]['state']),
                     fontsize=20)
        # Update the video writer
        fig.canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert RGB to BGR (native of opencv)
        buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        out.write(buf)

    out.release()
    plt.close()
    print('Convertion done!')

def confidence_ellipse(x: ArrayLike, y : ArrayLike, ax: matplotlib.axes.Axes, n_std: float =3.0, facecolor: str ='none', **kwargs):
    """ Plot the ellipse given the (x,y) axis, but not using anymore
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    input: 
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

    output:
        matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    # if mean_x is not np.nan:
    #     print('MeanX:{:.3f},StdX:{:.3f}\nMeanY:{:.3f}, StdY:{:.3f}\n'.format(mean_x,scale_x,mean_y,scale_y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
def plot_2D(df: pd.DataFrame, title: str = '', save: bool = False, path: str = '.') -> None:
    ''' Plot 2d embeddings
        df   : pandas dataframe in this format: {'state', 'PC1', 'PC2'}
        title: str, string that you want to put on the graph
        save : bool, if true, will store the plot in the corresponding path
        path : str,  path that the plot would be stored at 
    '''
    # Visualization 2D project
    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('UMAP feature 1', fontsize=15)
    ax.set_ylabel('UMAP feature 2', fontsize=15)
    ax.set_title('{}'.format(title), fontsize=20)

    targets = [str(idx) for idx in range(0, 23)] # Change this if you want to visualize state 0
    for target, color, marker in zip(targets, colors, markers):
        indicesToKeep = df['state'] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1']
                   , df.loc[indicesToKeep, 'PC2']
                   , c=color
                   , s=50, marker=marker, alpha=0.4, edgecolors='black', linewidth=0.5)

    ax.legend(targets, loc='best', ncol=2, frameon=True)
    ax.set_xlim([-0.25, 1.25])
    ax.set_ylim([-0.25, 1.25])
    ax.grid()
    if save:
        plt.savefig('{}/{}.svg'.format(path,title), dpi=500)

def plot_2D_with_images(df : pd.DataFrame, img_path: str, title: str = 'demo',save:bool = False,path:str = './', mode:str = 'real') -> None:
    ''' Given dataframe (after dimensionality reduction), 
        plot scatter plot.
    '''    
    # From dataframe, extract information from ['filename'], ['frameID'], and ['state'] form a list
    # Base on the list find the by 'state' (so We can integrate with same logic flow.)

    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('{}'.format(title), fontsize=  10)

    targets = [str(idx) for idx in range(0, 24)]

    for idx in df.index:
        # if idx%1000 == 0:
        #     print(idx)

        if mode == 'real':
            img_name = img_path + df.loc[idx,'filename'] + '_' + str(df.loc[idx,'frameID']).zfill(6) + '_' + str(df.loc[idx,'state']).zfill(2) + '.png'
        
        elif mode == 'synth':
            img_name = img_path + df.loc[idx,'filename'].zfill(8)+'.jpg'

        else:
            raise NotImplementedError
        
        ab = AnnotationBbox(OffsetImage(plt.imread(img_name),zoom=0.003),(df.loc[idx,'PC1'],df.loc[idx,'PC2']),frameon=False)
        ax.add_artist(ab)


    if save:
        plt.savefig('{}/{}.png'.format(path,title),bbox_inches='tight',dpi=3000)
        plt.close(fig)

def analyze_variation(df: pd.DataFrame, save: bool = False, path : str ='./')-> None:
    """Analyze the variation between 'recording' for data point from same states. The data points would be group 
    """
    num_of_recording = len(pd.unique(df["filename"]))
    targets = [str(idx) for idx in range(0, 24)]
    colors = (sns.color_palette('hls', num_of_recording).as_hex())
    markers = ['o', '^', 'v', '<', '>', 's',
               'D', 'd', 'p', 'h', 'H', '8',
               'X', '*', '.', 'P', 'x', '+',
               '1', '2', '3', '4', '|', '.',
               'o', '^', 'v']
    recorded_file = pd.unique(df["filename"])
    # Fiter dataframe by state
    for state in targets:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_xlabel('PC1', fontsize=15)
        ax.set_ylabel('PC2', fontsize=15)
        ax.set_xlim([-0.25, 1.25])
        ax.set_ylim([-0.25, 1.25])
        
        ax.grid()
        feature_per_state = df[df["state"].isin([state])]
        # If there is no data in this state, then skip
        if len(feature_per_state) == 0:
            # print("No instances in this state. skip state{}".format(state))
            plt.close()
            continue
        target_label = []
        recorded_file = pd.unique(feature_per_state['filename'])
        for idx, target in enumerate(recorded_file):
            indicesToKeep = feature_per_state['filename'] == target
            ax.scatter(feature_per_state.loc[indicesToKeep, 'PC1']
                       , feature_per_state.loc[indicesToKeep, 'PC2']
                       , c=colors[idx]
                       , s=50, label=target, marker=markers[idx], alpha=0.8, edgecolors='black', linewidth=0.5)

            target_label.append(target)
        ax.legend(labels=target_label, loc='best', ncol=2, frameon=True)
        # score = evaluate_clusters(feature_per_state,dim=2,label='filename')
        ax.set_title('Same assembly state{}, different recording.'.format(state), fontsize=24)
        for idx, target in enumerate(recorded_file):
            indicesToKeep = feature_per_state['filename'] == target
            confidence_ellipse(feature_per_state.loc[indicesToKeep, 'PC1'], feature_per_state.loc[indicesToKeep, 'PC2'],
                               ax, facecolor=colors[idx], edgecolor='black', alpha=0.3, zorder=0, label=None)
        if save:
            plt.savefig('{}/file_variation_state{}.png'.format(path, state), dpi=300)
        plt.close()

def plot_2D_4_publication(df: pd.DataFrame, title: bool = '', save: bool = False, path: str = '.') -> None:
    """Store without legend, grid, and x,y label and ticks. (Can uncommnt to show them). Save the figure in .png and .pdf format
    """
    colorlist = ['grey','cornflowerblue','green','darkorange',
                 'black','darkgoldenrod','turquoise','blue',
                 'brown','olivedrab','deepskyblue','mediumvioletred',
                 'saddlebrown','lawngreen','slategray','crimson',
                 'lightcoral','darkviolet','teal','yellowgreen',
                 'orangered','slateblue','olive','red']
    
    markers = ['+', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', 'x']
    # markers.reverse()
    # colors.reverse()
    # Visualization 2D project
    fig = plt.figure(figsize=(8, 6), dpi=300)
    
    ax = fig.add_subplot(1, 1, 1)

    targets = natsorted(np.unique(df['state']).astype(str).tolist())
    
    targets.reverse()
    for state in targets:
        indicesToKeep = df['state'] == state
        color = colorlist[int(state)]
        marker = markers[int(state)]
        if state == '0':
            sz = 10
            alpha_set = 0.1
        else:
            sz = 30
            alpha_set = 0.2
        ax.scatter(df.loc[indicesToKeep, 'PC1']
                   , df.loc[indicesToKeep, 'PC2']
                   , c = color
                   , s=sz, marker=marker, alpha=alpha_set, linewidth=0.5)


    # Give the legend before adding the confident interval otherwise the ellipse would also be a legned in the plot.
    ax.set_xticks([])
    ax.set_yticks([])
    if save:
        plt.savefig('{}/{}.pdf'.format(path,title), dpi=3000)
        plt.savefig('{}/{}.png'.format(path,title)) # For easier to check the result in the folder without open it up :)

def plot_2D_4_publication_PSR(df: pd.DataFrame, title: bool = '', save: bool = False, path: str = '.') -> None:
    """Store without legend, grid, and x,y label and ticks. (Can uncommnt to show them). Save the figure in .png and .pdf format
    """
    colorlist = ['grey','cornflowerblue','green','darkorange',
                 'black','darkgoldenrod','turquoise','blue',
                 'brown','olivedrab','deepskyblue','mediumvioletred',
                 'saddlebrown','lawngreen','slategray','crimson',
                 'lightcoral','darkviolet','teal','yellowgreen',
                 'orangered','slateblue','olive','red']
    markers = ['+', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', 'x']

    # Visualization 2D project
    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    targets = natsorted(np.unique(df['state']).astype(str).tolist())
    
    targets.reverse()
    for state in targets:
        indicesToKeep = df['state'] == state
        color = colorlist[int(state)]
        marker = markers[int(state)]
        if state == '0':
            sz = 10
            alpha_set = 0.1
        else:
            sz = 30
            alpha_set = 0.2
        ax.scatter(df.loc[indicesToKeep, 'PC1']
                   , df.loc[indicesToKeep, 'PC2']
                   , c = color
                   , s=sz, marker=marker, alpha=alpha_set, linewidth=0.5)

    # Give the legend before adding the confident interval otherwise the ellipse would also be a legned in the plot.
    ax.set_xticks([])
    ax.set_yticks([])
    if save:
        plt.savefig('{}/{}.pdf'.format(path,title), dpi=3000)
        plt.savefig('{}/{}.png'.format(path,title)) # For easier to check the result in the folder without open it up :)


def subsampled_df(df: pd.DataFrame, shrink_size : int = 4, state_tobe_subsample = '0') -> pd.DataFrame :
    """ There are too many inter-mediate state data point in the dataframe, so temporally downsampling the inter-state.
    """
    #shrink the intermediate state by factor of 4. 
    # Pharse intermediate state, then subsampling by 4. 
    non_zero_umap_df = df[df['state']!=state_tobe_subsample].reset_index()
    zero_umap_df     = df[df['state']==state_tobe_subsample].reset_index()
    zero_umap_df_sub = zero_umap_df.iloc[[idx for idx in range(0,len(zero_umap_df)-1, shrink_size)],:]
    new_df = pd.concat([non_zero_umap_df, zero_umap_df_sub], axis = 0)
    return new_df

def plot_all_state(df: pd.DataFrame, save: bool = False, path: str = '.',title: str ='') -> pd.DataFrame:
    """ Plot several subplot which only shows data points from same recordings.
    """
    targets = [str(idx) for idx in range(0, 24)]
    colorlist = ['grey','cornflowerblue','green','darkorange',
                 'black','darkgoldenrod','turquoise','blue',
                 'brown','olivedrab','deepskyblue','mediumvioletred',
                 'saddlebrown','lawngreen','slategray','crimson',
                 'lightcoral','darkviolet','teal','yellowgreen',
                 'orangered','slateblue','olive','red']
    markers = ['+', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', 'x']
    fig, axs = plt.subplots(4, 6, figsize=(15, 15), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=.5, wspace=.001)
    fig.tight_layout()
    for ax, target, color, marker in zip(axs.flat, targets, colorlist, markers):
        indicesToKeep = df['state'] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1']
                   , df.loc[indicesToKeep, 'PC2']
                   , c=color
                   , s=50
                   , marker=marker, label=target, alpha=0.8, edgecolors='black', linewidth=0.5)

        ax.legend(loc='best', frameon=True)
        cov_mtx = np.cov(df.loc[indicesToKeep, 'PC1'],df.loc[indicesToKeep, 'PC2'])
        ax.set_title('StdX:{:.3f}, Stdy:{:.3f}'.format(cov_mtx[0,0],cov_mtx[1,1]))
        ax.grid()
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.0])
    if save:
        plt.savefig('{}/{}.pdf'.format(path,title), dpi=300)

def load_img(img_path: str, bbox: list):
    ''' load and crop the images
    '''
    ##- Example to call the function:
    # load_dir =   # todo: path to your json
    # with open(load_dir / "image_paths.json") as fp:
    #     annots = json.load(fp)

    # for i in range(len(annots['images'])):
    #     img_path = Path(annots['images'][i])
    #     img = load_img(img_path, annots['bbox'][i])
    #     img.show()

    img = Image.open(img_path).convert("RGB")
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    center_x = x + int(w / 2)
    center_y = y + int(h / 2)

    side_length = max(w, h)

    x1 = center_x - int(side_length / 2)
    y1 = center_y - int(side_length / 2)

    x2 = center_x + int(side_length / 2)
    y2 = center_y + int(side_length / 2)

    # handle out of bound cases
    img_width = img.size[0]
    img_height = img.size[1]
    if side_length >= img_height:
        y1 = 0
        y2 = img_height
        x1 = center_x - int(img_height / 2)
        x2 = center_x + int(img_height / 2)
    elif y1 < 0:
        new_center_y = center_y + abs(y1)
        y1 = new_center_y - int(side_length / 2)
        y2 = new_center_y + int(side_length / 2)
    elif y2 > img_height:
        new_center_y = center_y - abs(img_height - y2)
        y1 = new_center_y - int(side_length / 2)
        y2 = new_center_y + int(side_length / 2)
    elif x1 < 0:
        new_center_x = center_x + abs(x1)
        x1 = new_center_x - int(side_length / 2)
        x2 = new_center_x + int(side_length / 2)
    elif x2 > img_width:
        # note: we are currently not handling (nor encountering) rare cases where x2 > img_width and y needs to be
        #  adjusted as well. keep in mind.
        new_center_x = center_x - abs(img_width - x2)
        x1 = new_center_x - int(side_length / 2)
        x2 = new_center_x + int(side_length / 2)

    assert 0 <= x1 <= img_width, f"x1 is outside of image range: {x1}. Image: {img_path}"
    assert 0 <= x2 <= img_width, f"x2 is outside of image range: {x2}. Image: {img_path}"
    assert 0 <= y1 <= img_height, f"y1 is outside of image range: {y1}. Image: {img_path}"
    assert 0 <= y2 <= img_height, f"y2 is outside of image range: {y2}. Image: {img_path}"

    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img = cropped_img.resize((800, 800))
    r,g,b = cropped_img.split()
    r_factor = 1.2
    g_factor = 1.0
    enhanced_img = Image.merge("RGB", (r.point(lambda i: i * r_factor),
                                       g.point(lambda i: i * g_factor),
                                       b))
    return cropped_img

def Plot_2d_with_croped_images(df: pd.DataFrame, img_annots,title: str = 'demo',save : bool = False,path : str = './', mode: str  = 'real') -> None:
    ''' Given dataframe (after dimensionality reduction), plot the image (cropped by bounding box) scatter on the corresponding 2D embedding space.
    '''    
    # From dataframe, extract information from ['filename'], ['frameID'], and ['state'] form a list
    # Base on the list find the by 'state' (so We can integrate with same logic flow.)
    root_2_remove = (r'\\asml.com\eu')
    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    targets = [str(idx) for idx in range(0, 24)]

    for idx in df.index:
        if idx%1000 == 0:
            print(idx)

        if mode == 'real':
            # img_name = img_path + df.loc[idx,'filename'] + '_' + str(df.loc[idx,'frameID']).zfill(6) + '_' + str(df.loc[idx,'state']).zfill(2) + '.png'
            img_path = Path((Path(img_annots['images'][idx]).as_posix()).replace(root_2_remove,"")).as_posix()
            croped_img = load_img(img_path.replace("\\","/"), img_annots['bbox'][idx])
        elif mode == 'synth':
            img_name = img_annots + df.loc[idx,'filename'].zfill(8)+'.jpg'

        ab = AnnotationBbox(OffsetImage(croped_img,zoom=0.003),(df.loc[idx,'PC1'],df.loc[idx,'PC2']),frameon=False)
        ax.add_artist(ab)


    if save:
        plt.savefig('{}/{}.png'.format(path,title),bbox_inches='tight',dpi=5000)
        plt.close(fig)


def plot_2D_intra_video(df: pd.DataFrame, title:str  = '' , save: bool = False, path: str ='.') -> None:
    """ Plot embeddings from several subplot from each recording"""
    colors = ['grey','cornflowerblue','green','darkorange',
                 'black','darkgoldenrod','turquoise','blue',
                 'brown','olivedrab','deepskyblue','mediumvioletred',
                 'saddlebrown','lawngreen','slategray','crimson',
                 'lightcoral','darkviolet','teal','yellowgreen',
                 'orangered','slateblue','olive','red']

    markers = ['+', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', 'x']
    fig, axs = plt.subplots(math.ceil(len(pd.unique(df['filename']))/ 3) , 3, figsize=(20, 25), sharex=True, sharey=True)
    suptitle = 'Transition of state {} to state {}\n\n'.format(df['state'].iloc[0],df['state'].iloc[-1])

    fig.suptitle(suptitle)
    fig.subplots_adjust(hspace=.8, wspace=.001)

    recordings = df['filename'].unique()
    for file_ID, (ax, filename) in enumerate(zip(axs.flat,df['filename'].unique())):
        file_df = df[df['filename'] == filename]
        for state_ID, state in enumerate(pd.unique(file_df['state'])):
            indicesToKeep = file_df['state'] == state
            if state == '0':
                sz = 80
            else:
                sz = 100
            ax.scatter(file_df.loc[indicesToKeep,'PC1'],
                       file_df.loc[indicesToKeep,'PC2'],
                       c = colors[int(state)],
                       marker = markers[int(state)],
                       s = sz, alpha= 0.4, edgecolors= 'black',linewidths= 0.5,label = state)
        ax.set_title(filename + ", # of frames:"+str(len(file_df)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        # if file_ID> 5:
            # break
    plt.tight_layout()
    if save:
        plt.savefig('{}/{}.png'.format(path,title,bbox_inches = 'tight'),dpi = 1000)
        plt.savefig('{}/{}.pdf'.format(path,title,bbox_inches = 'tight'),dpi = 1000)

  
def plot_2D_heatmap_progress(df: pd.DataFrame, title :str = '' , save:bool = False, path:str = '.')-> None:
    """ Plot embeddings between states in regardless recording. 
    Inter-mediate state is visualized by progress (precentage of progress,headmap). 
    """
    colors = ['grey','cornflowerblue','green','darkorange',
                 'black','darkgoldenrod','turquoise','blue',
                 'brown','olivedrab','deepskyblue','mediumvioletred',
                 'saddlebrown','lawngreen','slategray','crimson',
                 'lightcoral','darkviolet','teal','yellowgreen',
                 'orangered','slateblue','olive','red']

    markers = ['+', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', '.', \
                '.', '.', '.', '.', '.', 'x']
    state_markers = ['.', '^', 'v', '<', '>', 's', \
                'D', 'd', 'p', 'h', 'H', '8', \
                '>', '*', '.', 'P', 'x', '+', \
                 '1', '2', '3', '4', '|', 'X']

    fig = plt.figure(figsize=(8, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    recordings = df['filename'].unique()
    for file_ID, (filename) in enumerate((df['filename'].unique())):
        file_df = df[df['filename'] == filename].reset_index()

        # if file_ID == 10:
        #     break 
        targets = natsorted(pd.unique(file_df['state']).astype(str).tolist())
        targets.reverse()
        for state_ID, state in enumerate(targets):
            indicesToKeep = file_df['state'] == state
            if state == '0':
                sz = 20
                im = ax.scatter(file_df.loc[indicesToKeep,'PC1'],
                       file_df.loc[indicesToKeep,'PC2'],
                       c = (file_df.loc[indicesToKeep].index.astype(int)-np.min(file_df.loc[indicesToKeep].index.astype(int))) / (np.max(file_df.loc[indicesToKeep].index.astype(int))-np.min(file_df.loc[indicesToKeep].index.astype(int)))*100,
                    #    marker = markers[int(state)],
                        marker= markers[int(state)],
                       s = sz, alpha= 0.5, edgecolors= 'black',linewidths= 0.5,cmap= 'winter')
                
            else:
                sz = 60
                ax.scatter(file_df.loc[indicesToKeep,'PC1'],
                        file_df.loc[indicesToKeep,'PC2'],
                        # c = (file_df.loc[indicesToKeep].index.astype(int)) / np.max(file_df.loc[indicesToKeep].index.astype(int)),
                        c = colors[int(state)],
                        marker = markers[int(state)],
                        s = sz, alpha= 1, edgecolors= 'black',linewidths= 0.5,label = state,cmap= 'winter')

    cbar = fig.colorbar(im, orientation = 'vertical',pad=0.01)
    cbar.set_label(label  = 'Normalized progress (%)',size = 'large',weight='bold')
    cbar.ax.tick_params(axis='both',labelsize='large')
    plt.tight_layout()
    if save:
        plt.savefig('{}/{}.png'.format(path,title,bbox_inches = 'tight'),dpi = 1000)
        plt.savefig('{}/{}.pdf'.format(path,title,bbox_inches = 'tight'),dpi = 1000)
# END of the plotting function 