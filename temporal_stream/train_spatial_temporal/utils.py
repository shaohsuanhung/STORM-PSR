from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import json
import os
import glob
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datatable as dt
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from argparse import Namespace
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report, average_precision_score
from torchvision.transforms import ToTensor
import io
import PIL
import torchvision.transforms.functional as f

def load_yaml(file: str):
    """
    Load yaml file
    """
    with open(file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = Namespace(**cfg)
    return cfg

def load_embedding_and_label_df(data_path, label_path: str = None, statics=False, normalized=False):
    # MECCANO version
    ''' Given the path of csv file, and flags, (1)load, (2) show statics and (3) standardization return dataframe
    input:
        path: str, path of csv file
        statics: boolean flag, print statics the if true
        normalized: boolean, normalized the embeddings.
    output:
        dataframe with 4 column: 
        {'filename', 'frameID','state','embedding'}
    '''
    # Read csv file under the given directory
    # filename can come from dir, frame ID can come fomr indices, state ->
    state_list = []
    frame_list = []
    name_list = []
    embeddings = []
    # Sort the label and data to make sure they are in the same  in alphabetical order by filename
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
        # -- Normalize: (x_i - mean) / std
        # print('Normalizing...')
        standardized_data = StandardScaler().fit_transform(embeddings)
        embeddings = []
        for data in standardized_data:
            embeddings.append(data.astype(np.float32))

    df = pd.DataFrame(data={'filename': name_list, 'frameID': frame_list,
                      'state': state_list, 'embedding': embeddings})
    if statics:
        print("Labels from data:{}\n".format(df['state'].unique()))
        # Count of recording that we collected
        print("Num of recorderings:{}".format(len(df['frameID'].unique())))
        print("Num of frame of each recording:{}".format(
            df['frameID'].value_counts()))
        # Count number of state of all recording
        print("State count:\n{}\n".format(df['state'].value_counts()))
        # Count change of state in each single video
        for recording in df['filename'].unique():
            states = df[df['filename'].isin([recording])]['state']
            print('{} state ({} samples) in recording: {}'.format(
                len(states.unique()), len(states), recording))
            print(states.value_counts())
            print('-' * 30)

    return df

def load_embedding_df(data_path, statics=False, normalized=False):
    ''' Given the path of csv file, and flags, (1)load, (2) show statics and (3) standardization 
    :input:
    '''
    # embedding_data = pd.read_csv(data_path,engine = 'python')
    embedding_data = dt.fread(data_path).to_pandas()
    # Extract features
    embeddings = embedding_data['embedding'].tolist()
    state_labels = embedding_data['state'].astype(str)
    filenames = embedding_data['filename'].astype(str)
    frameIDs = embedding_data['frameID'].astype(int)

    # embeddings = [literal_eval(embed) for embed in embeddings]
    #
    for i, embed in enumerate(embeddings):
        embed = embed.split('[')[-1].split(']')[0]
        row_ele = [float(ele) for ele in embed.split(',')]
        embeddings[i] = row_ele
    # Normalize the embedding
    if normalized:
        # -- Normalize: (x_i - mean) / std
        # print('Normalizing...')
        standardized_data = StandardScaler().fit_transform(embeddings)
        embeddings = []
        for std_embedding in standardized_data:
            embeddings.append(std_embedding.astype(np.float32))

    d = {'filename': filenames, 'frameID': frameIDs,
         'state': state_labels, 'embedding': embeddings}
    df = pd.DataFrame(data=d)
    if statics:
        print("Labels from data:{}\n".format(state_labels.unique()))
        # Count of recording that we collected
        print("Num of recorderings:{}".format(len(filenames.unique())))
        print("Num of frame of each recording:{}".format(
            filenames.value_counts()))
        # Count number of state of all recording
        print("State count:\n{}\n".format(state_labels.value_counts()))
        # Count change of state in each single video
        for recording in df['filename'].unique():
            states = df[df['filename'].isin([recording])]['state']
            print('{} state ({} samples) in recording: {}'.format(
                len(states.unique()), len(states), recording))
            print(states.value_counts())
            print('-' * 30)
    return df

def get_recording_list( folder: Path, train=False, val=False, test=False) -> list:
    assert [train, val, test].count(True) < 2, f"You can currently only retrieve one set or all sets, not two. For " \
        f"all sets, simply do not specify a set."
    if train:
        sets = ['train']
    elif val:
        sets = ['val']
    elif test:
        sets = ['test']
    else:
        sets = ['train', 'val', 'test']
    recordings = []
    for set in sets:
        recordings.append([Path(f.path)
                           for f in os.scandir(folder / set) if f.is_dir()])
    recording_list = [item for sublist in recordings for item in sublist]
    return recording_list

def get_image(image_path, size=(224, 224), show=False):
    img = Image.open(image_path).convert("RGB")

    if size is not None:
        img = img.resize(size)

    if show:
        img.show()
    return f.pil_to_tensor(img).float() / 255

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def get_metrics(gt, pred, verbose=False, f_only=False):
    _, recall, fscore, _ = precision_recall_fscore_support(
        gt, pred, average='weighted', zero_division=0.0)
    _, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        gt, pred, average='macro', zero_division=0.0)

    if f_only:
        result = {
            "recall": float(recall),
            "fscore": float(fscore),
            "fscore_macro": float(fscore_macro)}
        return result

    classes = None
    if np.max(gt) > 1:
        gt_ = to_one_hot(gt)
        pred_ = to_one_hot(pred)

        # we must filter out places where ground truth has 0 true values since roc auc is undefined otherwise
        del_cols = []
        classes = []
        for i in range(gt_.shape[1]):
            non_zero = np.count_nonzero(gt_[:, i])
            if non_zero == 0:
                del_cols.append(i)
            else:
                classes.append(f"state {i}")
        print(
            f"Deleting classes {del_cols} because these states are not in test set!")
        gt_ = np.delete(gt_, del_cols, 1)
        pred_ = np.delete(pred_, del_cols, 1)
        gt = gt_
        pred = pred_

    ap = average_precision_score(gt, pred, average='weighted')
    roc_auc = roc_auc_score(gt, pred, average='weighted', multi_class='ovo')

    result = {
        "recall": float(recall),
        "fscore": float(fscore),
        "fscore_macro": float(fscore_macro),
        "ap": float(ap),
        "roc_auc": float(roc_auc),
    }

    if verbose:
        print(f"Recall (accuracy): \t\t {recall :.2f}")
        print(f"F-score: \t\t\t\t {fscore :.3f}")
        print(f"F-score macro: \t\t\t\t {fscore_macro :.3f}")
        print(f"Area under RoC curve: \t {roc_auc:.3f}")
        print(f"Average precision: \t\t {ap:.3f}")
        if classes is not None:
            for i, class_ in enumerate(classes):
                print(f"{i}\t{class_}")
        print(classification_report(gt, pred))
        print('-' * 69)
    else:
        return result

def plot_trainlog_result(epoch: int, log_dir: str, save_path: str):
    train_loss_list = list()
    train_microF1_list = list()
    val_loss_list = list()
    val_microF1_list = list()
    # log_dir = ("/shared/nl011006/res_ds_ml_restricted/shaohung/train_log/runs")
    # run_name = '0609-frame-64-g'
    for idx in range(1, epoch+1):
        with open(log_dir / f"train_progress_epoch{idx}.txt") as f:
            for line in f:
                ele = line.split(',')
                train_loss = float(re.sub('{"train_loss":', '', ele[0]))
                train_micro_F1 = float(re.sub('"train_accuracy":', '', ele[1]))
                val_loss = float(re.sub('"validation_loss":', '', ele[2]))
                val_micro_F1 = float(
                    re.sub('"validation_accuracy":', '', ele[3]))

                train_loss_list.append(train_loss)
                train_microF1_list.append(train_micro_F1)
                val_loss_list.append(val_loss)
                val_microF1_list.append(val_micro_F1)

    # mlp overfit on the moons dataset

    fig, axs = plt.subplots(2)
    fig.set_figheight(3)
    fig.set_figwidth(8)
    fig.subplots_adjust(top=2)
    axs[0].plot(range(1, epoch+1), train_microF1_list,
                marker='.', label='Train')
    axs[0].plot(range(1, epoch+1), val_microF1_list, marker='.', label='Val')
    axs[1].plot(range(1, epoch+1), train_loss_list, marker='.', label='Train')
    axs[1].plot(range(1, epoch+1), val_loss_list, marker='.', label='Val')
    axs[0].set_title('Macro F1 score w.r.t epoch')
    axs[1].grid()
    axs[0].grid()
    # axs[0].set_xticks([1, 5, 10, 15, 20, 25])
    # axs[1].set_xticks([1, 5, 10, 15, 20, 25])
    # axs[0].set_yticks([0.8, 0.85, 0.90, 0.95, 1])
    # axs[0].set_yticks([0.8,0.85, 0.90, 0.95, 1])
    # axs[0].set_ylim([0.8, 1])
    # axs[1].set_ylim([0, 0.6])
    # axs[0].set_xlim([1, 25])
    # axs[1].set_xlim([1, 25])
    axs[0].set_ylabel('macro F1 score')
    axs[1].set_ylabel('Binary cross entropy loss')
    axs[1].set_title('Binary cross entropy loss w.r.t. epoch')
    axs[0].set_title('macro F1 score w.r.t. epoch')

    axs[0].legend()
    axs[1].legend()
    plt.savefig(save_path / "train_log.png", dpi=300, bbox_inches='tight')
    plt.close()
    return

def createConfusionMatrix(loader, model):
    y_pred = []  # save predction
    y_true = []  # save ground truth

    # iterate over data
    for inputs, labels in loader:
        inputs = torch.autograd.Variable(src).cuda()
        labels = torch.autograd.Variable(target).cuda()
        # Check the number of labels...

        with torch.no_grad():
            output = model(inputs)  # Feed Network
            # output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            # for PSR multi-label classification
            output = torch.sigmoid(output)
            output = output.round().data.cpu().numpy()

        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    # constant for classes
    # classes = [(idx) for idx in range(24)]
    classes = [(idx) for idx in range(11)]

    # Build confusion matrix
    cf_matrix = confusion_matrix(
        y_true, y_pred, labels=classes, normalize='true')

    # print("y_true:{}",y_true)
    # print("y_pred:{}",y_pred)
    # print("cf_matrix:\n{}\n",cf_matrix)
    # print(type(cf_matrix))
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=[i for i in classes],
    #                      columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    return sn.heatmap(cf_matrix, annot=True).get_figure()

def createMutliLabelConfusionMatrix(loader, model):
    y_pred = []  # save predction
    y_true = []  # save ground truth

    # iterate over data
    for inputs, labels in loader:
        inputs = torch.autograd.Variable(src).cuda()
        labels = torch.autograd.Variable(target).cuda()
        # Check the number of labels...

        with torch.no_grad():
            output = model(inputs)  # Feed Network
            # output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            # for PSR multi-label classification
            output = torch.sigmoid(output)
            output = output.round().data.cpu().numpy()

        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mlcf_mtx = multilabel_confusion_matrix(y_true, y_pred)
    f, axes = plt.subplots(3, 4, figsize=(25, 15))
    cmap = "Blues"
    axes = axes.ravel()
    for i in range(11):
        # disp = ConfusionMatrixDisplay(confusion_matrix(y_true[:,i],
        #                                             y_pred[:,i]),
        #                             display_labels=[0, i])
        # disp.plot(ax=axes[i], values_format='.4g')
        # disp.ax_.set_title(f'class {i}')
        # if i<10:
        #     disp.ax_.set_xlabel('')
        # if i%5!=0:
        #     disp.ax_.set_ylabel('')
        # disp.im_.colorbar.remove()
        axes[i].set_title(i)
        disp = ConfusionMatrixDisplay(confusion_matrix=mlcf_mtx[i], display_labels=str(i)).plot(
            include_values=True, cmap=cmap, ax=axes[i], values_format='.3f')
        axes[i].xaxis.set_ticklabels(['', '', '', ''])
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', which='both',
                            bottom=False, top=False, labelbottom=False)
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    return image
