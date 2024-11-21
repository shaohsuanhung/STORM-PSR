"""
Dataset for trainning the spatial encoder in a weakly supervised setting using Key-frame sampling (KFS)
"""
#%%
import torch
import torch.utils.data
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.functional as f
from PIL import Image
import json
import numpy as np
import random
from pathlib import Path
import os
import csv
import glob
import pandas as pd
import collections
from natsort import natsorted

class TemporalAwareContrastiveDataset(torch.utils.data.Dataset):
    """
    Implementation of Key-frame sampling (KFS).
    """
    def __init__(self,args,psr_load_path, rec_path, split = "train", state_category = None, error_state:bool = True):
        assert split == "train", "Currently only using this dataset class for training!"
        self.real_img_dir = Path(rec_path)
        self.dir = Path(args.syn_path)
        self.n_iters = args.n_iters
        self.n_classes = args.n_classes
        self.rng = np.random.default_rng(seed=args.seed)

        #-- Build real image pool from temopra-aware sampler (sampling several frame after the step completion moment), and get load path
        self.df_state_imgs_from_psr,self.df_bg_imgs_from_psr = KFS_sampling(Path(rec_path), Path(psr_load_path), split, N_frame2sample=args.n_frames, error_state=error_state,rng=self.rng, state_categories=state_category)
        self.real_annotations = dict({'labels':[],'images':[]})
        self.real_bg_annotations = dict({'labels':[],'images':[]})
        self.real_annotations['labels'] = np.array(self.df_state_imgs_from_psr['state'])
        self.real_annotations['images'] = np.array(self.df_state_imgs_from_psr['path'])
        self.real_bg_annotations['labels'] = np.array(self.df_bg_imgs_from_psr['state'])
        self.real_bg_annotations['images'] = np.array(self.df_bg_imgs_from_psr['path'])

        #-- Load Synthetic iamge pool if available
        if args.n_synth != 0:
            self.synth_img_dir = self.dir / "synth" / "images"
            with open(self.dir / "synth" / "labels.json") as f:
                synth_annotations = json.load(f)
            synth_annotations['labels'] = np.array(synth_annotations['labels'])

            #-- Setting all background images to state 0 -- this excludes the additional generalization test set from training
            #TODO: the error_state would change depends on the definition of different dataset.
            error_state = 22
            synth_annotations['labels'][synth_annotations['labels'] > error_state] = 0

            #-- Intermediate/background states are all at end of dataset. so just find 1st background and remove rest.
            first_bg_idx = np.where(synth_annotations['labels'] == 0)[0][0]
            synth_annotations["images"] = synth_annotations["images"][:first_bg_idx]
            synth_annotations["labels"] = synth_annotations["labels"][:first_bg_idx]
            synth_annotations["bbox"] = synth_annotations["bbox"][:first_bg_idx]

            
            #-- Randomly shuffle data (seed fixed = labels keep matching)
            random.Random(args.seed).shuffle(synth_annotations["images"])
            random.Random(args.seed).shuffle(synth_annotations["labels"])
            random.Random(args.seed).shuffle(synth_annotations["bbox"])
            synth_transforms = get_transform(train=True, synth=True, args=args)
            
        else:
            synth_annotations = None
            synth_transforms = None

        self.synth_annotations = synth_annotations
        self.real_transforms = get_transform(train=True, synth=False, args=args)
        self.synth_transforms = synth_transforms

        self.w = args.img_w
        self.h = args.img_h

        self.channels = args.channels
        self.resize_to = (self.w, self.h)

        #-- Batch creation data
        self.n_real = args.n_real
        self.n_synth = args.n_synth
        self.n_bg = args.n_bg
        self.rng = np.random.default_rng(seed=args.seed)
        
        if self.n_real == 0 and self.n_bg > 0:
            raise ValueError(f"You shouldn't have 0 real images and simultaneously load real background images!")
        #-- We sample unique IDs from synthetic set (since some images are only present in synth world)
        if self.n_synth == 0:
            print(f"Warning - training without any synthetic images!")
            self.unique_ids = np.unique(self.real_annotations["labels"])
            self.unique_ids = self.unique_ids[self.unique_ids > 0]
            self.unique_ids = self.unique_ids[self.unique_ids < 12]
        else:
            self.unique_ids = np.unique(self.synth_annotations["labels"])
        n_unique_classes = np.unique(self.unique_ids).size
        if n_unique_classes < args.n_classes:
            print(f"There are not enough unique classes ({n_unique_classes}) to fit in desired n_classes "
                  f"{args.n_classes} --> manually reducing n_classes to {n_unique_classes}")
            self.n_classes = n_unique_classes
        else:
            self.n_classes = args.n_classes
        print(f"Unique classes for training: {self.unique_ids}")
        self.batch_size = int(self.n_classes * self.n_real) + int(self.n_classes * self.n_synth) + self.n_bg
        print(f"# images per batch: {self.batch_size} --> "
              f"{self.n_classes} classes x {self.n_real} real images + "
              f"{self.n_classes} classes x {self.n_synth} synthetic images + {self.n_bg} error states")
        

    def __getitem__(self,index):
        images = torch.empty(self.batch_size, self.channels, self.h, self.w)
        labels = torch.empty(self.batch_size, dtype=torch.float)
        labels_array = np.empty(self.batch_size,dtype=int)
        c = 0

        #-- sample the classes to include in batch
        sampled_classes = self.rng.choice(self.unique_ids, self.n_classes, replace=False)

        #-- sample images per class
        for class_id in sampled_classes:
            # get real images
            class_idxes = np.where(self.real_annotations['labels'] == class_id)[0]

            # if we don't find any, simply don't load real positives, load 2x synt positives (if n_syn !=0)
            # else if n_syn == 0, then skip this class
            if len(class_idxes) == 0:
                # print(f"Skip class:{class_id}.")
                real_positives_exist = False
            else:
                real_positives_exist = True
                sampled_idxes = self.rng.choice(class_idxes, self.n_real, replace=False)
                for sampled_idx in sampled_idxes:
                    label = self.real_annotations['labels'][sampled_idx]
                    image_path = self.real_annotations["images"][sampled_idx]
                    img = get_image(image_path, size=self.resize_to)

                    images[c, :, :, :] = self.real_transforms(img)
                    labels[c] = label
                    labels_array[c] = label
                    c += 1

            #-- get synthetic images. Sample also n_real if there were no real images for this class
            if self.n_synth!=0:
                class_idxes = np.where(self.synth_annotations['labels'] == class_id)[0]
                if real_positives_exist:
                    sampled_idxes = self.rng.choice(class_idxes, self.n_synth, replace=False)
                else:
                    sampled_idxes = self.rng.choice(class_idxes, self.n_synth + self.n_real, replace=False)

                for sampled_idx in sampled_idxes:
                    label = self.synth_annotations['labels'][sampled_idx]
                    image_path = self.synth_img_dir / self.synth_annotations["images"][sampled_idx]
                    img = get_image(image_path, size=self.resize_to)
                    images[c, :, :, :] = self.synth_transforms(img)
                    labels[c] = label
                    labels_array[c] = label
                    c += 1
            else:
                continue

        #-- sample the n_bg background images to include in batch
        if self.n_bg != 0:
            bg_idxes = np.where(self.real_bg_annotations['labels'] == 0)[0]
            sampled_idxes = self.rng.choice(bg_idxes, self.n_bg, replace=False)
            for sampled_idx in sampled_idxes:
                label = self.real_bg_annotations['labels'][sampled_idx]
                image_path = self.real_bg_annotations["images"][sampled_idx]
                img = get_image(image_path, size=self.resize_to)
                images[c, :, :, :] = self.real_transforms(img)
                labels[c] = label
                labels_array[c] = label
                c += 1

        return images, labels

    def __len__(self):
        return self.n_iters

class RealContrastiveDatasetWithInters_PSR(torch.utils.data.Dataset):
    def __init__(self, dir, psr_load_path, split = "train", state_category = None,w=224, h=224, skip_factor=10, only_clean=False, args = None):
        # For recording under dir:
        self.image_dir = dir / split
        
        #   glob every frame -> sort this 
        #   The append to the image_dir_list
        df_state_imgs_from_psr, df_hard_bg_from_psr = KFS_sampling(rec_path = Path(dir), psr_load_path = Path(psr_load_path), N_frame2sample= 20,split = split, state_categories = state_category, error_state= False)
        df = pd.concat([df_state_imgs_from_psr, df_hard_bg_from_psr], ignore_index= True)
        self.annotations = dict({'labels':[],'images':[]})
        self.annotations['images'] = np.array(df['path'])
        self.annotations['labels'] = np.array(df['state'])
        zeros = np.where(self.annotations['labels'] > 0)[0][::skip_factor]
        non_zeros = np.where((self.annotations['labels'] > 0) & (self.annotations['labels'] != len(state_category)-1))[0]
        self.indexes = np.concatenate((zeros, non_zeros))
        # Keep the following line
        self.transforms = get_transform(train=False, synth=False, args = args)

        self.w = w
        self.h = h
        self.channels = 3
        self.resize_to = (self.w, self.h)
        # print(f"Load {len(self.indexes)} images from {split} set.")

    def __getitem__(self, i):
        index = self.indexes[i]
        image_path = self.image_dir / self.annotations["images"][index]
        label = self.annotations['labels'][index]

        img = get_image(image_path, size=self.resize_to)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.indexes)

class testtime_dataset(torch.utils.data.Dataset):
    # To boost the computing speed by multiple subprocess (num_workers)
    def __init__(self,rec_path,split, preprocess = None):
        #-1. Recording list under the split 
        self.preprocess = preprocess
        if split == 'train':
            recordings = get_recording_list(rec_path, train=True)
        
        elif split == 'val':
            recordings = get_recording_list(rec_path, val = True)
        
        elif split =='test':
            recordings = get_recording_list(rec_path, test = True)
            
        #2. Gathering recording df
        if 'industreal' in str(rec_path).lower():
            modality = 'rgb'
            recordings = [rec / modality for rec in recordings]

        filename_list = list()
        frame_list = list()
        frameID = list()
        for vid in recordings:
            paths = natsorted(list((vid).glob("*.jpg")))
            frame_list.extend(paths)
            if 'industreal' in str(rec_path).lower():
                filename_list.extend([paths[0].parents[1].name for _ in range(len(paths))])
            elif 'meccano' in str(rec_path).lower():
                filename_list.extend([paths[0].parents[0].name for _ in range(len(paths))])
            else:
                raise NotImplementedError(f"Currently only support industreal and meccano, but get {rec_path}")
            
            frameID.extend([img.name for img in paths])

        self.df = pd.DataFrame(data={"FileName":filename_list, "FrameID":frameID,"Path":frame_list})
        self.prev_vid_name = self.df.loc[0]['FileName']
        self.num_frames = len(self.df)

    def __getitem__(self,idx): 
        """Do not randomize this dataset when building the datalodaer, since we want to extract embedding by order."""
        #-- Uncomment for sanity check
        # print(self.df.loc[idx]['Path'])
        img = get_image(self.df.loc[idx]['Path'])
    
        if self.preprocess is not None:
            img = self.preprocess(img)
            
        return img

    def __len__(self):
        return self.num_frames

def get_transform(train=False, synth=False, args=None):
    if args == None:
        raise ValueError(f"Should pass args to 'get_transform' function in the dataset.")
    
    #TODO: Need more implementation here if we have more datasets. 
    if synth:
        if 'industreal' in str(args.data_path).lower():
        # mean and std data for the IndustReal synthetic training data
            mean = (0.838, 0.805, 0.761)
            std = (0.134, 0.143, 0.136)
        elif 'meccano' in str(args.data_path).lower():
        # Do not have Mean and std data for MECCANO 
            raise NotImplementedError("Currently there is not synthetic datat for MECCANO dataset.")
        else:
            raise NotImplementedError("Currently only support IndustReal and MECCANO dataset.")

    # Real-images 
    else:
        if 'industreal' in str(args.data_path).lower():
        # mean and std data for the real-world test data
            mean = (0.608, 0.545, 0.520)
            std = (0.172, 0.197, 0.188)

        elif 'meccano' in str(args.data_path).lower():
            mean = (0.4144,0.4014,0.3777)
            std = (0.2312,0.2458,0.2684)
        else:
            raise NotImplementedError("Currently only support IndustReal and MECCANO dataset.")
    
    #-- Data augmentations
    custom_transforms = []
    if train:
        if args is None:
            custom_transforms.append(v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.01, 2.0)))
            custom_transforms.append(v2.ColorJitter(brightness=0.1, saturation=0.7, contrast=0.1))
        else:
            custom_transforms.append(v2.GaussianBlur(kernel_size=args.kernel_size, sigma=(args.sigma_l, args.sigma_h)))
            custom_transforms.append(v2.ColorJitter(brightness=args.bright, saturation=args.sat, contrast=args.cont))
    custom_transforms.append(v2.Normalize(mean=mean, std=std))
    return torchvision.transforms.Compose(custom_transforms)

def get_image(image_path, size=(224, 224), show=False):
    img = Image.open(image_path).convert("RGB")

    if size is not None:
        img = img.resize(size)

    if show:
        img.show()
    return f.pil_to_tensor(img).float() / 255

def KFS_sampling(rec_path: Path, psr_load_path: Path, split: str, N_frame2sample:int =20,error_state:bool = True, rng = None, state_categories:list = None):
    """ This function implement the key-frame sampling (KFS)
    Selected several images that after the step completion moment, and we assume that these images contain the identical assembly state (thus it's weakly-supervised).
    """
    #-- For reproduciblilty
    if rng == None:
        rng = np.random.default_rng(seed=1234)

    if state_categories == None:
        print("Did not provided predefined assembly state. Using the ()IndustReal assembly state.")
        state_categories = ['background',
                    '10000000000',  # state 1
                    '10010010000',  # state 2
                    '10010100000',  # state 3
                    '10010110000',  # state 4
                    '11100000000',  # state 5
                    '11110010000',  # state 6
                    '11110100000',  # state 7
                    '11110110000',  # state 8
                    '11110111100',  # state 9
                    '11110111110',  # state 10
                    '11110110001',  # state 11
                    '11110111101',  # state 12
                    '11110111111',  # state 13
                    '11110101111',  # state 14
                    '11110011111',  # state 15
                    '11110011110',  # state 16
                    '11110101110',  # state 17
                    '11100001110',  # state 18
                    '11101101110',  # state 19
                    '11101011110',  # state 20
                    '11101111110',  # state 21
                    '11101111111',  # state 22
                    'error_state']
        
    #-- Get Key-frame informtaion from PSR label
    if split == 'train':
        recordings = get_recording_list(rec_path, train=True)
    
    elif split == 'val':
        recordings = get_recording_list(rec_path, val = True)

    elif split =='test':
        recordings = get_recording_list(rec_path, test = True)
    
    else:
        raise ValueError(f"Expect split to be 'train','val','test' but got {split}.")
    
    data_read = []
    lenght_of_frames = dict()
    for i, rec in enumerate(recordings):
        with open(psr_load_path /split / rec.name / "PSR_labels_raw.csv") as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            for i, row in enumerate(reader):
                frame = int(row[0][:-4])
                states = ((list(row[1:])))
                if error_state == True:
                    entry = {
                        "filename":rec.name,
                        "frame": frame,
                        "raw state label": states,
                        "state": find_matched_state(states,error_state = True, categories = state_categories) # Turn error_state to True to get only non-errror states. 
                    }
                else:
                    states = only_positive_states(states)
                    entry = {
                        "filename":rec.name,
                        "frame": frame,
                        "raw state label": states,
                        "state": find_matched_state(states,error_state = False, categories = state_categories) # Turn error_state to False to get non-errror states. 
                    }
                data_read.append(entry)
        #-- Get maximum number of the recording
        #TODO: Need more implementation here if we have more datasets. 
        if('industreal' in str(psr_load_path).lower()):
            lenght_of_frames.update({rec.name:len(glob.glob(os.path.join(rec / 'rgb','*.jpg')))})
        elif('meccano' in str(psr_load_path).lower()):
            lenght_of_frames.update({rec.name:len(glob.glob(os.path.join(rec,'*.jpg')))})    
        else:
            raise NotImplementedError(f"Expect indsutreal or meccano dataset. But get {psr_load_path}.")
    
    key_frame_df = pd.DataFrame(data=data_read,columns=['filename','frame','raw state label','state'])
    #-- Sampling state images from key-frames
    state_list = np.arange(len(state_categories))
    img_batch_state = list()
    hard_negative_batch = list()
    for state in state_list:
        df_state = key_frame_df[key_frame_df['state']==state].reset_index()
        if df_state.empty:
            continue
        sampled_moment = np.arange(len(df_state))
        for moment in sampled_moment:
            filename = df_state.loc[moment]['filename']
            frame    = df_state.loc[moment]['frame']
            #-- Selected several images that after the step completion moment, and we assume that these images contain the identical assembly state. 
            sampled_frame = np.arange(frame, min(int(lenght_of_frames[filename]), frame+N_frame2sample))

            #TODO: Need more implementation here if we have more datasets. 
            for frame in sampled_frame:
                if 'industreal' in str(psr_load_path).lower():
                    entry = {
                        'state':state,
                        'path':rec_path / split / filename /'rgb'/(str(frame.item()).zfill(6)+'.jpg')
                    }

                elif 'meccano' in str(psr_load_path).lower():
                   entry = {
                        'state':state,
                        'path':rec_path / split / filename / (str(frame.item()).zfill(5)+'.jpg')
                    }
                img_batch_state.append(entry)
            
            # If args.n_bg!=0, We give the hard negative background image to 12~14 sec before the step completion.
            #TODO: Need more implementation here if we have more datasets. 
            hard_negative_bg = np.arange(max(0,frame-7*N_frame2sample),max(0,frame-6*N_frame2sample))
            if hard_negative_bg is not None:
                for hard_frame in hard_negative_bg:
                    if 'industreal' in str(psr_load_path).lower():
                        hard_entry = {
                        'state':0,
                        'path':rec_path  / split / filename /'rgb'/(str(hard_frame.item()).zfill(6)+'.jpg')
                        }
                    elif 'meccano' in str(psr_load_path).lower():
                        hard_entry = {
                        'state':0,
                        'path':rec_path  / split / filename /(str(hard_frame.item()).zfill(5)+'.jpg')
                        }

                    hard_negative_batch.append(hard_entry)

    df_state_imgs_from_psr = pd.DataFrame(img_batch_state,columns=['state','path'])
    df_hard_bg_from_psr = pd.DataFrame(hard_negative_batch,columns=['state','path'])
    
    #-- Rule out Error state (len(categories)) and 0 (backgrond, intermediate state) for all datasets.
    df_state_imgs_from_psr = df_state_imgs_from_psr[(df_state_imgs_from_psr['state']!=int(len(state_categories)-1)) & (df_state_imgs_from_psr['state']!=0)]

    return df_state_imgs_from_psr,df_hard_bg_from_psr

def find_matched_state(state2matched, error_state:bool = False, categories:list = None):
    # The categories should be changed if using other dataset. 
    if (error_state) and ((('-1') in state2matched)):
        return int(len(categories))
    
    for idx, ele in enumerate(categories):
        if list_to_state_string(state2matched) == ele:
            return idx

        elif idx == (len(categories)-1):
            return 0
        
def list_to_state_string(state_list: list) -> str:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    state_string = str()
    for ele in state_list:
        state_string += ele
    return state_string  # From ['1','0', '1','1'] -> '1011'


def only_positive_states(states):
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
    return ['0' if num == '-1' else num for num in states]

def get_recording_list(folder: Path, train=False, val=False, test=False) -> list:
    """From https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/psr_utils.py"""
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
        recordings.append([Path(f.path)for f in os.scandir(folder / set) if f.is_dir()])
    recording_list = [item for sublist in recordings for item in sublist]
    
    return natsorted(recording_list)