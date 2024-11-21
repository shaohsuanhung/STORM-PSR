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


class ContrastiveDataset(torch.utils.data.Dataset):
    """ IndustReal dataset class for contrastive learning.

    Note that this dataset class returns an entire batch, not a single sample.
    """
    def __init__(self, args, split="train"):
        assert split == "train", "Currently only using this dataset class for training!"
        self.dir = Path(args.data_path)
        self.n_iters = args.n_iters
        self.n_classes = args.n_classes
        # real images
        self.real_image_dir = self.dir / split / "images"
        self.real_label_file = self.dir / split / "labels.json"

        with open(self.real_label_file) as f:
            self.real_annotations = json.load(f)
        self.real_annotations['labels'] = np.array(self.real_annotations['labels'])
        self.real_annotations['visibility'] = np.array(self.real_annotations['visibility'])

        # exclude images with intermediate states and those that don't have matching visibility!
        imgs_no_error_no_intermediate = np.logical_and(self.real_annotations['labels'] > 0, self.real_annotations['labels'] < 23)
        idxes_with_state = np.where(imgs_no_error_no_intermediate)[0]
        idx_has_visibility_mask = self.real_annotations["visibility"][idxes_with_state] == 3
        self.real_images_with_states = idxes_with_state[idx_has_visibility_mask]  # images with visible states
        # synthetic images
        # loading the synthetic data
        self.synth_image_dir = self.dir / "synth" / "images"
        with open(self.dir / "synth" / "labels.json") as f:
            synth_annotations = json.load(f)
        synth_annotations['labels'] = np.array(synth_annotations['labels'])

        # setting all background images to state 0 -- this excludes the additional generalization test set from training
        synth_annotations['labels'][synth_annotations['labels'] > 22] = 0

        # intermediate/background states are all at end of dataset. so just find 1st background and remove rest.
        first_bg_idx = np.where(synth_annotations['labels'] == 0)[0][0]
        synth_annotations["images"] = synth_annotations["images"][:first_bg_idx]
        synth_annotations["labels"] = synth_annotations["labels"][:first_bg_idx]
        synth_annotations["bbox"] = synth_annotations["bbox"][:first_bg_idx]
        print(
            f"N samples: {len(synth_annotations['labels'])} \t Unique training classes after deleting intermediate "
            f"classes: {np.unique(synth_annotations['labels'])}")

        # randomly shuffle data (seed fixed = labels keep matching)
        random.Random(args.seed).shuffle(synth_annotations["images"])
        random.Random(args.seed).shuffle(synth_annotations["labels"])
        random.Random(args.seed).shuffle(synth_annotations["bbox"])

        self.synth_annotations = synth_annotations

        self.real_transforms = get_transform(train=True, synth=False, args=args)
        self.synth_transforms = get_transform(train=True, synth=True, args=args)

        self.w = args.img_w
        self.h = args.img_h

        self.channels = args.channels
        self.resize_to = (self.w, self.h)

        # batch creation data
        self.n_real = args.n_real
        self.n_synth = args.n_synth
        self.n_bg = args.n_bg

        if self.n_real == 0 and self.n_bg > 0:
            raise ValueError(f"You shouldn't have 0 real images and simultaneously load real background images!")

        # we sample unique IDs from synthetic set (since some images are only present in synth world)
        if self.n_synth == 0:
            print(f"Warning - training without any synthetic images!")
            self.unique_ids = np.unique(self.real_annotations["labels"])
            self.unique_ids = self.unique_ids[self.unique_ids > 0]
            self.unique_ids = self.unique_ids[self.unique_ids < 23]
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

        self.rng = np.random.default_rng(seed=args.seed)

    def __getitem__(self, index):
        images = torch.empty(self.batch_size, self.channels, self.h, self.w)
        labels = torch.empty(self.batch_size, dtype=torch.float)
        c = 0

        # sample the classes to include in batch
        sampled_classes = self.rng.choice(self.unique_ids, self.n_classes, replace=False)

        # sample images per class
        for class_id in sampled_classes:
            # get real images
            class_idxes = np.where(self.real_annotations['labels'] == class_id)[0]

            # if we don't find any, simply don't load real positives, load 2x synt positives
            if len(class_idxes) == 0:
                real_positives_exist = False
            else:
                real_positives_exist = True
                sampled_idxes = self.rng.choice(class_idxes, self.n_real, replace=False)
                for sampled_idx in sampled_idxes:
                    label = self.real_annotations['labels'][sampled_idx]
                    image_path = self.real_image_dir / self.real_annotations["images"][sampled_idx]
                    img = get_image(image_path, size=self.resize_to)
                    images[c, :, :, :] = self.real_transforms(img)
                    labels[c] = label
                    c += 1

            # get synthetic images. Sample also n_real if there were no real images for this class
            class_idxes = np.where(self.synth_annotations['labels'] == class_id)[0]
            if real_positives_exist:
                sampled_idxes = self.rng.choice(class_idxes, self.n_synth, replace=False)
            else:
                sampled_idxes = self.rng.choice(class_idxes, self.n_synth + self.n_real, replace=False)

            for sampled_idx in sampled_idxes:
                label = self.synth_annotations['labels'][sampled_idx]
                image_path = self.synth_image_dir / self.synth_annotations["images"][sampled_idx]
                img = get_image(image_path, size=self.resize_to)
                images[c, :, :, :] = self.synth_transforms(img)
                labels[c] = label
                c += 1

        # sample the n_bg background images to include in batch
        bg_idxes = np.where(self.real_annotations['labels'] == 0)[0]
        sampled_idxes = self.rng.choice(bg_idxes, self.n_bg, replace=False)
        for sampled_idx in sampled_idxes:
            label = self.real_annotations['labels'][sampled_idx]
            image_path = self.real_image_dir / self.real_annotations["images"][sampled_idx]
            img = get_image(image_path, size=self.resize_to)
            images[c, :, :, :] = self.real_transforms(img)
            labels[c] = label
            c += 1
        return images, labels

    def __len__(self):
        return self.n_iters


def filter_dict_by_idxes(d: dict, l: list) -> dict:
    return {key: [value[i] for i in l if i < len(value)] for key, value in d.items()}


class ErrorDataset(torch.utils.data.Dataset):
    """ Loads an image with error state, class of intended state, and the error category """
    def __init__(self, dir, only_clean=False, w=224, h=224):
        self.image_dir = dir / "images"
        self.label_file = dir / "labels.json"
        with open(self.label_file) as f:
            annotations = json.load(f)

        if only_clean:
            indexes_to_keep = [i for i, clean in enumerate(annotations["clean"]) if clean == 1]
            annotations = filter_dict_by_idxes(annotations, indexes_to_keep)

        self.annotations = annotations
        self.annotations['labels'] = np.array(self.annotations['labels'])
        print(f"Number of annotations: {len(self.annotations['labels'])}")
        self.annotations['clean'] = np.array(self.annotations['clean'])
        self.annotations['error_cat'] = np.array(self.annotations['error_cat'])
        self.annotations['intended'] = np.array(self.annotations['intended'])

        self.transforms = get_transform(train=False, synth=False)

        self.w = w
        self.h = h
        self.channels = 3
        self.resize_to = (self.w, self.h)

    def __getitem__(self, index):
        image_path = self.image_dir / self.annotations["images"][index]
        error_cat = self.annotations["error_cat"][index]
        int_state = self.annotations["intended"][index]

        img = get_image(image_path, size=self.resize_to)

        if self.transforms is not None:
            img = self.transforms(img)

        return img,  int_state, error_cat

    def __len__(self):
        return len(self.annotations["images"])


def get_image(image_path, size=(224, 224), show=False):
    img = Image.open(image_path).convert("RGB")

    if size is not None:
        img = img.resize(size)

    if show:
        img.show()
    return f.pil_to_tensor(img).float() / 255


def get_transform(train=False, synth=False, args=None):
    if synth:
        # mean and std data for the synthetic training data
        mean = (0.838, 0.805, 0.761)
        std = (0.134, 0.143, 0.136)
    else:
        # mean and std data for the real-world test data
        mean = (0.608, 0.545, 0.520)
        std = (0.172, 0.197, 0.188)

    # data augmentations
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


class RealContrastiveDatasetWithInters(torch.utils.data.Dataset):
    def __init__(self, dir, w=224, h=224, skip_factor=10, only_clean=False):
        self.image_dir = dir / "images"
        self.label_file = dir / "labels.json"
        with open(self.label_file) as f:
            annotations = json.load(f)

        self.annotations = annotations
        self.annotations['labels'] = np.array(self.annotations['labels'])
        self.annotations['visibility'] = np.array(self.annotations['visibility'])

        if dir.name in ["val", "test"]:
            zeros = np.where(self.annotations['labels'] == 0)[0][::skip_factor]
            if only_clean:
                non_zeros = np.where((self.annotations['labels'] > 0) &
                                 (self.annotations['labels'] != 23) &
                                 (self.annotations['visibility'] == 3))[0]
            else:
                non_zeros = np.where((self.annotations['labels'] > 0) & (self.annotations['labels'] != 23))[0]
            errors = np.where(self.annotations['labels'] == 23)[0]
            self.indexes = np.concatenate((zeros, errors, non_zeros))
        else:
            zeros = np.where(self.annotations['labels'] == 0)[0][::skip_factor]
            if only_clean:
                non_zeros = np.where((self.annotations['labels'] > 0) &
                                 (self.annotations['labels'] != 23) &
                                 (self.annotations['visibility'] == 3))[0]
            else:
                non_zeros = np.where((self.annotations['labels'] > 0) & (self.annotations['labels'] != 23))[0]
            self.indexes = np.concatenate((zeros, non_zeros))

        self.transforms = get_transform(train=False, synth=False)

        self.w = w
        self.h = h
        self.channels = 3
        self.resize_to = (self.w, self.h)

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

class RealContrastiveDatasetWithInters(torch.utils.data.Dataset):
    def __init__(self, dir, w=224, h=224, skip_factor=10, only_clean=False):
        self.image_dir = dir / "images"
        self.label_file = dir / "labels.json"
        with open(self.label_file) as f:
            annotations = json.load(f)

        self.annotations = annotations
        self.annotations['labels'] = np.array(self.annotations['labels'])
        self.annotations['visibility'] = np.array(self.annotations['visibility'])

        if dir.name in ["val", "test"]:
            zeros = np.where(self.annotations['labels'] == 0)[0][::skip_factor]
            if only_clean:
                non_zeros = np.where((self.annotations['labels'] > 0) &
                                 (self.annotations['labels'] != 23) &
                                 (self.annotations['visibility'] == 3))[0]
            else:
                non_zeros = np.where((self.annotations['labels'] > 0) & (self.annotations['labels'] != 23))[0]
            errors = np.where(self.annotations['labels'] == 23)[0]
            self.indexes = np.concatenate((zeros, errors, non_zeros))
        else:
            zeros = np.where(self.annotations['labels'] == 0)[0][::skip_factor]
            if only_clean:
                non_zeros = np.where((self.annotations['labels'] > 0) &
                                 (self.annotations['labels'] != 23) &
                                 (self.annotations['visibility'] == 3))[0]
            else:
                non_zeros = np.where((self.annotations['labels'] > 0) & (self.annotations['labels'] != 23))[0]
            self.indexes = np.concatenate((zeros, non_zeros))

        self.transforms = get_transform(train=False, synth=False)

        self.w = w
        self.h = h
        self.channels = 3
        self.resize_to = (self.w, self.h)

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
class AdditionalGeneralizationDatasetContrastive(torch.utils.data.Dataset):
    def __init__(self, dir, split, w=224, h=224):
        assert split in ["reference", "query"]
        self.image_dir = dir / "images"
        self.label_file = dir / "labels.json"
        with open(self.label_file) as f:
            annotations = json.load(f)

        self.annotations = annotations
        self.annotations['labels'] = np.array(self.annotations['labels'])

        classes = np.unique(self.annotations['labels'])
        n_samples_per_class = 100
        if split == "query":
            classes = classes[classes > 22]
            n_samples_per_class = int(n_samples_per_class/5)

        self.indexes = []
        for c in classes:
            if split == "reference":
                idxes = np.where(self.annotations['labels'] == c)[0][:n_samples_per_class]
            else:
                idxes = np.where(self.annotations['labels'] == c)[0][-n_samples_per_class:]  # grab from end to ensure new
            self.indexes.extend(list(idxes))

        self.transforms = get_transform(train=False, synth=True)

        self.w = w
        self.h = h
        self.channels = 3
        self.resize_to = (self.w, self.h)

    def __getitem__(self, i):
        index = self.indexes[i]
        image_path = self.image_dir / self.annotations["images"][index]
        label = self.annotations['labels'][index]

        img = get_image(image_path, size=self.resize_to)

        if self.transforms is not None:
            img = self.transforms(img)

        if label == 23:
            label = 0
        return img, label

    def __len__(self):
        return len(self.indexes)
