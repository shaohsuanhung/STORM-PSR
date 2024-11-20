import json
import torch
import numpy as np
import os
import time
import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report, \
    average_precision_score
import pytorch_metric_learning.utils.accuracy_calculator as accuracy_calculator

from models import ContrastiveModel


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_dict(log_path: Path, txt_name, dict_name):
    with open(log_path / txt_name, 'w') as file:
        file.write(json.dumps(dict_name))


def cosine_similarity(a, b):
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    return np.dot(a, b) / (mag_a * mag_b)


def get_rng(seed=1234):
    """ Gets us a consistent RNG for reproducibility """
    rng = np.random.default_rng(seed=seed)
    return rng


def load_run_model(run_path: Path, model_weights: Path):
    with open(run_path / 'args.txt') as file:
        args_raw = json.load(file)
        print(f"Loaded run parameters from {run_path / 'args.txt'}")

    args = Namespace(use_pretrained_weights=False, model=args_raw["model"], run_name=args_raw["run_name"],
                     loss=args_raw["loss"], hidden=args_raw["hidden"],
                     workers=args_raw["workers"], channels=args_raw["channels"])
    if args.loss == "ce":
        model = ContrastiveModel(args, weights_dir=None, classifier=True)
    else:
        model = ContrastiveModel(args, weights_dir=None)
    if model_weights is not None:
        model.load_weights_encoder(model_weights)
        print(f"Succesfully loaded model from {model_weights}")
    else:
        print(f"WARNING -- NOT LOADING ANY WEIGHTS!")
    return model


def batch_log_print(d, i, t):
    c = datetime.datetime.now().strftime('%H:%M:%S')
    str = f"{c} - Iteration:{i} \t Loss:{d['loss']:.6f}"
    for key in d:
        str += f" \t {key}: {d[key]:.5f}"
    str += f" \t {time.time() - t:.3f} seconds/iteration"
    print(str)


def compute_embeddings(loader, model, synthetic=False):
    """ adapted from https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/utils.py#L207 """
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    if synthetic:
        total_embeddings = np.zeros((len(loader.dataset) * loader.dataset.batch_size, model.embed_dim))
        total_targets = np.zeros(len(loader.dataset) * loader.dataset.batch_size)
    else:
        total_embeddings = np.zeros((len(loader)*loader.batch_size, model.embed_dim))
        total_targets = np.zeros(len(loader)*loader.batch_size)

    model.float().to(DEVICE)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            if synthetic:
                images = images.squeeze(0)
                targets = targets.squeeze(0)
            images = images.to(DEVICE)
            bsz = targets.shape[0]
            embed = model(images)
            total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
            total_targets[idx * bsz: (idx + 1) * bsz] = targets.detach().numpy()

            del images, targets, embed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.float32(total_embeddings), total_targets.astype(int)


def get_error_embeddings(loader, model):
    """
    adapted from https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/utils.py#L207
    differs from compute_embeddings since we dont have labels (targets) but intended states and error categories.
    """
    model.eval()
    model.use_projection_head(False)
    model.float().to(DEVICE)

    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    n = len(loader) * loader.batch_size
    total_embeddings = np.zeros((n, model.embed_dim))
    total_int_states = np.zeros(n)
    total_error_cats = np.zeros(n)

    stds = []
    means = []
    with torch.no_grad():
        t1 = time.time()
        for idx, (images, int_states, error_cats) in enumerate(loader):
            std, mean = torch.std_mean(images)
            stds.append(std)
            means.append(mean)
            images = images.to(DEVICE)
            bsz = int_states.shape[0]
            embed = model(images)
            total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
            total_int_states[idx * bsz: (idx + 1) * bsz] = int_states.detach().numpy()
            total_error_cats[idx * bsz: (idx + 1) * bsz] = error_cats.detach().numpy()

            del images, int_states, error_cats, embed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        t2 = time.time()
        t = t2-t1
        s_per_two = t/n * 2
        fps = 1 / (s_per_two)
        print(f"Time to evaluate {n} samples: {t:.0f} seconds ({s_per_two:.4f} sec per batch of 2 - {fps:.1f} FPS)")
    print(f"Batch mean+-std = {sum(means)/len(means):.3f}+-{sum(stds)/len(stds):.3f}")
    return np.float32(total_embeddings), total_int_states.astype(int), total_error_cats.astype(int)


def binary_verification_for_errors(rng, error_embeds, int_states, error_cats, embeds, labels, anchor_embeds,
                                   anchor_labels, dist="l2"):
    error_embeds = np.around(error_embeds, decimals=5)
    embeds = np.around(embeds, decimals=5)
    anchor_embeds = np.around(anchor_embeds, decimals=5)
    similarities = np.zeros(len(int_states) * 2, dtype=np.float32)
    targets = np.zeros(len(int_states) * 2, dtype=int)
    errors = np.zeros(len(int_states) * 2, dtype=int)

    c = 0
    for i in range(int_states.shape[0]):
        error_cat = error_cats[i]
        intended = int_states[i]

        error_embed = error_embeds[i, :]

        # Similarity with anchor:
        idx_anchor = np.where(anchor_labels == intended)[0][0]
        anchor_class = anchor_labels[idx_anchor]
        assert anchor_class == intended, f"Classes should be equal but are {anchor_class} and {intended}"
        anchor_embed = anchor_embeds[idx_anchor, :]
        if dist == "l2":
            anchor_l2 = np.linalg.norm(error_embed - anchor_embed)
            similarities[c] = 1 - min(anchor_l2, 1)
        else:
            sim = cosine_similarity(error_embed, anchor_embed)
            similarities[c] = sim
        targets[c] = 0  # target is 0, because anchor is not same as our error state embedding!
        errors[c] = error_cat
        c += 1


        idx_correct = rng.choice(np.where(labels == intended)[0])
        correct_class = labels[idx_correct]
        assert anchor_class == correct_class, f"Classes should be equal but are {anchor_class} and {correct_class}"
        correct_embed = embeds[idx_correct, :]
        if dist == "l2":
            correct_l2 = np.linalg.norm(correct_embed - anchor_embed)
            similarities[c] = 1 - min(correct_l2, 1)
        else:
            sim = cosine_similarity(correct_embed, anchor_embed)
            similarities[c] = sim
        targets[c] = 1  # target is 1, because anchor is  same as our error state embedding!
        errors[c] = error_cat
        c += 1

    mAP = average_precision_score(targets, similarities, average='weighted')
    ROC = roc_auc_score(targets, similarities, average='weighted')

    score_per_category = {
        "1": {"mAP": 0, "ROC": 0, "name": "Missing", "n": 0},
        "2": {"mAP": 0, "ROC": 0, "name": "Placement", "n": 0},
        "3": {"mAP": 0, "ROC": 0, "name": "Orientation", "n": 0},
        "4": {"mAP": 0, "ROC": 0, "name": "Part-level", "n": 0},
    }
    for error_cat in score_per_category:
        error_cat = int(error_cat)
        mask = errors == error_cat
        sims_ = similarities.copy()[mask]
        tars_ = targets.copy()[mask]
        mAP_ = average_precision_score(tars_, sims_, average='weighted')
        ROC_ = roc_auc_score(tars_, sims_, average='weighted')
        score_per_category[str(error_cat)]["mAP"] = mAP_
        score_per_category[str(error_cat)]["ROC"] = ROC_
        score_per_category[str(error_cat)]["n"] = len(sims_)

    return mAP, ROC, score_per_category


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def get_recording_list(folder: Path, train=False, val=False, test=False) -> list:
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
        recordings.append([Path(f.path) for f in os.scandir(folder / set) if f.is_dir()])
    return flatten_list(recordings)


class YourCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 5, self.avg_of_avgs,
                                                  self.return_per_class, self.label_comparison_fn)

    def calculate_f1_at_1(self, knn_labels, query_labels, **kwargs):
        top1_predictions = knn_labels[:, 0]
        gt = query_labels.clone().cpu().detach().numpy()
        pred = top1_predictions.clone().cpu().detach().numpy()
        precision, recall, fscore, _ = precision_recall_fscore_support(gt, pred,
                                                                       average='macro', zero_division=0.0)
        print(f"Precision: {precision:.3f} \t recall: {recall:.3f} \t fscore: {fscore:.3f} \t ")
        return fscore

    def requires_knn(self):
        return super().requires_knn() + ["calculate_precision_at_5", "calculate_f1_at_1"]


def get_acc_dict(calc, query, query_labels, reference, reference_labels):
    acc_dict = calc.get_accuracy(
        query=query,
        query_labels=query_labels,
        reference=reference,
        reference_labels=reference_labels,
        ref_includes_query=False)
    return acc_dict
