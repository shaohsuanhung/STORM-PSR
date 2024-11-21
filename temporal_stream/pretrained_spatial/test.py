import argparse
import torch
import torch.utils.data
import numpy as np
import time
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
from pytorch_metric_learning.utils.inference import CustomKNN

from datasets import RealContrastiveDatasetWithInters, AdditionalGeneralizationDatasetContrastive
from utils import compute_embeddings, load_run_model, DEVICE, YourCalculator, get_acc_dict
from temporal_aware_CL.temporal_aware_datasets import RealContrastiveDatasetWithInters_PSR

def test_metric(model, data_dir, psr_load_path , split="test", w=224, h=224, dist="cos", skip_factor=4, state_category = None, args = None):
    model.use_projection_head(False)
    model.eval()
    model.to(DEVICE)

    print(f"Loading test dataset...")
    t = time.time()
    # test_dataset = RealContrastiveDatasetWithInters_PSR(data_dir / split, w=w, h=h, skip_factor=skip_factor)
    test_dataset = RealContrastiveDatasetWithInters_PSR(data_dir, psr_load_path=psr_load_path, split = split, skip_factor= 10, state_category = state_category, args = args)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)
    print(f"Loaded test dataset. Time taken: {time.time() - t:.2f} seconds")

    print(f"Loading training dataset...")
    t = time.time()
    # train_dataset = RealContrastiveDatasetWithInters_PSR(data_dir / "train", w=w, h=h, skip_factor=skip_factor)
    train_dataset = RealContrastiveDatasetWithInters_PSR(data_dir, psr_load_path=psr_load_path, split = 'train', skip_factor= 10,state_category = state_category, args= args)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=True)
    print(f"Loaded training dataset. Time taken: {time.time() - t:.2f} seconds")

    test_embeddings, test_targets = compute_embeddings(test_loader, model)
    print(f"Computed test embeddings. Time taken: {time.time() - t:.2f} seconds")

    print(f"Computing training embeddings...")
    t = time.time()
    train_embeddings, train_targets = compute_embeddings(train_loader, model)
    print(f"Computed training embeddings. Time taken: {time.time() - t:.2f} seconds")

    print(f"Computing performance...")
    t = time.time()
    if dist == "cos":
        distance_fn = CosineSimilarity()
    elif dist == "l2":
        distance_fn = LpDistance(normalize_embeddings=False)
    else:
        raise ValueError(f"Distance {dist} not known")

    custom_knn = CustomKNN(distance_fn, batch_size=64)
    calculator = YourCalculator(k="max_bin_count", avg_of_avgs=False, knn_func=custom_knn,
                                include=("precision_at_1", "mean_average_precision_at_r", "f1_at_1"))
    acc_dict = get_acc_dict(calculator, test_embeddings, test_targets, train_embeddings, train_targets)
    del calculator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_calculator = YourCalculator(k="max_bin_count", avg_of_avgs=True, knn_func=custom_knn, return_per_class=False,
                                    include=("precision_at_1", "mean_average_precision_at_r"))
    avg_acc_dict = get_acc_dict(avg_calculator, test_embeddings, test_targets, train_embeddings, train_targets)
    del avg_calculator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_class_calculator = YourCalculator(k="max_bin_count", avg_of_avgs=False, knn_func=custom_knn,
                                          return_per_class=True, include=("precision_at_1", "mean_average_precision_at_r", ))
    map_at_r_dict = get_acc_dict(per_class_calculator, test_embeddings, test_targets, train_embeddings, train_targets)
    del per_class_calculator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Per-class PR@1: {map_at_r_dict['precision_at_1']}")
    map_at_r_list = map_at_r_dict['mean_average_precision_at_r']
    map_at_r_inters = map_at_r_list[0]
    map_at_r_states = sum(map_at_r_list[1:]) / len(map_at_r_list[1:])
    avg_acc_dict["mean_average_precision_at_r_zero"] = map_at_r_inters
    avg_acc_dict["mean_average_precision_at_r_nonzero"] = map_at_r_states
    print(f"Computed performance. Time taken: {time.time() - t:.2f} seconds")
    print(acc_dict)
    print(f"-----------------------------------------------------------------")

    return acc_dict, avg_acc_dict


def test_metric_generalization(model, data_dir, w=224, h=224, dist="cos"):
    model.use_projection_head(False)
    model.eval()
    model.to(DEVICE)

    print(f"Loading test dataset...")
    t = time.time()
    test_dataset = AdditionalGeneralizationDatasetContrastive(data_dir / "synth", "query", w=w, h=h)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=True)
    print(f"Loaded test dataset. Time taken: {time.time() - t:.2f} seconds")

    print(f"Loading training dataset...")
    t = time.time()
    train_dataset = AdditionalGeneralizationDatasetContrastive(data_dir / "synth", "reference", w=w, h=h)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=True)
    print(f"Loaded training dataset. Time taken: {time.time() - t:.2f} seconds")

    test_embeddings, test_targets = compute_embeddings(test_loader, model)
    print(f"Computed test embeddings. Time taken: {time.time() - t:.2f} seconds")

    print(f"Computing training embeddings...")
    t = time.time()
    train_embeddings, train_targets = compute_embeddings(train_loader, model)
    print(f"Computed training embeddings. Time taken: {time.time() - t:.2f} seconds")

    print(f"Computing performance...")
    t = time.time()
    if dist == "cos":
        distance_fn = CosineSimilarity()
    elif dist == "l2":
        distance_fn = LpDistance(normalize_embeddings=False)
    else:
        raise ValueError(f"Distance {dist} not known")

    custom_knn = CustomKNN(distance_fn, batch_size=32)
    calculator = YourCalculator(k="max_bin_count", avg_of_avgs=False, knn_func=custom_knn,
                                include=("precision_at_1", "mean_average_precision_at_r", "f1_at_1"))
    calculator_class = YourCalculator(k="max_bin_count", avg_of_avgs=False, knn_func=custom_knn, return_per_class=True,
                                      include=("precision_at_1", "mean_average_precision_at_r"))
    print(f"Output unique train labels: {np.unique(train_targets, return_counts=True)}")
    print(f"Output unique test labels: {np.unique(test_targets, return_counts=True)}")
    acc_dict = get_acc_dict(calculator, test_embeddings, test_targets, train_embeddings, train_targets)
    print(calculator)
    acc_dict_class = get_acc_dict(calculator_class, test_embeddings, test_targets, train_embeddings, train_targets)

    print(f"Computed performance. Time taken: {time.time() - t:.2f} seconds")
    print(acc_dict)
    print(acc_dict_class)
    print(f"-----------------------------------------------------------------")
    return acc_dict


# Test the features
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str, help='Path to the run directory, e.g. ./runs/run_name')
    parser.add_argument("--checkpoint", type=str, default=None, help='Name of the checkpoint to be tested')
    parser.add_argument("--dist", type=str, default="cos", help='l2 or cos (cosine) distance. default = cos')
    parser.add_argument("--data_path", type=str, default="./data", help='Location of the IndustReal data dir')
    parser.add_argument("--w", type=int, default=224,
                        help='width and height of the image inputs to the model.')
    parser.add_argument("--h", type=int, default=224,
                        help='width and height of the image inputs to the model.')
    args = parser.parse_args()

    if args.checkpoint is None:
        checkpoint = "best.pth"
    else:
        checkpoint = args.checkpoint

    run_path = Path(args.run_path)
    data_path = Path(args.data_path)
    print(f"Testing with intermediate states in test set.")
    name = run_path.name
    print(f"-----------------------------------------------------------------")
    print(f"-----------------------------------------------------------------")
    print(f"RUNNING {name}")

    print(f"Running {checkpoint}")
    model_weight_path = run_path / "checkpoints" / checkpoint
    save_dir = run_path / checkpoint
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model...")
    t = time.time()
    model = load_run_model(run_path, model_weight_path)

    # a skip factor of 4 is applied to all tests to fit the KNN algorithm on a single GPU. This means that for each
    # background image taken, the next three are skipped. This is not done to non-background images.
    acc_dict, avg_acc_dict = test_metric(model, data_path, split="test", skip_factor=4, args = args)

    acc_dict_gen = test_metric_generalization(model, data_path)

    log_str = f"{checkpoint} \t " \
              f"mAP@R: {acc_dict['mean_average_precision_at_r'] * 100:.2f} \t " \
              f"PR@1: {acc_dict['precision_at_1'] * 100:.2f} \t " \
              f"F1@1: {acc_dict['f1_at_1'] * 100:.2f} \t " \
              f"mAP@R(-): {avg_acc_dict['mean_average_precision_at_r_zero'] * 100:.2f} \t " \
              f"mAP@R(+): {avg_acc_dict['mean_average_precision_at_r_nonzero'] * 100:.2f} \t " \
              f"mAP@R(gen): {acc_dict_gen['mean_average_precision_at_r'] * 100:.2f} \t " \
              f"PR@1(gen): {acc_dict_gen['precision_at_1'] * 100:.2f} \t " \
              f"F1@1(gen): {acc_dict_gen['f1_at_1'] * 100:.2f} \n"
    print(f"{name} - {checkpoint}")
    print(log_str)
    file = open(run_path / 'test_log.txt', 'a')
    file.write(log_str)
    file.close()

    print(f"Computed performance. Time taken: {time.time() - t:.2f} seconds")
    print(f"---------------------------------------------")
