"""
Pre-train the spatial encoder
"""
import time
import datetime
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR, CyclicLR

from utils import DEVICE, save_dict, AverageMeter, get_rng, batch_log_print
from datasets import ContrastiveDataset
from losses import BatchAllTripletLoss, BatchHardTripletLoss, SupConLoss
from models import ContrastiveModel
from test import test_metric
#
from temporal_aware_CL.temporal_aware_datasets import TemporalAwareContrastiveDataset, get_recording_list, RealContrastiveDatasetWithInters_PSR

#-- Dataset variable Setting
MECCANO_categories = [
     'background',        
     '10001000100000000', # state 1
     '11001100100000000', # state 2
     '11001100111000000', # state 3
     '11101110111000000', # state 4
     '11111110111001000', # state 5
     '11111111111001000', # state 6
     '11111111111001001', # state 7
     '11111111111001101', # state 8
     '11111111111101101', # state 9
     '11111111111111101', # state 10
     '11111111111111111', # state 11
     'error_state'
]

IndustReal_categories = ['background',
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
def set_options():
    parser = argparse.ArgumentParser(description='Passing arguments.')
    # Run parameters
    parser.add_argument("run_name", type=str,
                        help='Name of the run to be tested or evaluated')
    parser.add_argument("--run_path", type=str, default="./runs",
                        help='Location of where to save run data. Default: ./runs/run_name')
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs to train the model')
    parser.add_argument("--model", type=str, default="vit_small_patch16_224.augreg_in21k_ft_in1k",
                        help='Model architecture to use. Currently available: resnet50.a1_in1k, resnet34.a1_in1k, '
                             'resnet18.a1_in1k, vit_small_patch16_224.augreg_in21k_ft_in1k')
    parser.add_argument("--loss", type=str, default="supcon",
                        help='Type of loss function to use. Implemented: batch_all, batch_hard, supcon, simclr, ce')
    parser.add_argument("--use_pretrained_weights", default=True, action='store_false',
                        help='Use pretrained weights instead of random initialization.')
    parser.add_argument("--hidden", type=int, default=128,
                        help='Number of hidden parameters in the projection layers.')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument("--scheduler", type=str, default="cosine_restart",
                        help='Type of loss function to use. Implemented: step, cosine_restart')
    parser.add_argument("--T_0", type=int, default=40,
                        help='T_0 value, indicating the number of epochs in first cycle of cos annealing warm restart')
    parser.add_argument("--lr_gamma", type=float, default=0.975,
                        help='Learning rate exponential decay factor. Gamma = 1 --> no decaying LR')
    parser.add_argument("--lr_step", type=int, default=10,
                        help='Learning rate step size.')
    parser.add_argument("--warmup", type=int, default=15,
                        help='Use warmup learning rate for this amount of epochs..')
    parser.add_argument("--n_iters", type=int, default=500,
                        help='Number of iterations per epoch. Needed because we randomly sample our batches.')
    parser.add_argument("--margin", type=float, default=0.01,
                        help='Margin used in the contrastive loss function.')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='Weight decay for the optimizer.')
    parser.add_argument("--temperature", type=float, default=0.07,
                        help='Temperature value for the supervised contrastive (supcon) loss.')
    parser.add_argument("--stop_after", type=int, default=40,
                        help='Limit for number of epochs without improvement before killing training.')
    # Data parameters
    parser.add_argument("--data_path", type=str, default="./data",
                        help='Location of the training data')
    parser.add_argument("--syn_path", type=str, default="./syn_data",
                        help='Location of the synthetic data')
    parser.add_argument("--psr_label_path",type=str,default='./label',
                        help='Location of the label of data')
    parser.add_argument("--n_classes", type=int, default=11,
                        help='Number of classes to sample per batch')
    parser.add_argument("--n_real", type=int, default=8,
                        help='Number of real-world images per class')
    parser.add_argument("--n_synth", type=int, default=8,
                        help='Number of synthetic images per class')
    parser.add_argument("--n_bg", type=int, default=16,
                        help='Number of background images to sample')
    parser.add_argument("--img_w", type=int, default=224,
                        help='width and height of the image inputs to the model.')
    parser.add_argument("--img_h", type=int, default=224,
                        help='width and height of the image inputs to the model.')
    parser.add_argument("--exclude_bg", default=False, action='store_true',
                        help='ISIL modification - Excludes the background (intermediate) from acting as positives in the loss.')
    parser.add_argument("--seed", type=int, default=1234,
                        help='Seed for splitting training data')
    parser.add_argument("--channels", type=int, default=3,
                        help='Input channels in data')
    parser.add_argument("--workers", type=int, default=8,
                        help='Number of workers for the dataloader')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='Batch size of the validation dataloader')
    parser.add_argument("--kernel_size", type=float, default=5,
                        help='Data augmentation - kernel of gaussian blur')
    parser.add_argument("--sigma_l", type=float, default=0.01,
                        help='Data augmentation - lower boundary for random gaussian blur')
    parser.add_argument("--sigma_h", type=float, default=2.0,
                        help='Data augmentation - upper boundary for random gaussian blur')
    parser.add_argument("--bright", type=float, default=0.1,
                        help='Data augmentation - brightness jitter max')
    parser.add_argument("--sat", type=float, default=0.7,
                        help='Data augmentation - saturation jitter max')
    parser.add_argument("--cont", type=float, default=0.1,
                        help='Data augmentation - contrast jitter max')
    parser.add_argument("--rotate", default=False, action='store_true',
                        help='Data augmentation - randomly rotate images 90 degree with p=0.5')
    parser.add_argument("--n_frames", type=float, default=20,
                        help='Number of frames to be sampled in the Key-frame sampling.')

    args = parser.parse_args()

    assert args.scheduler in ["cosine_restart", "step", "cyclic"], f"Scheduler {args.scheduler} not implemented.."
    assert Path(args.data_path).exists(), f"The path to data does not exist: {args.data_path}"

    return parser.parse_args()

def train_contrastive(loss_f, model, loader, optimizer, criterion, val=False, writer=None, it=None, n_views=10):
    if val:
        model.eval()
        print(f"Validating epoch...")
    else:
        model.train()
        print(f"Training epoch...")
    model.float().to(DEVICE)

    print(f'Total Iterations:{len(loader)}')

    running_loss = AverageMeter()
    running_active_triplets = AverageMeter()

    for i, (imgs, targets) in enumerate(loader):
        batch_dict = {}
        t1 = time.time()
        optimizer.zero_grad()
        imgs = imgs.squeeze(0)
        targets = targets.squeeze(0)

        imgs, targets = imgs.float().to(DEVICE), targets.long().to(DEVICE)

        embeddings = model(imgs)

        if loss_f == "supcon":
            bsz, n_features = embeddings.size()
            n_samples = bsz // n_views
            embeddings = embeddings.view(n_samples, n_views, n_features)
            targets = targets[::n_views]
            loss = criterion(embeddings, targets)
            batch_active_triplets = torch.Tensor([0])
        elif loss_f == "batch_all" or loss_f == "batch_hard":
            loss, batch_active_triplets = criterion(embeddings, targets)
        elif loss_f == "ce":
            loss = criterion(embeddings, targets)
            batch_active_triplets = torch.Tensor([0])
        else:
            raise ValueError(f"Loss {loss_f} unexpected!")

        if not val:
            loss.backward()
            optimizer.step()
            it += 1

        running_loss.update(loss.item())
        running_active_triplets.update(batch_active_triplets.item())
        batch_dict["loss"] = running_loss.val
        batch_dict["active_triplets"] = running_active_triplets.val

        del imgs, targets, embeddings
        torch.cuda.empty_cache()

        if not val:
            writer.add_scalar('Train loss', batch_dict["loss"], it)
            writer.add_scalar('Train active triplets', running_active_triplets.val, it)
        if i % 25 == 0:
            batch_log_print(batch_dict, i, t1)
    if val:
        return running_loss.avg
    else:
        return running_loss.avg, it


if __name__ == "__main__":
    #--- Set options
    args = set_options()
    print(args)
    #--- Check the args of dataset is correct
    #TODO: Need further implementation to run on other datasts
    if 'industreal' in str(args.data_path).lower():
        categories = IndustReal_categories

    elif 'meccano' in str(args.data_path).lower():
        categories = MECCANO_categories
        if args.n_synth != 0:
            raise ValueError(f"Assign {args.n_synth} images for synthetic data in a batch, but there is no synthetic image in meccano dataset.")
    else:
         raise NotImplementedError(f"Only support 'IndustReal' and 'MECCANO', but get {args.data_path}.")
           
    #--- Setup data log & folder for the run 
    run_path = Path(args.run_path) / args.run_name
    modelsave_path = run_path / "checkpoints"
    log_path = run_path
    tb_dir = run_path / "tensorboard"
    if run_path.exists():
        ans = input(f"Run name {args.run_name} already exists. "
                    f"Are you sure you want to overwrite this folder? (y/*)")
        if ans == "y":
            print(f"Overwriting {args.run_name}")
        else:
            raise ValueError(f"Not overwriting {args.run_name}")

    run_path.mkdir(parents=True, exist_ok=True)
    modelsave_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(tb_dir)

    with open(run_path / 'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(f"Saved run parameters to {run_path / 'args.txt'}")

    print(f"PyTorch Version {torch.__version__}")
    print(f"Start time: {datetime.datetime.now()}")
    print("\n" * 5, "-" * 79, "\n", "-" * 79)
    print("Args for this run: \n", args)

    #--- Define model
    num_state = len(categories) - 2 # except inter state & error state 
    if args.loss == "ce":
        model = ContrastiveModel(args, weights_dir=Path(args.run_path).parent / "spatial_model", classes= num_state)
    else:
        model = ContrastiveModel(args, weights_dir=Path(args.run_path).parent / "spatial_model", classes= num_state)

    #--- Defining the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Loaded optimizer")

    #--- Learning rate scheduler
    if args.scheduler == "cosine_restart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == "cyclic":
        scheduler = CyclicLR(optimizer, base_lr=args.lr/100, max_lr=args.lr, step_size_up=args.lr_step, mode="exp_range",
                             gamma=args.lr_gamma, cycle_momentum=False)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not implemented..")

    #--- Setup loss, and scheduler
    if args.warmup > 0 and args.scheduler != "cyclic":
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, scheduler], milestones=[args.warmup])

    if args.loss == "batch_hard":
        criterion = BatchHardTripletLoss(margin=args.margin, exclude_bg=args.exclude_bg)
        dist = "l2"
    elif args.loss == "batch_all":
        criterion = BatchAllTripletLoss(margin=args.margin, exclude_bg=args.exclude_bg)
        dist = "l2"
    elif args.loss in ["supcon", "simclr"]:
        criterion = SupConLoss(temperature=args.temperature, base_temperature=args.temperature,
                               exclude_background=args.exclude_bg)
        dist = "cos"
    elif args.loss == "ce":
        # cross entropy loss
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        dist = "cos"
    else:
        raise NotImplementedError(f"Loss {args.loss} not implemented.")

    rng = get_rng(args.seed)

    #-- loading dataset
    split = 'train'
    recordings = get_recording_list(Path(args.data_path), train=True)
    train_dataset = TemporalAwareContrastiveDataset(args,args.psr_label_path,args.data_path, split = split, state_category= categories)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.workers)
    n_views = args.n_real + args.n_synth
    if args.n_bg != 0:
        assert args.n_bg % n_views == 0, f"We fold all images into a view per state. Therefore, n_bg must be divisible " \
                                         f"by n_real + n_synth"
    print("Loaded datasets")

    # --- Running training epochs
    best = {"mAP@R": 0., "pr@1": 0., "last_improvement": 0}
    start_epoch = 0
    it = 0

    print("Device", DEVICE)
    print("Start Training")
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch:{epoch}")

        lr = scheduler.get_last_lr()
        print(f"Lr rate:{lr}")
        writer.add_scalar('Learning rate', lr[0], it)
        model.use_projection_head(True)
        train_loss, it = train_contrastive(args.loss, model, train_loader, optimizer, criterion, val=False,
                                           writer=writer, it=it, n_views=n_views)
        model.use_projection_head(False)
        acc_dict, avg_acc_dict = test_metric(model, Path(args.data_path), args.psr_label_path, split='val', w=args.img_w, h=args.img_h,
                                             dist=dist,state_category = categories, args = args)

        if acc_dict["precision_at_1"] > best["pr@1"]:
            print(f"! New best validation pr@1: {acc_dict['precision_at_1']:.4f} (vs. {best['pr@1']:.4f}). Saving model")
            best["pr@1"] = acc_dict["precision_at_1"]
            best["last_improvement"] = epoch

            checkpointname = 'best.pth'
            torch.save(model.state_dict(), modelsave_path / checkpointname)
            print('Checkpoint Saved')
        if avg_acc_dict["mean_average_precision_at_r_nonzero"] > best["mAP@R"]:

            print(f"! New best validation mAP@R for non-zero states: "
                  f"{avg_acc_dict['mean_average_precision_at_r_nonzero']:.4f} (vs. {best['mAP@R']:.4f})")
            best["mAP@R"] = avg_acc_dict["mean_average_precision_at_r_nonzero"]
            best["last_improvement"] = epoch

            checkpointname = 'best_map.pth'
            torch.save(model.state_dict(), modelsave_path / checkpointname)
            print('Checkpoint Saved')

        scheduler.step()

        #-- Logging
        writer.add_scalar('Val pr@1', acc_dict["precision_at_1"], it)
        writer.add_scalar('val pr@1_avg', avg_acc_dict["precision_at_1"], it)
        writer.add_scalar('Val mAP@R zeros', avg_acc_dict["mean_average_precision_at_r_zero"], it)
        writer.add_scalar('val mAP@R states', avg_acc_dict["mean_average_precision_at_r_nonzero"], it)

        # --- Creating dict to save training progress
        dict_to_save = {
            "train_loss": train_loss,
            "pr@1": acc_dict["precision_at_1"],
            "pr@1_avg": avg_acc_dict["precision_at_1"],
            "mAP@R": avg_acc_dict["mean_average_precision_at_r_nonzero"],
            "mAP@R_zero": avg_acc_dict["mean_average_precision_at_r_zero"],
            "lr": lr
        }

        progressfilename = 'train_progress_epoch' + str(epoch) + '.txt'
        save_dict(log_path, progressfilename, dict_to_save)
        c = datetime.datetime.now().strftime('%H:%M:%S')
        log_str = f"{c} - Epoch: {epoch}\t Train loss: {train_loss:.6f}\t pr@1: {acc_dict['precision_at_1']:.4f}\t" \
                  f"pr@1_avg: {avg_acc_dict['precision_at_1']:.4f} \t " \
                  f"mAP@R: {dict_to_save['mAP@R']:.4f} \t "\
                  f"mAP@R_zero: {dict_to_save['mAP@R_zero']:.4f} \t " \
                  f"LR: {lr} \n"
        f = open(log_path / 'train_log.txt', 'a')
        f.write(log_str)
        f.close()

        if epoch % 10 == 0:
            print(f"Saving weight for epoch {epoch}")
            checkpointname = str(epoch) + '.pth'
            torch.save(model.state_dict(), modelsave_path / checkpointname)
            print('Checkpoint Saved')

        # quit training if no improvements for long time
        epochs_since_improvement = epoch - best["last_improvement"]
        print(f"{epochs_since_improvement} epochs since last improvement")
        print(f"Epoch:{epoch} completed")
        print(log_str)
        if epochs_since_improvement > args.stop_after:
            print(f"Quiting training - no improvements for more than {args.stop_after} epochs!")
            break

    writer.close()
    print("Training Completed")
