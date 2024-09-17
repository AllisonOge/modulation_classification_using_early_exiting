"""
Finetune a model head on a downstream task
"""
import torch
import torch.nn.functional as F
from sslFineTuner import blHandler
from sslGetData import loadRML22, IQDataset
from sklearn.model_selection import StratifiedShuffleSplit
import functools
import argparse

parser = argparse.ArgumentParser(
    description='Finetune a model head on a downstream task')
# parser.add_argument('--model', type=str,
#                     choices=["bl", "eeV0", "eeV1", "eeV2", "eeV3"], default='bl', help='Model to train')
parser.add_argument('--data', type=str, default='./Data/RML22_ValDataset.pickle.16S',
                    help='Path to the data file')
parser.add_argument('--split_ratio', type=float, default=0.2,
                    help='Ratio to split the dataset')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to use for data loading')
parser.add_argument('--saved_model_path', type=str, default='./Models/byolModel_byol.pth',
                    help='Path to the saved model')
parser.add_argument('--feature_dim', type=int, default=256,
                    help='Feature dimension of the model')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate for training')
parser.add_argument('--weight_decay', type=float, default=1e-8,
                    help='Weight decay for training')
parser.add_argument('--lr_warmup_epochs', type=int, default=10,
                    help='Number of warmup epochs')
parser.add_argument('--lr_warmup_method', type=str, choices=['linear', 'constant'],
                    default='linear', help='Warmup method to use')
parser.add_argument('--lr_warmup_decay', type=float, default=0.1,
                    help='Warmup decay for learning rate')
parser.add_argument('--lr_scheduler', type=str, choices=['cosineannealinglr', 'exponentiallr'],
                    default='cosineannealinglr', help='Learning rate scheduler to use')
parser.add_argument('--lr_min', type=float, default=1e-8,
                    help='Minimum learning rate for cosine annealing scheduler')
parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help='Gamma for exponential learning rate scheduler')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd', 'sdg_nesterov', 'adamw'], default='adam',
                    help='Optimizer to use for training')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for SGD optimizer')


args = parser.parse_args()


def main():
    # finetune the blModel on the downstream task, later expand to other models
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = functools.partial(torch.optim.SGD,
                                      lr=args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay,
                                      nesterov="nesterov" in opt_name,
                                      )
    elif opt_name == "adam":
        optimizer = functools.partial(
            torch.optim.Adam, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "adamw":
        optimizer = functools.partial(
            torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    trainer = blHandler(
        saved_model_path=args.saved_model_path,
        feature_dim=args.feature_dim,
        optimizer=optimizer,
        criterion=F.cross_entropy,
    )

    optimizer = trainer.get_optimizer()
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise NotImplementedError
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[
                args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    # get the dataloaders
    X, y, snrs = loadRML22(args.data)

    sss = StratifiedShuffleSplit(
        n_splits=2, test_size=args.split_ratio, random_state=0)
    train_index, val_index = next(sss.split(X, y))
    train_ds = IQDataset(X[train_index], y[train_index], snrs[train_index])
    val_ds = IQDataset(X[val_index], y[val_index], snrs[val_index])

    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    history = trainer.train(
        args.epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scheduler=lr_scheduler
    )

    # print(history)

    trainer.plot_history(history)


if __name__ == "__main__":
    main()
