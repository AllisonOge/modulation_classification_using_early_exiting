"""
Train a model using SSL techniques
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from getModel import blModel, eeModel_V0, eeModel_V1, eeModel_V2, eeModel_V3
from sslTrainer import blHandler, eeHandler
from sslGetData import loadRML22, BYOLRData, SimCLRData, MoCoV3Data, DINOData
import functools
import argparse

parser = argparse.ArgumentParser(
    description='Train a model using SSL techniques')
# parser.add_argument('--model', type=str,
#                     choices=["bl", "eeV0", "eeV1", "eeV2", "eeV3"], default='bl', help='Model to train')
# parser.add_argument('--ssl', type=str, choices=[
#                     "byol", "simclr", "mocov3", "dino"], default='byol', help='SSL technique to use')
parser.add_argument('--data', type=str, default='./Data/RML22.pickle.01A',
                    help='Path to the data file')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate for training')
parser.add_argument('--weight_decay', type=float, default=1e-4,
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
    # train bl model on RML22 dataset using byol technique and expand the training script to include other techniques

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

    backbone = nn.Sequential(*list(blModel().children())[:-1])
    classifier = nn.Sequential(*list(blModel().children())[-1])
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def loss_fn(x0: torch.Tensor, x1: torch.Tensor):
        return 2. - 2. * F.cosine_similarity(x0, x1, dim=-1).mean()

    trainer = blHandler(
        backbone=backbone,
        feature_dim=256,
        criterion=loss_fn,
        ssl_optimizer=optimizer,
        classifier=classifier,
        classifier_optimizer=classifier_optimizer,
        classifier_crit=F.cross_entropy,
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

    byol_data = BYOLRData(X, y, snrs, nsplits=2, test_size=0.4,
                          batch_size=args.batch_size, num_workers=4)

    signal_length = X.shape[2]

    ssl_dataloader, train_dataloader, val_dataloader = byol_data.get_dataloaders(
        signal_length)

    history = trainer.byol_train(
        args.epochs,
        ssl_dataloader,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scheduler=lr_scheduler
    )

    # print(history)

    trainer.plot_history(history)


if __name__ == "__main__":
    main()
