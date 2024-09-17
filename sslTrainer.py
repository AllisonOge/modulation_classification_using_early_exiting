"""
Trainer script for 4 SSL techniques on Radio Signal dataset
- BYOL
- SimCLR
- MoCov3
- DINO
"""
from typing import Callable
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import copy
import numpy as np

from lightly.models.modules import (
    BYOLProjectionHead,
    BYOLPredictionHead,
    SimCLRProjectionHead
)
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum


class BYOLR(nn.Module):
    def __init__(self, backbone, feature_dim=256):
        super(BYOLR, self).__init__()

        self.online_encoder = backbone
        self.projection_head = BYOLProjectionHead(
            input_dim=feature_dim, hidden_dim=1024, output_dim=feature_dim)
        self.prediction_head = BYOLPredictionHead(
            input_dim=feature_dim, hidden_dim=1024, output_dim=feature_dim)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projection_head = copy.deepcopy(self.projection_head)

        # deactivate the requires_grad for the target encoder pipeline
        deactivate_requires_grad(self.target_encoder)
        deactivate_requires_grad(self.target_projection_head)

    def forward(self, x):
        # forward pass for the online encoder
        rep = self.online_encoder(x).flatten(start_dim=1)
        proj = self.projection_head(rep)
        pred = self.prediction_head(proj)
        return pred

    def forward_target(self, x):
        # forward pass for the target encoder
        rep = self.target_encoder(x).flatten(start_dim=1)
        proj = self.target_projection_head(rep)
        proj = proj.detach()
        return proj


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(256, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class MocoV3R(nn.Module):
    pass


class DINO(nn.Module):
    pass


class Tracker:
    def __init__(self, metric_name, mode='auto'):
        self.metric_name = metric_name
        self.mode = mode
        self.mode_dict = {
            'auto': np.less if 'loss' in self.metric_name else np.greater,
            'min': np.less,
            'max': np.greater
        }
        self.operator = self.mode_dict.get(self.mode, self.mode_dict['auto'])

        self._best = float(
            'inf') if self.operator == np.less else float('-inf')

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        self._best = value


class blHandler:
    """
    Trainer for the baseline model using the SSL techniques
    """

    def __init__(self, backbone: nn.Module,
                 feature_dim: int,
                 classifier: nn.Module,
                 ssl_optimizer: Callable,
                 criterion: nn.Module,
                 classifier_optimizer: torch.optim.Optimizer,
                 classifier_crit: nn.Module,
                 monitor: str = 'val_loss',
                 device: str = None):
        self.backbone = backbone
        self.classifier = classifier
        self.criterion = criterion
        self.classifier_optimizer = classifier_optimizer
        self.classifier_crit = classifier_crit
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'

        # initialize the BYOL model
        self.byol_model = BYOLR(backbone, feature_dim)
        self.byol_model.to(self.device)
        self.classifier.to(self.device)

        self.byol_optimizer = ssl_optimizer(
            params=self.byol_model.parameters())

        self.tracker = Tracker(monitor)

    def get_optimizer(self):
        """
        Get the optimizer for the model
        """
        return self.byol_optimizer

    def get_metric(self, history, metric):
        """
        Get the metric from the history
        """
        return history.get(metric, [None])[-1]

    def save_checkpoint(self, history, path='./Models/blModel_byol.pth', with_opt=False):
        """
        Save the model checkpoint
        """
        # create directory in path if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_metric = self.get_metric(history, self.tracker.metric_name)
        if new_metric is None:
            return
        # TODO: add the optimizer state_dict
        if self.tracker.operator(new_metric, self.tracker.best):
            self.tracker.best = new_metric
            torch.save(self.byol_model.online_encoder.state_dict(), path)
            print(
                f"Model saved with {self.tracker.metric_name}: {new_metric} to {path}")

    def load_checkpoint(self, path='./Models/blModel_byol.pth', with_opt=False):
        """
        Load the model checkpoint
        """
        if not os.path.exists(path):
            return

        # TODO: add the optimizer state_dict
        try:
            self.byol_model.online_encoder.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model from {path}: {e}")

    def validate_byol_training_one_epoch(self, train_dataloader, val_dataloader, criterion):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        if not train_dataloader or not val_dataloader or not criterion:
            return history

        running_tloss = 0.0
        for data, target in train_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.classifier_optimizer.zero_grad()
            output = self.classifier(self.byol_model.online_encoder(data))
            loss = criterion(output, target)
            loss.backward()
            self.classifier_optimizer.step()
            running_tloss += loss.item()

        avg_tloss = running_tloss / len(train_dataloader)
        history['train_loss'].append(avg_tloss)

        self.classifier.eval()
        running_vloss = 0.0
        for data, target in val_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                output = self.classifier(
                    self.byol_model.online_encoder(data))
                loss = criterion(output, target)
                running_vloss += loss.item()
        avg_vloss = running_vloss / len(val_dataloader)
        history['val_loss'].append(avg_vloss)

        return history

    def byol_train(self, epochs, ssl_dataloader, train_dataloader=None, val_dataloader=None, scheduler=None):
        """
        Train the model using BYOL technique
        """
        history = {
            "ssl_loss": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
            "epochs": []
        }
        for epoch in range(epochs):
            running_ssl_tloss = 0.0
            self.byol_model.train()
            self.classifier.train()
            momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
            for batch, _ in ssl_dataloader:
                x0, x1 = batch
                update_momentum(self.byol_model.online_encoder,
                                self.byol_model.target_encoder, m=momentum_val)
                update_momentum(self.byol_model.projection_head,
                                self.byol_model.target_projection_head, m=momentum_val)
                x0, x1 = x0.to(self.device), x1.to(self.device)
                self.byol_optimizer.zero_grad()
                # Stage 1
                # v0 -> backbone -> projection_head -> prediction_head -> p0
                # v1 -> target_backbone -> target_projection_head -> z1
                # Stage 2
                # v1 -> backbone -> projection_head -> prediction_head -> p1
                # v0 -> target_backbone -> target_projection_head -> z0
                # Symmetric loss
                # loss = loss_fn(p0, z1) + loss_fn(p1, z0)
                p0 = self.byol_model(x0)
                z1 = self.byol_model.forward_target(x1)
                p1 = self.byol_model(x1)
                z0 = self.byol_model.forward_target(x0)
                loss = self.criterion(p0, z1) + self.criterion(p1, z0)
                loss.backward()
                self.byol_optimizer.step()
                running_ssl_tloss += loss.item()

            avg_tloss = running_ssl_tloss / len(train_dataloader)
            history['ssl_loss'].append(avg_tloss)
            history['epochs'].append(epoch+1)

            # validate SSL with MLP for clasification task
            # train the classifier
            new_history = self.classifier and self.validate_byol_training_one_epoch(
                train_dataloader, val_dataloader, self.classifier_crit)

            # update the history
            history['train_loss'].extend(new_history['train_loss'])
            history['val_loss'].extend(new_history['val_loss'])
            history['train_acc'].extend(new_history['train_acc'])
            history['val_acc'].extend(new_history['val_acc'])
            last_lr = torch.tensor([param_group['lr']
                                    for param_group in self.byol_optimizer.param_groups]).mean().item()
            history['learning_rate'].append(last_lr)

            if scheduler:
                scheduler.step()

            self.print_history(history)
            self.save_checkpoint(history)

        return history

    def print_history(self, history):
        stats = []
        for key, value in history.items():
            if len(value) == 0:
                continue
            if key != 'epochs':
                stats.append(f"{key}: {value[-1]:.4f}")
        print(f'Epochs: {history["epochs"][-1]},', ", ".join(stats))

    def plot_history(self, history, keys=None, names=None):
        fig = plt.figure(figsize=(12, 8))
        keys = keys or history.keys()
        names = names or [k.replace('_', ' ').capitalize().replace(
            'Ssl', 'SSL') for k in keys]

        assert len(keys) == len(
            names), "Keys and Names should be of same length"

        ax = fig.add_subplot(211)
        for key, name in zip(keys, names):
            if len(history[key]) == 0 or key == 'epochs' or key == 'learning_rate':
                continue
            ax.plot(history['epochs'], history[key], label=name)
        ax.legend()
        ax.grid()

        ax = fig.add_subplot(212)
        ax.plot(history['epochs'], history['learning_rate'],
                label='Learning Rate')
        ax.legend()
        ax.grid()

        fig.savefig('./stats_byol.png', dpi=300)

    def simclr_train(self, **kwargs):
        """
        Train the model using SimCLR technique
        """
        pass

    def mocov3_train(self, **kwargs):
        """
        Train the model using MoCov3 technique
        """
        pass

    def dino_train(self, **kwargs):
        """
        Train the model using DINO technique
        """
        pass


class eeHandler:
    """
    Trainer for the early exiting model using the SSL techniques
    """

    def byol_train(self, **kwargs):
        """
        Train the model using BYOL technique
        """
        pass

    def simclr_train(self, **kwargs):
        """
        Train the model using SimCLR technique
        """
        pass

    def mocov3_train(self, **kwargs):
        """
        Train the model using MoCov3 technique
        """
        pass

    def dino_train(self, **kwargs):
        """
        Train the model using DINO technique
        """
        pass
