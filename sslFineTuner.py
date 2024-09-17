"""
Handler for fine tuning the SSL model
"""
from typing import Collection, Callable, Optional
import torch
import torch.nn as nn
from getModel import blModel
from sslTrainer import Tracker
import os
import numpy as np
import matplotlib.pyplot as plt


class blHandler:
    def __init__(self, saved_model_path: str,
                 feature_dim: int, optimizer: torch.optim, criterion: torch.nn.Module,
                 metrics: Optional[Collection[Callable]] = None,
                 monitor: str = "val_loss",
                 device: str = None):
        self.backbone = nn.Sequential(
            *list(blModel(feature_dim=feature_dim).children())[:-1])
        assert os.path.isfile(
            saved_model_path), f"Model path {saved_model_path} does not exist"
        self.backbone.load_state_dict(torch.load(
            saved_model_path, weights_only=True))
        print(f"Model loaded from {saved_model_path}")
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        # freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            *list(blModel(feature_dim=feature_dim).children())[-1])

        self.model = nn.Sequential(self.backbone, self.head).to(self.device)
        self.optimizer = optimizer(params=self.model.parameters())
        self.criterion = criterion
        self.metrics = metrics or []

        self.tracker = Tracker(metric_name=monitor)

    def get_optimizer(self):
        return self.optimizer

    def save_checkpoint(self, history, path='./Models/fineTunedModel.pth'):
        """
        Save the model checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_metric = history.get(self.tracker.metric_name, [None])[-1]
        if new_metric is None:
            return
        if self.tracker.operator(new_metric, self.tracker.best):
            self.tracker.best = new_metric
            torch.save(self.model.state_dict(), path)
            print(
                f"Model saved with {self.tracker.metric_name}: {new_metric} to {path}")

    def load_checkpoint(self, path):
        pass

    def print_history(self, history):
        stats = []
        for key, value in history.items():
            if len(value) == 0:
                continue
            if key != "epochs":
                stats.append(f"{key}: {value[-1]:.4f}")
        print(f"Epoch :{history['epochs'][-1]},", ", ".join(stats))

    def plot_history(self, history, keys=None, names=None):
        fig = plt.figure(figsize=(12, 8))
        keys = keys or history.keys()
        names = names or [k.replace('_', ' ').capitalize() for k in keys]

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

        fig.savefig('./stats_finetunned.png', dpi=300)

    def train(self, epochs, train_dataloader, val_dataloader=None, scheduler=None):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
            "epochs": []
        }
        self.model.train()
        for epoch in range(1, epochs+1):
            running_tloss = 0.
            for data, target in train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_tloss += loss.item()
            running_tloss /= len(train_dataloader)
            history["train_loss"].append(running_tloss)
            history["epochs"].append(epoch)

            running_vloss = 0.
            for data, target in val_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_vloss += loss.item()
            running_vloss /= len(val_dataloader)
            history["val_loss"].append(running_vloss)

            last_lr = torch.tensor([param_group['lr']
                                    for param_group in self.optimizer.param_groups]).mean().item()
            history['learning_rate'].append(last_lr)

            if scheduler:
                scheduler.step()

            self.print_history(history)
            self.save_checkpoint(history)
        return history

    @staticmethod
    def infer(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = None):
        """
        Infer the model on the dataloader
        """
        model.eval()
        device = device or "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        actuals = []
        preds = []
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
                # convert the logits to labels
                _, output = torch.max(torch.softmax(output, dim=-1), dim=-1)
                actuals.append(target.detach().cpu())
                preds.append(output.detach().cpu())

        actuals = torch.cat(actuals, dim=0)
        preds = torch.cat(preds, dim=0)
        # calculate the accuracy vs snrs
        snrs = dataloader.dataset.get_snrs()
        accuracy = {}
        for snr in np.unique(snrs):
            mask = (snrs == snr)
            accuracy[snr] = (actuals[mask] == preds[mask]
                             ).float().mean().item()
        return preds, accuracy
