# uses ria hub model builder package
from pathlib import Path
from model_builder.core import (
    Learner,
    DataLoaders,
    SaveModelCallback,
    TerminateOnNaNCallback,
    ReduceLROnPlateauCallback,
    EarlyStoppingCallback,
)

from model_builder.core.metrics import multiclass_accuracy


class Scheduler:
    cosine = "cosine"
    exponential = "exponential"
    linear = "linear"
    one_cycle = "one_cycle"


class Handler:
    def __init__(self, net, nepochs, crit, metrics=[multiclass_accuracy], opt=None, sch=None, modelPath='./models/', modelName='eeHandler'):
        self.net = net
        self.nepochs = nepochs
        self.crit = crit
        self.metrics = metrics
        self.opt = opt
        self.sch = sch
        self.modelPath = modelPath
        self.modelName = modelName

        if self.sch is not None:
            self.sch = Scheduler.cosine
        else:
            assert self.sch in Scheduler.__dict__.values()

    def _create_callbacks(self, learner, metric_name):
        return [
            SaveModelCallback(learner, monitor=metric_name,
                              name=self.modelName),
            TerminateOnNaNCallback(),
            ReduceLROnPlateauCallback(
                learner, monitor=metric_name, patience=10),
            EarlyStoppingCallback(learner, monitor=metric_name, patience=10)
        ]

    def train(self, trainds, valds, bs, val_bs=None, device=None, lr=1e-3, wd=1e-8, metric_name="multiclass_accuracy"):
        dls = DataLoaders.create(
            train_ds=trainds, valid_ds=valds, bs=bs, val_bs=val_bs, device=device)
        self.learner = Learner(dls, self.net, opt_func=self.opt, loss_func=self.crit, metrics=self.metrics, model_dir=self.modelPath) if self.opt else Learner(
            dls, self.net, loss_func=self.crit, metrics=self.metrics, model_dir=self.modelPath)
        # augment the data using mixup
        self.learner = self.learner.mixup()

        cbs = self._create_callbacks(self.learner, metric_name)

        # fit the model
        if self.sch == Scheduler.cosine:
            self.learner.fit_fc(self.nepochs, cbs=cbs,
                                curve_type='cosine', lr=lr, wd=wd)
        elif self.sch == Scheduler.linear:
            self.learner.fit_fc(self.nepochs, cbs=cbs,
                                curve_type='linear', lr=lr, wd=wd)
        elif self.sch == Scheduler.exponential:
            self.learner.fit_fc(
                self.nepochs, cbs=cbs, curve_type='exponential', lr=lr, wd=wd)
        elif self.sch == Scheduler.one_cycle:
            self.learner.fit_one_cycle(self.nepochs, cbs=cbs, lr=lr, wd=wd)

        print('\n\nTraining completed\n\n')

        # validate the model
        val_loss, metrics = self.learner.validate()
        metrics = [metrics] if not isinstance(metrics, list) else metrics
        fmt = ','.join([f'{" ".join(list(map(lambda s: s.capitalize(), name.split("_"))))}: {metric:.3f}' for name, metric in zip(
            self.learner.recorder.metrics_names, metrics)])
        print(
            f'Validation Loss: {val_loss:.5f}', fmt)

        # save the model
        Path.mkdir(Path(self.modelPath), exist_ok=True)
        saved_model = self.learner.save_checkpoint(self.modelName)
        print(f'Model saved at: {saved_model}')

    def infer(self):
        pass
