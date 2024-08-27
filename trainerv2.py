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


class Scheduler:
    cosine = "cosine"
    exponential = "exponential"
    linear = "linear"
    one_cycle = "one_cycle"


class Handler:
    def __init__(self, net, crit, nepochs, opt=None, sch=None, modelPath='./models/', modelName='eeHandler'):
        self.net = net
        self.crit = crit
        self.opt = opt
        self.sch = sch
        self.nepochs = nepochs
        self.modelPath = modelPath
        self.modelName = modelName

        if self.sch is not None:
            self.sch = Scheduler.cosine
        else:
            assert self.sch in Scheduler.__dict__.values()

    def _create_callbacks(self, learner):
        return [
            SaveModelCallback(learner, monitor='accuracy'),
            TerminateOnNaNCallback(),
            ReduceLROnPlateauCallback(learner, monitor='accuracy'),
            EarlyStoppingCallback(learner, monitor='accuracy')
        ]

    def train(self, trainds, valds, bs, val_bs=None, device=None, lr=1e-3, wd=1e-8):
        dls = DataLoaders.create(
            train_ds=trainds, valid_ds=valds, bs=bs, val_bs=val_bs, device=device)
        self.learner = Learner(dls, self.net, opt_func=self.opt, loss_func=self.crit) if self.opt else Learner(
            dls, self.net, loss_func=self.crit)
        # augment the data using mixup
        self.learner = self.learner.mixup()

        cbs = self._create_callbacks(self.learner)

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
        val_loss, accuracy = self.learner.validate()
        print(f'Validation loss: {val_loss:.5f}, Accuracy: {accuracy:.3f}')

        # save the model
        saved_model = self.learner.save_checkpoint(
            Path(self.modelPath)/self.modelName)
        print(f'Model saved at: {saved_model}')

    def infer(self):
        pass
