import torch
import torch.nn as nn
from torch.nn.functional import softmax
import pytorch_lightning as pl
import torchmetrics
from train_utils import accuracy, AverageMeter

class LitConvNetCam(pl.LightningModule):

    def __init__(self, batch_size, lr_scheduler_milestones, lr_gamma, loss=torch.nn.CrossEntropyLoss(), nclass=7, nfeatures=20, length=120, lr=1e-2, L2_reg=1e-3, top_acc=1):
        super().__init__()

        self.batch_size = batch_size
        self.nclass = nclass
        self.nfeatures = nfeatures
        self.length = length

        # Add a dummy channel dimension for conv1D layer
        self.input_size = (batch_size, 1, length)

        self.loss = loss
        self.lr = lr
        self.lr_scheduler_milestones = lr_scheduler_milestones
        self.lr_gamma = lr_gamma
        self.L2_reg = L2_reg

        # Metrics to log
        if not top_acc < nclass:
            raise ValueError('`top_acc` must be strictly smaller than `nclass`.')
        self.train_acc = torchmetrics.Accuracy(top_k=top_acc)
        self.val_acc = torchmetrics.Accuracy(top_k=top_acc)
        self.train_f1 = torchmetrics.F1(nclass, average='macro')
        self.val_f1 = torchmetrics.F1(nclass, average='macro')

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=nfeatures, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nfeatures),
            nn.ReLU(True)
        )
        self.pool = nn.AvgPool1d(kernel_size=self.length)
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )

    def forward(self, x):
        # (batch_size x length TS)
        x = self.features(x)

        # (batch_size x nfeatures x length_TS)
        # Average pooling for CAM: global pooling so set kernel size to all data
        x = self.pool(x)

        # (batch_size x nfeatures x length_pool; length_pool=1 if global pooling)
        # Flatten features (size batch, lengthpool * nchannels)
        x = x.view(self.batch_size, x.size(2)*self.nfeatures)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.L2_reg)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_scheduler_milestones, gamma=self.lr_gamma),
            'name': 'LearningRate'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        series, label = batch['series'], batch['label']
        series = series.view(self.input_size)
        prediction = self(series)
        #################################################
        # Todo: Check if always necessary
        #label = label.type(torch.LongTensor)
        #################################################
        train_loss = self.loss(prediction, label)
        train_acc = self.train_acc(softmax(prediction, dim=1), label)
        train_f1 = self.train_f1(softmax(prediction, dim=1), label)
        # Add logs of the batch to tensorboard
        self.log('train_loss', train_loss, on_step=True)
        self.log('train_acc', train_acc, on_step=True)
        self.log('train_f1', train_f1, on_step=True)
        return {'loss': train_loss, 'preds': prediction, 'target': label}

    # This hook receive the outputs of all training steps as a list of dictionaries
    def training_epoch_end(self, train_outputs):
        mean_loss = torch.stack([x['loss'] for x in train_outputs]).mean()
        self.log('MeanEpoch/train_loss', mean_loss)
        # The .compute() of Torchmetrics objects compute the average of the epoch and reset for next one
        self.log('MeanEpoch/train_acc', self.train_acc.compute())
        self.log('MeanEpoch/train_f1', self.train_f1.compute())

    def validation_step(self, batch, batch_idx):
        series, label = batch['series'], batch['label']
        series = series.view(self.input_size)
        prediction = self(series)
        #################################################
        # Todo: Check if always necessary
        #label = label.type(torch.LongTensor)
        #################################################
        val_loss = self.loss(prediction, label)
        val_acc = self.val_acc(softmax(prediction, dim=1), label)
        val_f1 = self.val_f1(softmax(prediction, dim=1), label)
        self.log('MeanEpoch/val_acc', val_acc, on_epoch=True, prog_bar=True)
        self.log('MeanEpoch/val_f1', val_f1, on_epoch=True)
        return {'loss': val_loss}

    # This hook receive the outputs of all validation steps as a list of dictionaries
    def validation_epoch_end(self, val_outputs):
        mean_loss = torch.stack([x['loss'] for x in val_outputs]).mean()
        self.log('MeanEpoch/val_loss', mean_loss)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items