from importlib import import_module
from os import path

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import accuracy, cohen_kappa


class LitModel(LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        self.save_hyperparameters()
        self.config = config
        self.__init_model()

        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def evaluate(self, batch, stage=None):
        inputs = batch['data'].type(torch.float)
        labels = batch['label'].type(torch.float)

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)
        acc = accuracy(preds, labels,
                       num_classes=self.config['n_class'], task='multiclass')
        kappa = cohen_kappa(preds, labels,
                            num_classes=self.config['n_class'], task='multiclass')

        self.log(f'{stage}_loss', loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)
        self.log(f'{stage}_acc', acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)
        self.log(f'{stage}_kappa', kappa,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            self.sample_batch = batch['data'].type(torch.float)
        loss = self.evaluate(batch, stage='train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.evaluate(batch, stage='val')
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.evaluate(batch, stage='test')
        return {'loss': loss}

    # def on_train_epoch_end(self):
    #     if self.current_epoch == 0:
    #         self.logger.experiment.add_graph(self.model, self.sample_batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch['data'].type(torch.float)
        return self(inputs)

    def configure_optimizers(self):
        self.__init_optimizers()
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler_config,
                }

    def __init_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'],
                                     betas=self.config['betas'],
                                     weight_decay=self.config['weight_decay'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['EPOCHS'],
                                                                 eta_min=1e-6)
        self.lr_scheduler_config = {
            'scheduler': self.lr_scheduler,
            'internal': 'epoch'
        }

    def __init_model(self):
        model_module = import_module(f"model.{self.config['model']}")
        self.model = model_module.Net(self.config)


def get_checkpoint_callback(monitor: str, config: dict):
    if monitor == 'val_acc':
        return ModelCheckpoint(monitor=monitor,
                               dirpath=str(path.join(config['CKPT_PATH'], config['LOG_NAME'])),
                               filename=f"S{config['target_subject']:02d}_{{epoch:02d}}-{{val_acc:.3f}}",
                               save_top_k=1,
                               mode='max')
    elif monitor == 'val_loss':
        return ModelCheckpoint(monitor=monitor,
                               dirpath=str(path.join(config['CKPT_PATH'], config['LOG_NAME'])),
                               filename=f"S{config['target_subject']:02d}_{{epoch:02d}}-{{val_acc:.3f}}",
                               save_top_k=1,
                               mode='min')
    else:
        return None


def get_callbacks(monitor: str, config: dict):
    checkpoint_callback = get_checkpoint_callback(monitor=monitor, config=config)
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    return [checkpoint_callback, lr_logger]
