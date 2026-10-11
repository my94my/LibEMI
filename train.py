'''Import libraires'''
import glob
import logging
import torch
import torch.backends.cudnn as cudnn
import yaml

from argparse import ArgumentParser
from os import environ
from os import path
from time import perf_counter
from warnings import filterwarnings
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import logger
from torch.utils.data import DataLoader
from dataloader.Loader import get_dataset
from model.LitModel import LitModel, get_callbacks

parser = ArgumentParser()
parser.add_argument('--testno', type=str)
parser.add_argument('--subject', type=int)
parser.add_argument('--seed', type=int, default=103)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--config_name', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--method', type=str, default='within')
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()

#### Config setting ####
with open(f'./configs/{args.config_name}.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    config = EasyDict(config)

#### Update configs ####
config['LOG_NAME'] = path.join(args.config_name, args.model, f"test{args.testno}")
# Exit if ckpt file exists
ckpt_name = path.join(config['CKPT_PATH'], config['LOG_NAME'], f"S{args.subject:02d}*")
for g in glob.iglob(ckpt_name):
    if path.isfile(g):
        raise FileExistsError(f"{ckpt_name}: checkpoint exist, aboarting!")
config['dataset'] = args.config_name
config['model'] = args.model
config['target_subject'] = args.subject
config['train_percent'] = 0.8
train_dataset, val_dataset = get_dataset(config, stage='train', method=args.method)
config['lr'] = float(config['lr'])
config['batch_shape'] = train_dataset[0]['data'].shape

#### Set Device ####
environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True
torch.set_float32_matmul_precision('high')

#### Set logging ####
filterwarnings('ignore')
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# logger = TensorBoardLogger(config['LOG_PATH'],
#                            name=config['LOG_NAME'],
#                            log_graph=True)
logger = logger.DummyLogger()
print(f"{config['LOG_NAME']} S{args.subject:02d} start.")

#### Set SEED ####
seed_everything(args.seed, verbose=args.verbose)

#### Training ####
train_dataloader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              pin_memory=True,
                              num_workers=config['num_workers'],
                              shuffle=True,
                              drop_last=True
                              )

val_dataloader = DataLoader(val_dataset,
                            batch_size=config['batch_size'],
                            pin_memory=True,
                            num_workers=config['num_workers'],
                            shuffle=False, )

model = LitModel(config)

callbacks = get_callbacks(monitor='val_acc', config=config)

trainer = Trainer(enable_progress_bar=args.verbose,
                  enable_model_summary=args.verbose,
                  logger=logger,
                  max_epochs=config['EPOCHS'],
                  callbacks=callbacks,
                  default_root_dir=config['CKPT_PATH'],
                  accelerator='cuda',
                  )

start_time = perf_counter()
trainer.fit(model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)

print(f"{config['LOG_NAME']} S{args.subject:02d} training time: {perf_counter() - start_time:.2f} seconds.")
