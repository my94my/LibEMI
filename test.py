'''Import libraires'''
import logging
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
import argparse

from glob import glob
from os import environ, makedirs, path
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.logger import DummyLogger
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader
from dataloader.Loader import get_dataset
from model.LitModel import LitModel
from warnings import filterwarnings

parser = argparse.ArgumentParser()

parser.add_argument('--testno', type=str)
parser.add_argument('--seed', type=int, default=103)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--config_name', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--method', type=str, default='within')
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()

filterwarnings('ignore')

### Set confings
with open(f'./configs/{args.config_name}.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    config = EasyDict(config)

#### Set logging ####
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

#### Set SEED ####
seed_everything(args.seed, verbose=args.verbose)
# config['current_time'] = datetime.now().strftime('%Y%m%d')

#### Set Device ####
environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True
torch.set_float32_matmul_precision('high')

total_results = []
total_kappas = []
#### Update configs ####
config['dataset'] = args.config_name
config['model'] = args.model
config['LOG_NAME'] = path.join(args.config_name, args.model, f"test{args.testno}")
print(f"{config['LOG_NAME']} start.")

for subject in range(1, config['n_subject'] + 1):
    config['target_subject'] = subject
    dataset = get_dataset(config, stage='test', method=args.method)
    config['batch_shape'] = dataset[0]['data'].shape
    # config['channel_info'] = dataset.chan_info

    ### Evaluation
    test_dataloader = DataLoader(dataset,
                                 batch_size=config['batch_size'],
                                 pin_memory=False,
                                 num_workers=config['num_workers'],
                                 shuffle=False,
                                 )

    ckpt_path = sorted(glob(f"{config['CKPT_PATH']}/{config['LOG_NAME']}/S{subject:02d}*"))[-1]

    model = LitModel.load_from_checkpoint(ckpt_path, config=config, weights_only=False)

    logger = DummyLogger()
    trainer = Trainer(enable_progress_bar=args.verbose,
                      enable_model_summary=args.verbose,
                      logger=logger,
                      )

    preds = trainer.predict(model, dataloaders=test_dataloader)
    preds = torch.cat(preds, dim=0)
    preds = preds.argmax(dim=1)
    trues = torch.from_numpy(dataset.label).argmax(dim=1)

    torch.cuda.empty_cache()

    total_results.append(accuracy_score(preds, trues))
    total_kappas.append(cohen_kappa_score(preds, trues))

### Accuracy
acc_result_df = pd.DataFrame(total_results)
acc_result_df.index = [f'S{idx}' for idx in range(1, config['n_subject'] + 1)]
acc_result_df.loc['Avg.'] = acc_result_df.mean()

### Kappa
kappa_result_df = pd.DataFrame(total_kappas)
kappa_result_df.index = [f'S{idx}' for idx in range(1, config['n_subject'] + 1)]
kappa_result_df.loc['Avg.'] = kappa_result_df.mean()

result_df = pd.merge(acc_result_df, kappa_result_df, left_index=True, right_index=True, how='inner')
result_df.columns = ['Acc.', 'Kappa']

print('\n\n')
print('=' * 24)
print(config['dataset'], config['model'], ' Result: ')
print(result_df)
print('=' * 24)
print('\n\n')
