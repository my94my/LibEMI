from importlib import import_module

from torch.utils.data import Dataset

from dataloader.augmentation import *


def get_dataset(config, stage, method):
    assert method in ['cross', 'within']
    
    dataset_module = import_module(f"dataloader.{config['dataset']}")

    t_data, t_label, val_data, val_label = dataset_module.load_dataset(config, stage, method)

    onehot_t_label = np.eye(config['n_class'])[t_label]
    if stage == 'test':
        t_data = t_data[:, None, ..., -1000:]
        return BaseEEGDataset(t_data, onehot_t_label)
    else:
        epsilon = 0.05
        onehot_t_label = (1 - epsilon) * onehot_t_label + epsilon / config['n_class']

        # augmented_data, augmented_labels = cutcat(t_data, onehot_t_label)
        # t_data = np.concatenate((t_data, augmented_data), axis=0)
        # t_data = augmented_data
        t_data = t_data[:, None, ..., -1000:]

        onehot_t_label = np.concatenate((onehot_t_label,) * 2, axis=0)
        # onehot_t_label = augmented_labels

        val_data = val_data[:, None, ..., -1000:]
        onehot_val_label = np.eye(config['n_class'])[val_label]
        print(t_data.shape, val_data.shape)

        return BaseEEGDataset(t_data, onehot_t_label), BaseEEGDataset(val_data, onehot_val_label)


class BaseEEGDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'data': self.data[idx, ...], 'label': self.label[idx]}
