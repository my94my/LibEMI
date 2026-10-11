import os
from glob import glob

import numpy as np


def load_dataset(config, stage, method):
    t_data = []  # stores test data on stage==test, train on train
    t_label = []
    val_data = []
    val_label = []

    # 1
    if method == 'cross' and stage == 'train':
        files = sorted(glob(f"{config['DATA_PATH']}/*S*_data.npy"))
        files = (x for x in files
                 if not os.path.basename(x).startswith(str(config['target_subject'])))

        for file in files:
            data, label = load_npy(config['DATA_PATH'],
                                   file)
            t_data.append(data)
            t_label.append(label)

    # 2
    files = sorted(glob(f"{config['DATA_PATH']}/{config['target_subject']:02}S*_data.npy"))
    s_split = (len(files) + 1) // 2
    files = files[s_split:] if stage == 'train' else files[:s_split]

    for file in files:
        data, label = load_npy(config['DATA_PATH'],
                               file)
        if stage == 'train':
            if method == 'cross':
                val_data.append(data)
                val_label.append(label)
            else:
                train_size = int(len(data) * config['train_percent'])

                t_data.append(data[:train_size])
                t_label.append(label[:train_size])
                val_data.append(data[train_size:])
                val_label.append(label[train_size:])
        else:
            t_data.append(data)
            t_label.append(label)

    t_data = np.concatenate(t_data, axis=0)
    t_label = np.concatenate(t_label, axis=0)

    if stage == 'train':
        val_data = np.concatenate(val_data, axis=0)
        val_label = np.concatenate(val_label, axis=0)

    return t_data, t_label, val_data, val_label


def load_npy(root, data_file):
    label_file = data_file.replace('data.npy', 'label.npy')

    data = np.load(data_file)
    label = np.load(label_file)

    return data, label
