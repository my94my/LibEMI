import numpy as np


def load_dataset(config, stage, method):
    t_data = []  # stores test data on stage==test, train on train
    t_label = []
    val_data = []
    val_label = []

    # 1
    if method == 'cross' and stage == 'train':
        suffixs = ['T', 'E']
        target_subjects = (x for x in range(1, config['n_subject'] + 1)
                           if x != config['target_subject'])
        for sub in target_subjects:
            for suffix in suffixs:
                data, label = load_npy(config['DATA_PATH'],
                                       sub,
                                       suffix)
                t_data.append(data)
                t_label.append(label)

    # 2
    suffixs = ['T', ] if stage == 'train' else ['E', ]

    for suffix in suffixs:
        data, label = load_npy(config['DATA_PATH'],
                               config['target_subject'],
                               suffix)

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


def load_npy(root, sub, suffix):
    filename = f'{root}/{sub:02}{suffix}'
    data_file = filename + '_data.npy'
    label_file = filename + '_label.npy'

    data = np.load(data_file)
    label = np.load(label_file)

    return data, label
