import numpy as np


def scaling(eeg, augment_times=1, sigma=0.1):
    b, c, t = eeg.shape

    factors = np.random.normal(
        loc=1.0,
        scale=sigma,
        size=(augment_times, b, c, 1)
    )
    augmented = eeg[None, ...] * factors

    return augmented.reshape(augment_times * b, c, t)


def jitter(eeg, augment_times=1, sigma=0.05):
    b, c, t = eeg.shape

    channel_std = np.std(eeg, axis=-1, keepdims=True)
    noise = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=(augment_times, b, c, t)
    )
    augmented = eeg[None, ...] + noise * sigma * channel_std[None, ...]

    return augmented.reshape(augment_times * b, c, t)


def cutcat(data, labels, ratio=8):
    data = np.asarray(data)
    labels = np.asarray(labels)
    b, c, t = data.shape

    mixed_data = data.copy()
    mixed_labels = labels.astype(np.float32, copy=True)
    max_length = max(1, t // ratio)

    for i in range(b):
        partner = (i + np.random.randint(1, b)) % b

        length = np.random.randint(1, max_length + 1)
        start = np.random.randint(0, t - length + 1)
        end = start + length

        mixed_data[i, :, start:end] = data[partner, :, start:end]

        lam = length / t
        mixed_labels[i] = (1 - lam) * labels[i] + lam * labels[partner]

    augmented_data = np.concatenate([data, mixed_data], axis=0)
    augmented_labels = np.concatenate(
        [labels.astype(np.float32), mixed_labels], axis=0
    )
    return augmented_data, augmented_labels
