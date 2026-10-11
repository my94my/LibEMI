# LibEMI

This project is licensed under the **GNU General Public License v2 (GPL v2)**. When using, modifying, or distributing the project code, please comply with the terms of GPL v2.

**If you use this project's code, please also cite our paper.**

```bibtex

```

LibEMI is a PyTorch-based project for EEG classification experiments. It provides multiple models and dataset loaders, supports within-subject and cross-subject training, and reports classification accuracy and Cohen's kappa for each subject.

## 1. Project Structure

| Path | Description |
| --- | --- |
| `train.py` | Train a model for a single specified target subject |
| `test.py` | Load model weights and evaluate all subjects specified in the configuration sequentially |
| `configs/` | Dataset configurations |
| `dataloader/` | Dataset loading and data processing |
| `model/` | Model implementations and Lightning training modules |
| `checkpoints/` | Model weights saved during training |

## 2. Environment Setup

Create an Anaconda environment using the following commands:

```bash
conda create -n LibEMI python=3.12
conda activate LibEMI
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install PyYAML
pip install easydict
pip install torchmetrics
pip install pytorch_lightning
pip install scikit-learn
pip install mne
pip install -U 'tensorboard'
pip install triton
```

## 3. Data Preparation

### 3.1 Downloading the Data

This project provides **preprocessed data**, which can be used directly without additional preprocessing in this project. The preprocessing procedure is described in the paper.

**Download link:** https://pan.baidu.com/s/1DsnDj074zFpZ0TGv4Vuy1w?pwd=M94M

Download and extract the data, preserving the filenames in the downloaded archive. Then update the data paths in the configuration files under `configs/`.

To obtain the preprocessing code, please contact the authors of the paper.

### 3.2 Custom Dataset Loaders

To use your own data, refer to the existing dataset loaders in `dataloader/` and add a custom loader, such as `NewDataset.py`. Use the same function names, parameters, and return values as the existing loaders. The returned data must satisfy the following requirements:

- The data must have shape `(N, C, T)`, where `N` is the number of samples, `C` is the number of channels, and `T` is the number of time points.
- The labels must have shape `(N,)` and contain integers from `0` to `n_class - 1`. Do not convert labels to one-hot encoding in advance.
- Each data array and its corresponding label array must contain the same number of samples in the same order.

The current loading pipeline selects the last 1,000 time points and adds a dimension to produce model input tensors of shape `(N, 1, C, 1000)`. To use a different temporal length or input shape, modify the loading pipeline in `dataloader/Loader.py`.

Also add a corresponding YAML configuration file under `configs/`, such as `NewDataset.yaml`. Its filename must match the dataset loader's filename. At runtime, specify `--config_name NewDataset` (without `.yaml`).

## 4. Custom Models

To use your own model, refer to the existing implementations in `model/` and add a new file, such as `MyModel.py`, to that directory. The module must expose a model entry point that can be instantiated with `Net(config)`. At runtime, specify `--model MyModel` (without `.py` ).

We recommend copying an existing model file as a template and replacing its network architecture.

## 5. Training

### 5.1 Within-Subject Training

The following example uses experiment identifier `1`, EEGNet, and subject `1` from the BCIIV2A dataset:

```bash
python train.py --config_name BCIIV2A --model EEGNet --subject 1 --testno 1 --method within --gpu 0 --seed 103
```

After training, the model weights with the highest validation accuracy are saved to:

```text
checkpoints/BCIIV2A/EEGNet/test1/S01_epoch=XX-val_acc=X.XXX.ckpt
```

Only one best-performing checkpoint is stored for each subject.

### 5.2 Cross-Subject Training

Set `--method` to `cross`:

```bash
python train.py --config_name BCIIV2A --model EEGNet --subject 1 --testno 2 --method cross --gpu 0 --seed 103
```

| Mode | Training Data | Validation Data | Test Data |
| --- | --- | --- | --- |
| `within` | First 80% of samples in each training file for the target subject | Remaining 20% of samples in the same file | Test files for the target subject |
| `cross` | All training and test files for the other subjects | Training files for the target subject | Test files for the target subject |

## 6. Testing

The following example uses experiment identifier `1`, EEGNet, and the BCIIV2A dataset, following the training example for subject `1` above.

Ensure that all subjects have been trained under the same experiment identifier, then run:

```bash
python test.py --config_name BCIIV2A --model EEGNet --testno 1 --method within --gpu 0 --seed 103
```

The testing arguments `--config_name`, `--model`, and `--testno` must match those used for training. The input shape, number of classes, and model configuration must also remain consistent.

Test results are printed to the console, including test accuracy and Cohen's kappa for each subject, along with the arithmetic mean of each metric across subjects.

## 7. Command-Line Arguments

| Argument | Description | Default / Example |
| --- | --- | --- |
| `--config_name` | YAML configuration filename under `configs/` | Example: `BCIIV2A`; must be specified explicitly |
| `--model` | Model filename under `model/` | Example: `EEGNet`; must be specified explicitly |
| `--subject` | Target subject for training | Example: `1` |
| `--testno` | Experiment identifier string | Example: `test1`; explicit specification is recommended for each run |
| `--method` | Within-subject or cross-subject training: `within` or `cross` | `within` |
| `--gpu` | GPU identifier used to set `CUDA_VISIBLE_DEVICES` | `0` |
| `--seed` | Random seed | `583` |
| `--verbose` | Display progress bars, model summaries, and other details | `False` |
