# visualization using Jupyter notebook


import os
import matplotlib.pyplot as plt
import torch
import numpy as np


Optimizers = ['SGD', 'SGDR', 'AutoSGD']


def get_folder_path(use_pretrained=True):
    path = 'curve'
    if use_pretrained:
        path = os.path.join(path, 'pretrained')
    return path

def get_curve_data(use_pretrained=True, model='ResNet'):
    folder_path = get_folder_path(use_pretrained)
    filenames = [name for name in os.listdir(folder_path) if name.startswith(model.lower())]
    paths = [os.path.join(folder_path, name) for name in filenames]
    keys = [name.split('-')[1] for name in filenames]
    return {key: torch.load(fp) for key, fp in zip(keys, paths)}

def plot(use_pretrained=True, model='RNN', optimizers=None, curve_type='train'):
    assert model in ['RNN'], 'Invalid model name: {}'.format(model)
    assert curve_type in ['train', 'test'], 'Invalid curve type: {}'.format(curve_type)
    assert all(_ in Optimizers for _ in optimizers), 'Invalid optimizer'

    curve_data = get_curve_data(use_pretrained, model=model)

    plt.figure()
    plt.title('{} Accuracy for {} on PennTreebank'.format(curve_type.capitalize(), model))
    plt.xlabel('Epoch')
    plt.ylabel('{} Accuracy %'.format(curve_type.capitalize()))
    plt.ylim(80, 101 if curve_type == 'train' else 96)

    for optim in optimizers:
        linestyle = '--' if 'AutoSGD' in optim else '-'
        accuracies = np.array(curve_data[optim.lower()]['{}_acc'.format(curve_type)])
        plt.plot(accuracies, label=optim, ls=linestyle)

    plt.grid(ls='--')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot(use_pretrained=True, model='RNN', optimizers=Optimizers, curve_type='train')
    plot(use_pretrained=True, model='RNN', optimizers=Optimizers, curve_type=test)
