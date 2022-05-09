import numpy as np
import os
from torch.utils.data import DataLoader

def mini_batch(train, valid, batch_size):
    """

    :param train: not batch train data set
    :param valid: not batch
    :param batch_size:

    :Description data set
    1. shape: N, w, h, c -> batch, w, h, c

    :return:
    """

    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader

def data_set():
    """
    :Description data set
    1. shape: N, w, h, c -> mini batch

    :return: train, valid, test data
    """

