import itertools
import os

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def confuse_matrix(target, pred):

    confusion_mat = confusion_matrix(target, pred)

    return confusion_mat


def plot_confuse_mat(matrix, labels, log_dir):

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix!")
    plt.colorbar()

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)

    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, '{:,}'.format(matrix[i, j]),
                 horizontalalignment='center',
                 color='black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Prediction Label')
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))


