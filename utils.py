from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

ECNUM = ['Oxidoreductases', 'Transferases', 'Hydrolases', 'Lyases', 'Isomerases', 'Ligases', 'Translocase']


def plot_class_accuracies(accuracy, stderr):
    """
    Create seaborn plot
    Args:
        accuracy: accuracies
        stderr: standard errors

    Returns:

    """
    df = pd.DataFrame({'Localization': ECNUM,
                       "Accuracy": accuracy,
                       "std": stderr})
    sn.set_style('darkgrid')
    barplot = sn.barplot(x="Accuracy", y="Localization", data=df, ci=None)
    barplot.set(xlabel='Average accuracy', ylabel='')
    barplot.axvline(1)
    plt.errorbar(x=df['Accuracy'], y=ECNUM, xerr=df['std'], fmt='none', c='black', capsize=3)
    plt.tight_layout()
    plt.show()


def plot_confusion(results: np.ndarray):
    """
    Turns results into two confusion matrices, plots them side by side and writes them to tensorboard
    Args:
        results: [n_samples, 2] the first column is the prediction the second is the true label

    Returns:

    """

    confusion = confusion_matrix(results[:, 1], results[:, 0])  # confusion matrix for train
    confusion = confusion/confusion.sum(axis=0)
    train_cm = pd.DataFrame(confusion)

    sn.heatmap(train_cm, annot=True, cmap='Blues', fmt='.1%', rasterized=False)
    plt.show()
