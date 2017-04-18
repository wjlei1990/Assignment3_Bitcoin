from __future__ import print_function, division
import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc


def load_txt_data(fn):
    t1 = time.time()
    data = pd.read_csv(
        fn, header=None, index_col=None, sep=' ',
        names=['sender','receiver','transaction'])
    t2 = time.time()
    print("Loading data from file(%s) -- shape: {0} -- %.2f sec".format(data.shape) % (fn, t2 - t1))
    return data


def reform_train_data(dftrain, dim):
    train_csr = csr_matrix(
        (dftrain['transaction'], (dftrain['sender'], dftrain['receiver'])),
        shape=(dim, dim), dtype=float)
    # binarize matrix
    train_csr[train_csr != 0] = 1

    return train_csr


def load_data():
    dftrain = load_txt_data('./data/txTripletsCounts.txt')
    dftest = load_txt_data('./data/testTriplets.txt')

    dim = max((dftrain['sender'].max(), dftrain['receiver'].max(),
               dftest['sender'].max(), dftest['receiver'].max()))
    dim += 1
    print("Dimension of train sparse matrix: %d" % dim)
    train_csr = reform_train_data(dftrain, dim)
    return train_csr, dftest


def check_prob_in_range(probs):
    values = []
    for v in probs:
        if v < 0:
            values.append(0.0)
        elif v > 1:
            values.append(1.0)
        else:
            values.append(v)
    return values


def plot_voilin_curve(y_true, y_pred, figname=None):
    plt.figure(figsize=(5, 6))
    bins = np.power(10.0, np.arange(-25, 0, 0.2))
    for _label in [0, 1]:
        array = []
        for idx, v in enumerate(y_true):
            if np.isclose(v,  _label):
                array.append(y_pred[idx])
        print("Label(%d) -- size: %d" % (_label, len(array)))
        values, edge = np.histogram(array, bins=bins)
        #print("values 0:", values)
        values = values / np.max(values) * 0.4
        #plt.figure()
        #plt.plot(edge[:-1], values)
        #ax = plt.gca()
        #ax.set_xscale('log')
        #print("values:", values)
        #print("edge: %s" % edge)
        plt.plot(_label+values, edge[:-1], "k-")
        plt.plot(_label-values, edge[:-1], "k-")

    y_new = np.array(y_true)
    y_new = y_true + np.random.uniform(low=-0.3, high=0.3, size=y_new.shape)

    plt.plot(y_new, y_pred, 'k.', alpha=0.1)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_ylim([1e-25, 1])
    ax.set_xticks([0, 1])
    plt.xlabel("probability")
    plt.xlabel("test value")
    plt.tight_layout()

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)


def plot_roc_curve(y_true, y_pred, figname=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color='magenta', label='ROC(auc = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)


def plot_precision_recall(y_true, y_pred, figname=None):
    # Precision-Recall Curve
    print("Plot Precision-Recall Curve")
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_pred)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label="PR(auc=%.3f)" % pr_auc)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc=0)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
