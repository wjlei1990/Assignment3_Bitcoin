import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from utils import plot_voilin_curve, load_txt_data
from utils import plot_roc_curve


def plot_multiple_roc_curve(y_true, y_preds, figname=None):
    plt.figure()

    keys = y_preds.keys()
    keys.sort()
    for _k in keys:
        _y = y_preds[_k]
        fpr, tpr, thresholds = roc_curve(y_true, _y)
        roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve(%s) : %f" % (_k, roc_auc))
        plt.plot(fpr, tpr, label='ROC %s(area = %0.3f)' % (_k, roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    #plt.legend(loc="lower right")
    plt.legend(loc=0)
    plt.grid()

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)


def plot_multiple_precision_recall(y_true, y_preds, figname=None):
    # Precision-Recall Curve
    print("Plot Precision-Recall Curve")

    plt.figure()
    keys = y_preds.keys()
    keys.sort()
    for _k in keys:
        _v = y_preds[_k]
        precision, recall, thresholds = precision_recall_curve(
            y_true, _v)
        _auc = auc(recall, precision)
        plt.plot(recall, precision, label="%s(AUC=%.3f)" % (_k, _auc))

    plt.plot([0, 1], [0.1, 0.1], 'k--')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.ylim([0.0, 1.05])
    #plt.legend()
    plt.grid()
    plt.legend(loc=0)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)


def main():
    y_true = load_txt_data('./data/testTriplets.txt')["transaction"]

    y_pred_svd_1 = np.loadtxt("output.SVD/prediction.svd.1.txt")
    y_pred_svd_2 = np.loadtxt("output.SVD/prediction.svd.2.txt")
    y_pred_svd_5 = np.loadtxt("output.SVD/prediction.svd.5.txt")
    y_pred_svd_10 = np.loadtxt("output.SVD/prediction.svd.10.txt")
    y_pred_svd_11 = np.loadtxt("output.SVD/prediction.svd.11.txt")

    y_pred_nmf_1 = np.loadtxt("output.NMF/prediction.NMF.1.txt")
    y_pred_nmf_2 = np.loadtxt("output.NMF/prediction.NMF.2.txt")
    y_pred_nmf_5 = np.loadtxt("output.NMF/prediction.NMF.5.txt")
    y_pred_nmf_8 = np.loadtxt("output.NMF/prediction.NMF.8.txt")
    y_pred_nmf_11 = np.loadtxt("output.NMF/prediction.NMF.11.txt")
    y_pred_nmf_12 = np.loadtxt("output.NMF/prediction.NMF.12.txt")

    #plot_voilin_curve(y_true, y_pred)
    #preds = {"svd_10": y_pred_10, "svd_11": y_pred_11, "svd_12": y_pred_12,
    #         "svd_15": y_pred_15, "svd_16": y_pred_16}
    preds = {"svd_1": y_pred_svd_1, "svd_2": y_pred_svd_2, "svd_5": y_pred_svd_5,
             "svd_10": y_pred_svd_10, "svd_11": y_pred_svd_11,
             "NMF_12": y_pred_nmf_12, "NMF_11": y_pred_nmf_11,
             "NMF_8": y_pred_nmf_8, "NMF_2": y_pred_nmf_2, "NMF_5": y_pred_nmf_5,
             "NMF_1": y_pred_nmf_1}
    plot_multiple_roc_curve(y_true, preds)
    plot_multiple_precision_recall(y_true, preds)


def plot_one_svd():
    y_true = load_txt_data('./data/testTriplets.txt')["transaction"]
    y_pred = np.loadtxt("output.SVD/prediction.svd.11.txt")
    outputdir = "output.SVD"
    figname = os.path.join(outputdir, "svd_pr_voilin.pdf")
    print("figname: %s" % figname)
    plot_voilin_curve(y_true, y_pred, figname=figname)


def plot_one_nmf():
    y_true = load_txt_data('./data/testTriplets.txt')["transaction"]
    y_pred = np.loadtxt("output.NMF/prediction.NMF.11.txt")
    outputdir = "output.NMF"
    figname = os.path.join(outputdir, "nmf_pr_voilin.pdf")
    print("figname: %s" % figname)
    plot_voilin_curve(y_true, y_pred, figname=figname)


if __name__ == "__main__":
    #main()
    plot_one_svd()
    plot_one_nmf()
