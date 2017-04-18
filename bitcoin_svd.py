import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF, FactorAnalysis
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from utils import reform_train_data, load_data
from utils import check_prob_in_range, plot_voilin_curve, plot_precision_recall, \
    plot_roc_curve


def plot_eigen_values(train_csr, figname=None):
    print("Calculating varaince explained ratio")
    svd = TruncatedSVD(n_components=20, algorithm="arpack")
    #_, s, _ = svds(train_csr, k=100, tol=1e-10, which='LM')
    svd.fit(train_csr)

    data = svd.explained_variance_ratio_
    data = np.sort(svd.explained_variance_ratio_)
    data = data[::-1]

    plt.figure(figsize=(6, 6))
    plt.plot(data, ".-", label="Variance")

    cumu_data = []
    _sum = 0
    for i in range(0, len(data)):
        _sum += data[i]
        cumu_data.append(_sum)
    plt.plot(cumu_data, ".-", label="Cumulative Variance")

    plt.xlabel("Index of selected Singular Values")
    plt.ylabel("Variance Ratio")
    plt.legend(loc=0)
    plt.grid()
    plt.tight_layout()

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)


def train_SVD_model(train_csr, dftest, outputdir, ncomp=16):
    print("=" * 10 + "SVD(ncomp=%d)" % ncomp + "=" * 10)
    print("Outputdir: %s" % outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    figname = os.path.join(outputdir, "variance_explained.pdf")
    print("Output figname: %s" % figname)
    plot_eigen_values(train_csr, figname)

    # SVD make prediciton
    print("Number of Components used in SVD: %d" % ncomp)
    u, s, vt = svds(train_csr, k=ncomp, tol=1e-10, which='LM')
    pred = [np.sum(u[row['sender'], :] * s * vt[:, row['receiver']])
            for index, row in dftest.iterrows()]
    pred = check_prob_in_range(pred)

    outputfn = os.path.join(outputdir, "prediction.svd.%d.txt" % ncomp)
    print("Prediction file: %s" % outputfn)
    np.savetxt(outputfn, pred)

    # ROC Curve
    # print("Plot ROC Curve")
    #label = dftest['transaction']
    #plot_voilin_curve(label, pred)
    #plot_precision_recall(label, pred)
    #plot_roc_curve(label, pred)
    #plt.show()


def train_NMF_model(train_csr, dftest, outputdir, ncomp=12):
    print("=" * 10 + "NMF(ncomp=%d)" % ncomp + "=" * 10)
    print("Outputdir: %s" % outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    print("Number of components: %d" % ncomp)
    nmf = NMF(n_components=ncomp)
    train_new = nmf.fit_transform(train_csr)
    components = nmf.components_
    pred = [np.matmul(train_new[dftest["sender"][idx], :],
                      components[:, dftest["receiver"][idx]])
            for idx in range(len(dftest))]

    outputfn = os.path.join(outputdir, "prediction.NMF.%d.txt" % ncomp)
    print("Prediction file: %s" % outputfn)
    np.savetxt(outputfn, pred)
    return nmf.reconstruction_err_


def plot_fb_norm(xdata, ydata, figname):
    print("Figname: %s" % figname)
    plt.figure(figsize=(6, 6))
    plt.plot(xdata, ydata)
    plt.grid()
    plt.xlabel("Number of reduced dimension")
    plt.ylabel("Frobenius norm")
    plt.tight_layout()
    plt.savefig(figname)


def train_FA_model(train_csr, dftest, outputdir, ncomp=12):
    print("=" * 10 + "FactorAnalysis(ncomp=%d)" % ncomp + "=" * 10)
    print("Outputdir: %s" % outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    print("Number of components: %d" % ncomp)
    fa = FactorAnalysis(n_components=ncomp)
    train_new = fa.fit_transform(train_csr)
    components = fa.components_
    pred = [np.matmul(train_new[dftest["sender"][idx], :],
                      components[:, dftest["receiver"][idx]])
            for idx in range(len(dftest))]

    outputfn = os.path.join(outputdir, "prediction.FA.%d.txt" % ncomp)
    print("Prediction file: %s" % outputfn)
    np.savetxt(outputfn, pred)


def main():

    train_csr, dftest = load_data()
    #outputdir = "output.SVD"
    #train_SVD_model(train_csr, dftest, outputdir, ncomp=10)

    # outputdir = "output.NMF"
    # train_NMF_model(train_csr, dftest, outputdir, ncomp=11)
    outputdir = "output.NMF"
    #ncomps = [1, 2, 3, 4, 5, 6, 8, 11, 14, 17, 20]
    ncomps = [1, 2, 20]
    data = []
    for nc in ncomps:
        d = train_NMF_model(train_csr, dftest, outputdir, ncomp=nc)
        data.append(d)
    print(ncomps)
    print(data)
    figname = os.path.join(outputdir, "nmf_recon_error.pdf")
    plot_fb_norm(ncomps, data, figname)

    #outputdir = "output.FA"
    #for ncomp in [1, 2, 5, 10, 12]:
    #    train_FA_model(train_csr, dftest, outputdir, ncomp=ncomp)


if __name__ == "__main__":
    main()
