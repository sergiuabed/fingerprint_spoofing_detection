import numpy as np
from utils.data_utils import load_data, to_effective_prior, split_k_folds, get_empirical_prior
from utils.plot_utils import plot_hist, plot_scatter, plot_heatmap
from utils.dimensionality_reduction import PCA_matrix, LDA_matrix, PCA_variance_plot
from utils.measurements import minimum_bayes_risk

def main():
    DTR, LTR = load_data('dataset/Train.txt')

    print(DTR.shape)
    print(LTR.shape)

    #plot_hist(DTR, LTR)
    #plot_scatter(DTR, LTR)

    s, P = PCA_matrix(DTR, 2)
    DP = np.dot(P.T, DTR)

    plot_hist(DP, LTR, pca=2)
    plot_scatter(DP, LTR, pca=2)

def perform_pca():
    DTR, LTR = load_data('dataset/Train.txt')

    s, P = PCA_matrix(DTR, 2)

    DP = np.dot(P.T, DTR)
    print(s)

    #plot_hist(DP, LTR)
    #plot_scatter(DP, LTR)

    PCA_variance_plot(DTR)

def run_plot_heatmap():
    DTR, LTR = load_data('dataset/Train.txt')

    data_target = DTR[:, LTR==1]
    data_nontarget = DTR[:, LTR==0]

    plot_heatmap(DTR, "dataset", "gray_r")
    plot_heatmap(data_target, "target(authentic fingerprint)", "Oranges")
    plot_heatmap(data_nontarget, "non-target(spoofed)", "Blues")

    s, P = PCA_matrix(DTR, 10)
    DP = np.dot(P.T, DTR)

    plot_heatmap(DP, "dataset_PCA", "gray_r")
    plot_heatmap(DP[:, LTR==1], "target(authentic fingerprint)_PCA", "Oranges")
    plot_heatmap(DP[:, LTR==0], "non-target(spoofed)_PCA", "Blues")

if __name__ == '__main__':
    #main()
    #perform_pca()
    #run_plot_heatmap()
#    print(to_effective_prior((0.5, 1, 1)))
#
#    DTR, LTR = load_data("dataset/Train.txt")
#    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
#    labels = np.concatenate(labels_folds)
#
#    quadr_scores = np.load('logreg/results/scores_quadrlogreg_effpr_0.09090909090909091_znorm_False_pcadim_6_lambda_1e-05.npy')
#    print(minimum_bayes_risk(quadr_scores.reshape((quadr_scores.size,)), labels, (0.75, 1, 1)))
    
#    DTR, LTR = load_data("dataset/Train.txt")
#    print(f"Train set size: {len(LTR)}")
#    print(f"Authentic samples: {len(LTR[LTR == 1])}")
#    print(f"Spoofed samples: {len(LTR[LTR == 0])}")
    
    DTR, LTR = load_data("dataset/Train.txt")
    DTE, LTE = load_data("dataset/Test.txt")
    nr_samples, prior = get_empirical_prior(LTR)
    print(f"nr_samples={nr_samples} prior={prior}")