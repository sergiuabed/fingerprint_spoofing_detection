import numpy as np
from data_utils import load_data
from plot_utils import plot_hist, plot_scatter, plot_heatmap
from dimensionality_reduction import PCA_matrix, LDA_matrix, PCA_variance_plot

def main():
    DTR, LTR = load_data('dataset/Train.txt')

    print(DTR.shape)
    print(LTR.shape)

    plot_hist(DTR, LTR)
    plot_scatter(DTR, LTR)

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
    run_plot_heatmap()
