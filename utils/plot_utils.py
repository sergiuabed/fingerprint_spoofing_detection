import numpy as np
import matplotlib.pyplot as plt
from measurements import get_confusion_matrix, bayes_risk, minimum_bayes_risk, binary_optimal_bayes_decision

def plot_heatmap(data, title, color):
    cov_mat = covarMat(data)
    corr_mat = np.zeros((data.shape[0], data.shape[0]))

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            #cov = data[:, i:i+1].dot(data[:, j:j+1]) / (np.sqrt(data[:, i:i+1].var()) * np.sqrt(data[:, j:j+1].var()))
            corr_mat[i, j] = np.abs(cov_mat[i, j] / np.sqrt(np.var(data[i]) * np.var(data[j]))) # np.cov(data[:, i], data[:, j])# / (np.sqrt(data[:, i].var()) * np.sqrt(data[:, j].var()))

    plt.imshow(corr_mat, cmap=color, interpolation='nearest')

    plt.colorbar()

    plt.title(title)
    plt.savefig(f"plots/heatmaps/heatmap_{title}.jpg")
    plt.show()


def plot_hist(data, labels):
    classes = np.unique(labels)
    data_classes = [data[:, labels == l] for l in classes]

    for i in range(data.shape[0]):
        plt.figure()
        plt.xlabel(f"dim {i}")
        for c in classes:
            plt.hist(data_classes[c][i, :], bins = 10, density = True, alpha = 0.4, label=c)
        plt.legend()
        plt.savefig('plots/hist_feature%d.jpg' % i)
    #plt.show()


def plot_scatter(data, labels):
    classes = np.unique(labels)
    data_classes = [data[:, labels == l] for l in classes]
    
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            plt.figure()
            plt.xlabel(f'feature {i}')
            plt.ylabel(f'feature {j}')
            for c in classes:
                plt.scatter(data_classes[c][i, :], data_classes[c][j, :], label = c)

            plt.legend()
            plt.savefig(f"plots/scatter_feature{i}_feature{j}.jpg")
    #plt.show()

def plot_roc(llr, labels):
    ths = np.linspace(start=min(llr), stop=max(llr), num=700)
    TPRs = []
    FPRs = []
    for th in ths:
        predicted_labels = np.zeros(llr.shape)
        predicted_labels[llr > th] = 1

        cm = get_confusion_matrix(predicted_labels, labels, 2)
        FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
        FPR = cm[1, 0] / (cm[1, 0] + cm[0, 0])

        TPR = 1 - FNR

        TPRs.append(TPR)
        FPRs.append(FPR)

    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(FPRs, TPRs)
    plt.grid()
    plt.show()

def bayes_error_plot(llr, correct_labels, svm_scores=False):
    '''
    Set 'svm_scores=True' when passing scores from SVM models
    '''

    p_vals = np.linspace(-3, 3, 21)
    dcf = []
    min_dcf = []
    for p in p_vals:
        eff_prior = 1 / (1 + np.exp(-p))

        pl = binary_optimal_bayes_decision(llr, (eff_prior, 1, 1), svm_scores)
        cm = get_confusion_matrix(pl, correct_labels, 2)

        _, DCF_n = bayes_risk(cm, (eff_prior, 1, 1))
        minDCF = minimum_bayes_risk(llr, correct_labels, (eff_prior, 1, 1), svm_scores)

        dcf.append(DCF_n)
        min_dcf.append(minDCF)

    plt.plot(p_vals, dcf, label='DCF', color='r')
    plt.plot(p_vals, min_dcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()

def covarMat(X):
    mu = X.mean(axis=1)  # mean of the dataset
    mu = mu.reshape(mu.size, 1)
    # mu is suptracted from each column(sample) of X through broadcasting
    X_centered = X-mu

    covMat = np.dot(X_centered, X_centered.T) / X.shape[1]

    return covMat