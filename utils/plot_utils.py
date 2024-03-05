import sys
sys.path.append('.')
sys.path.append('./utils')

import numpy as np
import matplotlib.pyplot as plt
from utils.measurements import get_confusion_matrix, bayes_risk, minimum_bayes_risk, binary_optimal_bayes_decision
from utils.data_utils import load_data, split_k_folds, to_effective_prior
import os

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


def plot_hist(data, labels, pca=None):
    classes = np.unique(labels)
    data_classes = [data[:, labels == l] for l in classes]

    histograms_path = './plots/histograms' if pca == None else f'./plots/histograms_pca_{pca}'
    if not os.path.isdir(histograms_path):
        os.mkdir(histograms_path)

    for i in range(data.shape[0]):
        plt.figure()
        plt.xlabel(f"dim {i}")
        for c in classes:
            plt.hist(data_classes[c][i, :], bins = 10, density = True, alpha = 0.4, label=c)
        plt.legend()
        plt.savefig(f'{histograms_path}/hist_feature{i}.jpg')
    #plt.show()


def plot_scatter(data, labels, pca=None):
    classes = np.unique(labels)
    data_classes = [data[:, labels == l] for l in classes]
    
    scatter_path = './plots/scatter' if pca == None else f'./plots/scatter_pca_{pca}'
    if not os.path.isdir(scatter_path):
        os.mkdir(scatter_path)

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):#range(i+1, data.shape[0]):
            plt.figure()
            plt.xlabel(f'feature {i}')
            plt.ylabel(f'feature {j}')
            for c in classes:
                plt.scatter(data_classes[c][i, :], data_classes[c][j, :], label = c)

            plt.legend()
            plt.savefig(f"{scatter_path}/scatter_feature{i}_feature{j}.jpg")
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

def bayes_error_plot(llr, correct_labels, title, svm_scores=False):
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
    plt.plot(p_vals, min_dcf, label='minDCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.title(title)
    plt.show()

def covarMat(X):
    mu = X.mean(axis=1)  # mean of the dataset
    mu = mu.reshape(mu.size, 1)
    # mu is suptracted from each column(sample) of X through broadcasting
    X_centered = X-mu

    covMat = np.dot(X_centered, X_centered.T) / X.shape[1]

    return covMat

def logreg_perf_plot(path, working_point, title, training_working_point=(0.5, 1, 10)):
    DTR, LTR = load_data('dataset/Train.txt')
    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    labels = np.concatenate(labels_folds)

    _lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    plot_dict = {}
    for apply_znorm in [True, False]:
        for pca_dim in [-1, 9, 8, 7, 6]:
            plot_dict[(apply_znorm, pca_dim)] = []
            for l in _lambdas:
                #scores_path = f"{path}/scores_linearlogreg_effpr_{to_effective_prior(training_working_point)}_znorm_{apply_znorm}_pcadim_{pca_dim}_lambda_{l}.npy"
                scores_path = f"{path}/scores_quadrlogreg_effpr_{to_effective_prior(training_working_point)}_znorm_{apply_znorm}_pcadim_{pca_dim}_lambda_{l}.npy"
                scores = np.load(scores_path)

                #minDCF.append(minimum_bayes_risk(scores, labels, working_point, svm_scores=False))
                plot_dict[(apply_znorm, pca_dim)].append(minimum_bayes_risk(scores.reshape(scores.size,), labels, working_point, svm_scores=False))

            print(f"Done {(apply_znorm, pca_dim)}")

    plt.figure()
    plt.title(title)
    plt.xlabel("lambda")
    plt.ylabel("minDCF")
    plt.xscale('log')

    for k in plot_dict.keys():
        if k[1] == -1:
            legend = f"QuadrLogReg: no PCA, Z-norm={k[0]}"
        else:
            legend = f"QuadrLogReg: PCA_dim={k[1]}, Z-norm={k[0]}"

        plt.plot(_lambdas, plot_dict[k], label = legend)
    plt.legend()
    plt.show()

def linearSVM_perf_plot(path, working_point, title, training_working_point=(0.5, 1, 10)):
    DTR, LTR = load_data('dataset/Train.txt')
    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    labels = np.concatenate(labels_folds)

    c = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    plot_dict = {}

    for apply_znorm in [False, True]:#, False]:
        for pca_dim in [-1, 6, 7]:#, 8, 7, 6]:
            plot_dict[(apply_znorm, pca_dim)] = []
            for l in c:
                scores_path = f"{path}/scores_linearSVM_effpr_{to_effective_prior(training_working_point)}_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{l}.npy"
                scores = np.load(scores_path)

                plot_dict[(apply_znorm, pca_dim)].append(minimum_bayes_risk(scores.reshape(scores.size,), labels, working_point, svm_scores=True))

            print(f"Done {(apply_znorm, pca_dim)}")

    plt.figure()
    plt.title(title)
    plt.xlabel("lambda")
    plt.ylabel("minDCF")
    plt.xscale('log')

    for k in plot_dict.keys():
        if k[1] == -1:
            legend = f"LinearSVM: no PCA, Z-norm={k[0]}"
        else:
            legend = f"LinearSVM: PCA_dim={k[1]}, Z-norm={k[0]}"

        plt.plot(c, plot_dict[k], label = legend)
    plt.legend()
    plt.show()

def polySVM_perf_plot(path, working_point, title, training_working_point=(0.5, 1, 10)):
    DTR, LTR = load_data('dataset/Train.txt')
    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    labels = np.concatenate(labels_folds)

    c = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    plot_dict = {}

    for c_poly_hyperp in [0, 1]:#[0, 1]:
        for apply_znorm in [True, False]:
            for pca_dim in [-1, 6]:
                plot_dict[(apply_znorm, c_poly_hyperp, pca_dim)] = []
                for l in c:
                    scores_path = f"{path}/scores_polySVM_effpr_{to_effective_prior(training_working_point)}_degree_2_const_{c_poly_hyperp}_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{l}.npy"
                    scores = np.load(scores_path)

                    plot_dict[(apply_znorm, c_poly_hyperp, pca_dim)].append(minimum_bayes_risk(scores.reshape(scores.size,), labels, working_point, svm_scores=True))

                print(f"Done {(apply_znorm, c_poly_hyperp, pca_dim)}")

    plt.figure()
    plt.title(title)
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale('log')

    for k in plot_dict.keys():
        if k[2] == -1:
            legend = f"Poly(2)SVM: c_poly={k[1]}, Z-norm={k[0]}, no PCA"
        else:
            legend = f"Poly(2)SVM: c_poly={k[1]}, Z-norm={k[0]}, PCA_dim={k[2]}"

        plt.plot(c, plot_dict[k], label = legend)
    plt.legend()
    plt.show()

def poly_third_SVM_perf_plot(path, working_point, title, training_working_point=(0.5, 1, 10)):
    DTR, LTR = load_data('dataset/Train.txt')
    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    labels = np.concatenate(labels_folds)

    c = [1e-2, 1e-1, 1e0]
    plot_dict = {}

    for c_poly_hyperp in [1]:#[0, 1]:
        for apply_znorm in [False]:
            for pca_dim in [6]:
                plot_dict[(apply_znorm, c_poly_hyperp, pca_dim)] = []
                for l in c:
                    scores_path = f"{path}/scores_polySVM_effpr_{to_effective_prior(training_working_point)}_degree_3_const_{c_poly_hyperp}_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{l}.npy"
                    scores = np.load(scores_path)

                    plot_dict[(apply_znorm, c_poly_hyperp, pca_dim)].append(minimum_bayes_risk(scores.reshape(scores.size,), labels, working_point, svm_scores=True))

                print(f"Done {(apply_znorm, c_poly_hyperp, pca_dim)}")

    plt.figure()
    plt.title(title)
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale('log')

    for k in plot_dict.keys():
        if k[2] == -1:
            legend = f"Poly(2)SVM: c_poly={k[1]}, Z-norm={k[0]}, no PCA"
        else:
            legend = f"Poly(2)SVM: c_poly={k[1]}, Z-norm={k[0]}, PCA_dim={k[2]}"

        plt.plot(c, plot_dict[k], label = legend)
    plt.legend()
    plt.show()

def rbfSVM_perf_plot(path, working_point, title, training_working_point=(0.5, 1, 10)):
    DTR, LTR = load_data('dataset/Train.txt')
    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    labels = np.concatenate(labels_folds)

    c = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    plot_dict = {}

    for gamma in [1e-5, 0.0001, 0.001]:
        for apply_znorm in [True, False]:
            for pca_dim in [-1, 6]:
                plot_dict[(apply_znorm, gamma, pca_dim)] = []
                for l in c:
                    scores_path = f"{path}/scores_rbfSVM_effpr_{to_effective_prior(training_working_point)}_gamma_{gamma}_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{l}.npy"
                    scores = np.load(scores_path)

                    plot_dict[(apply_znorm, gamma, pca_dim)].append(minimum_bayes_risk(scores.reshape(scores.size,), labels, working_point, svm_scores=True))

                print(f"Done {(apply_znorm, gamma, pca_dim)}")

    plt.figure()
    plt.title(title)
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale('log')

    for k in plot_dict.keys():
        if k[2] == -1:
            legend = f"RBF SVM: γ={k[1]}, Z-norm={k[0]}, no PCA"
        else:
            legend = f"RBFSVM: γ={k[1]}, Z-norm={k[0]}, PCA_dim={k[2]}"

        line_syle='solid'
        if k[1] == 1e-5:
            line_syle = 'dotted'
        if k[1] == 0.0001:
            line_syle = 'dashdot'

        plt.plot(c, plot_dict[k], label=legend, linestyle=line_syle)
    plt.legend()
    plt.show()


def logreg_perf_eval_plot(path, working_point, title, training_working_point=(0.5, 1, 10)):
    path_eval = "evaluation_logs/quadrLogReg_scores"

    DTR, LTR = load_data('dataset/Train.txt')
    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    labels = np.concatenate(labels_folds)

    DTE, LTE = load_data('dataset/Test.txt')

    _lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    plot_dict = {}
    for apply_znorm in [False]:
        for pca_dim in [9, 7, 6]:
            plot_dict[(apply_znorm, pca_dim, 'validation')] = []
            plot_dict[(apply_znorm, pca_dim, 'evaluation')] = []
            for l in _lambdas:
                # validation
                scores_path = f"{path}/scores_quadrlogreg_effpr_{to_effective_prior(training_working_point)}_znorm_{apply_znorm}_pcadim_{pca_dim}_lambda_{l}.npy"
                scores = np.load(scores_path)

                plot_dict[(apply_znorm, pca_dim, 'validation')].append(minimum_bayes_risk(scores.reshape(scores.size,), labels, working_point, svm_scores=False))

                #evaluation
                eval_scores_path = f"{path_eval}/eval_scores_quadrlogreg_effpr_{to_effective_prior(training_working_point)}_znorm_{apply_znorm}_pcadim_{pca_dim}_lambda_{l}.npy"
                eval_scores = np.load(eval_scores_path)

                plot_dict[(apply_znorm, pca_dim, 'evaluation')].append(minimum_bayes_risk(eval_scores.reshape(eval_scores.size,), LTE, working_point, svm_scores=False))

            print(f"Done {(apply_znorm, pca_dim)}")

    colors = {6: 'steelblue', 7: 'orange', 9: 'limegreen'}

    plt.figure()
    plt.title(title)
    plt.xlabel("lambda")
    plt.ylabel("minDCF")
    plt.xscale('log')

    for apply_znorm in [False]:
        for pca_dim in [9, 7, 6]:
            if pca_dim == -1:
                legend = f"QuadrLogReg: no PCA, Z-norm={apply_znorm}"
            else:
                legend = f"QuadrLogReg: PCA_dim={pca_dim}, Z-norm={apply_znorm}"
    
            #if k[2] == 'validation':
            #    legend += 'validation'
            #else:
            #    legend += 'evaluation'
    
            plt.plot(_lambdas, plot_dict[(apply_znorm, pca_dim, 'validation')], label = legend+'[validation]', linestyle='dashed', color=colors[pca_dim])
            plt.plot(_lambdas, plot_dict[(apply_znorm, pca_dim, 'evaluation')], label = legend+'[evaluation]', color=colors[pca_dim])
    plt.legend()
    plt.show()

def polySVM_perf_plot_eval(path, working_point, title, training_working_point=(0.5, 1, 10)):
    path_eval = "evaluation_logs/svm_scores"

    DTR, LTR = load_data('dataset/Train.txt')
    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    labels = np.concatenate(labels_folds)

    DTE, LTE = load_data('dataset/Test.txt')

    c = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    plot_dict = {}

#    for c_poly_hyperp in [0, 1]:#[0, 1]:
#        for apply_znorm in [True, False]:
#            for pca_dim in [-1, 6]:
    for c_poly_hyperp, apply_znorm, pca_dim in [(1, False, 6), (1, True, -1)]:
        plot_dict[(apply_znorm, c_poly_hyperp, pca_dim, 'validation')] = []
        plot_dict[(apply_znorm, c_poly_hyperp, pca_dim, 'evaluation')] = []
        for l in c:
            scores_path = f"{path}/scores_polySVM_effpr_{to_effective_prior(training_working_point)}_degree_2_const_{c_poly_hyperp}_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{l}.npy"
            scores = np.load(scores_path)

            plot_dict[(apply_znorm, c_poly_hyperp, pca_dim, 'validation')].append(minimum_bayes_risk(scores.reshape(scores.size,), labels, working_point, svm_scores=True))

            eval_scores_path = f"{path_eval}/scores_polySVM_effpr_{to_effective_prior(training_working_point)}_degree_2_const_{c_poly_hyperp}_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{l}.npy"
            eval_scores = np.load(eval_scores_path)

            plot_dict[(apply_znorm, c_poly_hyperp, pca_dim, 'evaluation')].append(minimum_bayes_risk(eval_scores.reshape(eval_scores.size,), LTE, working_point, svm_scores=True))

        print(f"Done {(apply_znorm, c_poly_hyperp, pca_dim)}")

    colors = {6: 'steelblue', -1: 'orange'}

    plt.figure()
    plt.title(title)
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.xscale('log')

    for c_poly_hyperp, apply_znorm, pca_dim in [(1, False, 6), (1, True, -1)]:
        if pca_dim == -1:
            legend = f"Poly(2)SVM: c_poly={c_poly_hyperp}, Z-norm={apply_znorm}, no PCA"
        else:
            legend = f"Poly(2)SVM: c_poly={c_poly_hyperp}, Z-norm={apply_znorm}, PCA_dim={pca_dim}"

        plt.plot(c, plot_dict[(apply_znorm, c_poly_hyperp, pca_dim, 'validation')], label = legend+'[validation]', linestyle='dashed', color=colors[pca_dim])
        plt.plot(c, plot_dict[(apply_znorm, c_poly_hyperp, pca_dim, 'evaluation')], label = legend+'[evaluation]', color=colors[pca_dim])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    logreg_perf_plot("logreg/results", (0.5, 1, 10), f"Quadratic LogReg @ effPrior = {to_effective_prior((0.5, 1, 10))}")
    logreg_perf_plot("logreg/results", (0.5, 1, 1), f"Quadratic LogReg @ effPrior = {to_effective_prior((0.5, 1, 1))}")
    logreg_perf_plot("logreg/results", (0.9, 1, 1), f"Quadratic LogReg @ effPrior = {to_effective_prior((0.9, 1, 1))}")

    #logreg_perf_plot("logreg/results", (0.5, 1, 10), "Quadr LogReg")
    
#    linearSVM_perf_plot("svm/results_linear", (0.5, 1, 10), f"Linear SVM @ effPrior = {to_effective_prior((0.5, 1, 10))}")
#    linearSVM_perf_plot("svm/results_linear", (0.5, 1, 1), f"Linear SVM @ effPrior = {to_effective_prior((0.5, 1, 1))}")
#    linearSVM_perf_plot("svm/results_linear", (0.9, 1, 1), f"Linear SVM @ effPrior = {to_effective_prior((0.9, 1, 1))}")
    
#    polySVM_perf_plot("svm/results_poly", (0.5, 1, 10), f"Poly(2) SVM @ effPrior = {to_effective_prior((0.5, 1, 10))}")
#    polySVM_perf_plot("svm/results_poly", (0.5, 1, 1), f"Poly(2) SVM @ effPrior = {to_effective_prior((0.5, 1, 1))}")
#    polySVM_perf_plot("svm/results_poly", (0.9, 1, 1), f"Poly(2) SVM @ effPrior = {to_effective_prior((0.9, 1, 1))}")

#    poly_third_SVM_perf_plot("svm/results_poly_3rd", (0.5, 1, 10), f"Poly(3) SVM @ effPrior = {to_effective_prior((0.5, 1, 10))}")
#    poly_third_SVM_perf_plot("svm/results_poly_3rd", (0.5, 1, 1), f"Poly(3) SVM @ effPrior = {to_effective_prior((0.5, 1, 1))}")
#    poly_third_SVM_perf_plot("svm/results_poly_3rd", (0.9, 1, 1), f"Poly(3) SVM @ effPrior = {to_effective_prior((0.9, 1, 1))}")

    #poly_third_SVM_perf_plot("svm/results_poly_3rd", (0.5, 1, 10), "Poly(3) SVM")

#    rbfSVM_perf_plot("svm/results_rbf", (0.5, 1, 10), f"RBF SVM @ effPrior = {to_effective_prior((0.5, 1, 10))}")
#    rbfSVM_perf_plot("svm/results_rbf", (0.5, 1, 1), f"RBF SVM @ effPrior = {to_effective_prior((0.5, 1, 1))}")
#    rbfSVM_perf_plot("svm/results_rbf", (0.9, 1, 1), f"RBF SVM @ effPrior = {to_effective_prior((0.9, 1, 1))}")

    #poly_third_SVM_perf_plot("svm/results_poly_3rd", (0.5, 1, 10), "Poly(3) SVM")

#    logreg_perf_eval_plot("logreg/results", (0.5, 1, 10), f"Quadratic LogReg @ effPrior = {to_effective_prior((0.5, 1, 10))}")
#    logreg_perf_eval_plot("logreg/results", (0.5, 1, 1), f"Quadratic LogReg @ effPrior = {to_effective_prior((0.5, 1, 1))}")
#    logreg_perf_eval_plot("logreg/results", (0.9, 1, 1), f"Quadratic LogReg @ effPrior = {to_effective_prior((0.9, 1, 1))}")
#
#    polySVM_perf_plot_eval("svm/results_poly", (0.5, 1, 10), f"Poly(2) SVM @ effPrior = {to_effective_prior((0.5, 1, 10))}")
#    polySVM_perf_plot_eval("svm/results_poly", (0.5, 1, 1), f"Poly(2) SVM @ effPrior = {to_effective_prior((0.5, 1, 1))}")
#    polySVM_perf_plot_eval("svm/results_poly", (0.9, 1, 1), f"Poly(2) SVM @ effPrior = {to_effective_prior((0.9, 1, 1))}")

    DTR, LTR = load_data('dataset/Train.txt')
    folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    labels = np.concatenate(labels_folds)
    working_point = (0.5, 1, 10)

    DTE, LTE = load_data('dataset/Test.txt')

    #path = "gmm/results"
    eval_path = "evaluation_logs/svm_scores"
    pca_dim = -1
    #target_mode = 'tied'
    #nontarget_mode = 'diag'
    #num_comps_target = 2
    #num_comps_nontarget = 4

    znorm = True
    c = 0.1
    const = 1

    #scores_path = f"{eval_path}/eval_scores_quadrlogreg_effpr_0.09090909090909091_znorm_{znorm}_pcadim_{pca_dim}_lambda_{l}.npy"
    scores_path = f"{eval_path}/scores_polySVM_effpr_0.09090909090909091_degree_2_const_{const}_kv_0_znorm_{znorm}_pcadim_{pca_dim}_c_{c}.npy"
    #scores_path = f"{eval_path}/eval_scores_gmm_effpr_0.09090909090909091_targetmode_{target_mode}_nontargetmode_{nontarget_mode}_targetNumComps_{num_comps_target}_nontargetNumComps_{num_comps_nontarget}_pcadim_{pca_dim}.npy"
    
    scores = np.load(scores_path)

    #minDCF = minimum_bayes_risk(scores.reshape(scores.size,), labels, working_point, svm_scores=False)
    minDCF = minimum_bayes_risk(scores.reshape(scores.size,), LTE, working_point, svm_scores=True)
    print(minDCF)
