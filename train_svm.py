import numpy as np
from dimensionality_reduction import PCA_matrix
from data_utils import load_data, to_effective_prior, split_k_folds, get_empirical_prior, expand_features
from svm import svm_k_fold_train, kernel_rbf_wrap, kernel_polynomial_wrap
from plot_utils import bayes_error_plot
from measurements import *
import copy

K = 5
TRAIN_SEED = 22

def train_svm():
    DTR, LTR = load_data('dataset/Train.txt')
    LTR[LTR == 0] = -1
    working_point = (0.5, 1, 10)

    s, P = PCA_matrix(DTR, 6)

    DP = np.dot(P.T, DTR)

    eff_prior = to_effective_prior(working_point)
    
    folds, labels_folds = split_k_folds(DP, LTR, K, TRAIN_SEED)
    labels_sh = np.concatenate(labels_folds)
    labels_sh_copy = copy.deepcopy(labels_sh)
    labels_sh_copy[labels_sh_copy == -1] = 0

    # NOW PASSING DTR INSTEAD OF DP
    kv = 1.0
    c = 1.0
    gamma = 1e-3 # for kernel_rbf
    degree = 2 # for kernel_polynomial
    kernel = kernel_polynomial_wrap(c, degree, kv) #kernel_rbf_wrap(gamma, kv)
    params, scores = svm_k_fold_train(DP, LTR, K, TRAIN_SEED, kv, c, eff_prior, kernel=kernel)
    
    print("Scores: ")
    print(scores[scores < 0])

    pl = binary_optimal_bayes_decision(scores, working_point, svm_scores=True)
    cm = get_confusion_matrix(pl.reshape((pl.size,)), labels_sh_copy, 2) # "labels_sh_copy" has labels 0 and 1 (labels_sh has -1 and 1)

    print(cm)

    _, dcfn = bayes_risk(cm, working_point)
    mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), labels_sh_copy, working_point, svm_scores=True)

    print(f"Minimum DCF: {mindcf}")
    print(f"Actual DCF: {dcfn}")
    bayes_error_plot(scores.reshape(scores.size,), labels_sh_copy, svm_scores=True)

if __name__ == '__main__':
#    DTR, LTR = load_data('dataset/Train.txt')
#    nr_samples, priors = get_empirical_prior(LTR)
#
#    print(f"Nr samples: {nr_samples}")
#    print(f"Priors: {priors}")
    train_svm()

#    eff_prior = to_effective_prior((0.5, 1, 10))
#    print(eff_prior)