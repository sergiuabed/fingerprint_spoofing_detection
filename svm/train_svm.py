import sys
sys.path.append('.')
sys.path.append('./utils')

import os
import numpy as np
from utils.dimensionality_reduction import PCA_matrix
from utils.data_utils import load_data, to_effective_prior, split_k_folds, get_empirical_prior, expand_features
from svm import svm_k_fold_train, kernel_rbf_wrap, kernel_polynomial_wrap
from utils.plot_utils import bayes_error_plot
from utils.measurements import *
import copy

K = 5
TRAIN_SEED = 22

def train_svm():
    DTR, LTR = load_data('dataset/Train.txt')
    LTR[LTR == 0] = -1
    working_point = (0.5, 1, 10)

    #s, P = PCA_matrix(DTR, 6)

    #DP = np.dot(P.T, DTR)

    eff_prior = to_effective_prior(working_point)
    
    #folds, labels_folds = split_k_folds(DP, LTR, K, TRAIN_SEED)
    folds, labels_folds = split_k_folds(DTR, LTR, K, TRAIN_SEED)
    labels_sh = np.concatenate(labels_folds)
    labels_sh_copy = copy.deepcopy(labels_sh)
    labels_sh_copy[labels_sh_copy == -1] = 0

    # NOW PASSING DTR INSTEAD OF DP
#    kv = 1.0
#    c = 1.0
#    gamma = 1e-3 # for kernel_rbf
#    degree = 2 # for kernel_polynomial
#    kernel = kernel_polynomial_wrap(c, degree, kv) #kernel_rbf_wrap(gamma, kv)

    path = os.getcwd() + '/svm/results_linear'
    if not os.path.isdir(path):
        os.mkdir(path)

    with open(f'{path}/log.txt', 'w') as f:
        kernel = None
        for kv in [0]:
            for apply_znorm in [True, False]:
                for pca_dim in [-1, 7, 6]:#[-1, 9, 8, 7, 6]:
                    for c in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]:
                        print(f"--------------------kv: {kv}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")
                        f.write(f"--------------------kv: {kv}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")

                        params, scores = svm_k_fold_train(DTR, LTR, K, TRAIN_SEED, kv, c, apply_znorm, pca_dim, eff_prior, kernel=kernel)

                    #    print("Scores: ")
                    #    print(scores[scores < 0])

                        pl = binary_optimal_bayes_decision(scores, working_point, svm_scores=True)
                        cm = get_confusion_matrix(pl.reshape((pl.size,)), labels_sh_copy, 2) # "labels_sh_copy" has labels 0 and 1 (labels_sh has -1 and 1)

                        print(cm)

                        _, dcfn = bayes_risk(cm, working_point)
                        mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), labels_sh_copy, working_point, svm_scores=True)

                        print(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n")
                        f.write(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n\n")

                        # save scores and params
                        np.save(f"{path}/scores_linearSVM_effpr_{eff_prior}_kv_{kv}_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}", scores)
                        np.save(f"{path}/params_linearSVM_effpr_{eff_prior}_kv_{kv}_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}", params)

                        #bayes_error_plot(scores.reshape(scores.size,), labels_sh_copy, svm_scores=True)


def train_svm_poly():
    DTR, LTR = load_data('dataset/Train.txt')
    LTR[LTR == 0] = -1
    working_point = (0.5, 1, 10)

    #s, P = PCA_matrix(DTR, 6)

    #DP = np.dot(P.T, DTR)

    eff_prior = to_effective_prior(working_point)
    
    #folds, labels_folds = split_k_folds(DP, LTR, K, TRAIN_SEED)
    folds, labels_folds = split_k_folds(DTR, LTR, K, TRAIN_SEED)
    labels_sh = np.concatenate(labels_folds)
    labels_sh_copy = copy.deepcopy(labels_sh)
    labels_sh_copy[labels_sh_copy == -1] = 0

    # NOW PASSING DTR INSTEAD OF DP
#    kv = 1.0
#    c = 1.0
#    gamma = 1e-3 # for kernel_rbf
#    degree = 2 # for kernel_polynomial
#    kernel = kernel_polynomial_wrap(c, degree, kv) #kernel_rbf_wrap(gamma, kv)

    path = os.getcwd() + '/svm/results_poly_3rd'
    if not os.path.isdir(path):
        os.mkdir(path)

    with open(f'{path}/log.txt', 'w') as f:
        for const in [1]:
            for d in [3]:#, 2]:
                for kv in [0]:
                    for apply_znorm in [False]:#[True, False]:
                        for pca_dim in [6]:#[-1, 6]:#[-1, 9, 8, 7, 6]:
                            for c in [1e-2, 1e-1, 1.0]:#[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]:
                                print(f"--------------------degree: {d} const: {const}  kv: {kv}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")
                                f.write(f"--------------------degree: {d} const: {const}  kv: {kv}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")

                                kernel = kernel_polynomial_wrap(const, d, kv)

                                params, scores = svm_k_fold_train(DTR, LTR, K, TRAIN_SEED, kv, c, apply_znorm, pca_dim, eff_prior, kernel=kernel)

                            #    print("Scores: ")
                            #    print(scores[scores < 0])

                                pl = binary_optimal_bayes_decision(scores, working_point, svm_scores=True)
                                cm = get_confusion_matrix(pl.reshape((pl.size,)), labels_sh_copy, 2) # "labels_sh_copy" has labels 0 and 1 (labels_sh has -1 and 1)

                                print(cm)

                                _, dcfn = bayes_risk(cm, working_point)
                                mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), labels_sh_copy, working_point, svm_scores=True)

                                print(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n")
                                f.write(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n\n")

                                # save scores and params
                                np.save(f"{path}/scores_polySVM_effpr_{eff_prior}_degree_{d}_const_{const}_kv_{kv}_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}", scores)
                                np.save(f"{path}/params_polySVM_effpr_{eff_prior}_degree_{d}_const_{const}_kv_{kv}_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}", params)



def train_svm_rbf():
    DTR, LTR = load_data('dataset/Train.txt')
    LTR[LTR == 0] = -1
    working_point = (0.5, 1, 10)

    #s, P = PCA_matrix(DTR, 6)

    #DP = np.dot(P.T, DTR)

    eff_prior = to_effective_prior(working_point)
    
    #folds, labels_folds = split_k_folds(DP, LTR, K, TRAIN_SEED)
    folds, labels_folds = split_k_folds(DTR, LTR, K, TRAIN_SEED)
    labels_sh = np.concatenate(labels_folds)
    labels_sh_copy = copy.deepcopy(labels_sh)
    labels_sh_copy[labels_sh_copy == -1] = 0

    # NOW PASSING DTR INSTEAD OF DP
#    kv = 1.0
#    c = 1.0
#    gamma = 1e-3 # for kernel_rbf
#    degree = 2 # for kernel_polynomial
#    kernel = kernel_polynomial_wrap(c, degree, kv) #kernel_rbf_wrap(gamma, kv)

    path = os.getcwd() + '/svm/results_rbf'
    if not os.path.isdir(path):
        os.mkdir(path)

    with open(f'{path}/log.txt', 'a') as f:
            for gamma in [1e-3]:#[1e-5, 1e-4, 1e-3]:
                for kv in [0]:
                    for apply_znorm in [True, False]:
                        for pca_dim in [-1, 6]:#[-1, 9, 8, 7, 6]:
                            for c in [1.0]: #[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]:
                                print(f"--------------------gamma: {gamma}  kv: {kv}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")
                                f.write(f"--------------------gamma: {gamma}  kv: {kv}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")

                                kernel = kernel_rbf_wrap(gamma, kv)

                                params, scores = svm_k_fold_train(DTR, LTR, K, TRAIN_SEED, kv, c, apply_znorm, pca_dim, eff_prior, kernel=kernel)

                            #    print("Scores: ")
                            #    print(scores[scores < 0])

                                pl = binary_optimal_bayes_decision(scores, working_point, svm_scores=True)
                                cm = get_confusion_matrix(pl.reshape((pl.size,)), labels_sh_copy, 2) # "labels_sh_copy" has labels 0 and 1 (labels_sh has -1 and 1)

                                print(cm)

                                _, dcfn = bayes_risk(cm, working_point)
                                mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), labels_sh_copy, working_point, svm_scores=True)

                                print(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n")
                                f.write(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n\n")

                                # save scores and params
                                np.save(f"{path}/scores_rbfSVM_effpr_{eff_prior}_gamma_{gamma}_kv_{kv}_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}", scores)
                                np.save(f"{path}/params_rbfSVM_effpr_{eff_prior}_gamma_{gamma}_kv_{kv}_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}", params)




if __name__ == '__main__':
#    DTR, LTR = load_data('dataset/Train.txt')
#    nr_samples, priors = get_empirical_prior(LTR)
#
#    print(f"Nr samples: {nr_samples}")
#    print(f"Priors: {priors}")
    
#    train_svm()
#    train_svm_poly()
    train_svm_rbf()


#    eff_prior = to_effective_prior((0.5, 1, 10))
#    print(eff_prior)