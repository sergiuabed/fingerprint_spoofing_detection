import sys
sys.path.append('.')
sys.path.append('./utils')

import os
import numpy as np
from utils.dimensionality_reduction import PCA_matrix
from utils.data_utils import load_data, to_effective_prior, split_k_folds, get_empirical_prior, expand_features
from logreg import logreg_k_fold_train
from utils.plot_utils import bayes_error_plot
from utils.measurements import *

K = 5
TRAIN_SEED = 22

def train_logreg():
    DTR, LTR = load_data('dataset/Train.txt')
    working_point = (0.5, 1, 10)

    #s, P = PCA_matrix(DTR, 6)

    #DP = np.dot(P.T, DTR)
    #DP = expand_features(DTR)
    #DP = expand_features(DP)

    eff_prior = to_effective_prior(working_point)
    
    #folds, labels_folds = split_k_folds(DP, LTR, K, TRAIN_SEED)
    folds, labels_folds = split_k_folds(DTR, LTR, K, TRAIN_SEED)
    labels_sh = np.concatenate(labels_folds)

    path = os.getcwd() + '/logreg/results'
    if not os.path.isdir(path):
        os.mkdir(path)

    expand_feat = True
    with open(f'{path}/log.txt', 'w') as f:
        for apply_znorm in [True, False]:
            for pca_dim in [-1, 9, 8, 7, 6]:
                for _lambda in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]:
                    print(f"--------------------apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  lambda: {_lambda}--------------------\n")
                    f.write(f"--------------------apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  lambda: {_lambda}--------------------\n")

#                    if apply_znorm == True:
#                        asdòasd
#                    
#                    if pca_dim is None:
#                        DP = DTR
#                    else:
#                        s, P = PCA_matrix(DTR, pca_dim)
#                        DP = np.dot(P.T, DTR) #project on lower space

                    # NOW PASSING DTR INSTEAD OF DP
                    #params, scores = logreg_k_fold_train(DP, LTR, K, TRAIN_SEED, _lambda, eff_prior)
                    params, scores = logreg_k_fold_train(DTR, LTR, K, TRAIN_SEED, _lambda, expand_feat, pca_dim, apply_znorm, eff_prior)

                    pl = binary_optimal_bayes_decision(scores, working_point)
                    cm = get_confusion_matrix(pl.reshape((pl.size,)), labels_sh, 2)

                    print("Confusion matrix:")
                    print(cm)

                    _, dcfn = bayes_risk(cm, working_point)
                    mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), labels_sh, working_point)

                    print(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n")
                    f.write(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n\n")

                    # save scores and params
                    np.save(f"{path}/scores_quadrlogreg_effpr_{eff_prior}_znorm_{apply_znorm}_pcadim_{pca_dim}_lambda_{_lambda}", scores)

                    #bayes_error_plot(scores.reshape(scores.size,), labels_sh)

if __name__ == '__main__':
#    DTR, LTR = load_data('dataset/Train.txt')
#    nr_samples, priors = get_empirical_prior(LTR)
#
#    print(f"Nr samples: {nr_samples}")
#    print(f"Priors: {priors}")
    train_logreg()

#    eff_prior = to_effective_prior((0.5, 1, 10))
#    print(eff_prior)