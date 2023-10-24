import sys
sys.path.append('.')
sys.path.append('./utils')

import os
import pickle
import numpy as np
from utils.dimensionality_reduction import PCA_matrix
from utils.data_utils import load_data, to_effective_prior, split_k_folds
from mvg import MVG_k_fold_train
from utils.plot_utils import bayes_error_plot
from utils.measurements import *

K = 5
TRAIN_SEED = 22

def train_mvg():
    DTR, LTR = load_data('dataset/Train.txt')
    working_point = (0.5, 1, 10)

    s, P = PCA_matrix(DTR, 10)

    DP = DTR#np.dot(P.T, DTR)

    eff_prior = to_effective_prior(working_point)
    
    folds, labels_folds = split_k_folds(DP, LTR, K, TRAIN_SEED)
    labels_sh = np.concatenate(labels_folds)

    path = os.getcwd() + '/mvg/results'
    if not os.path.isdir(path):
        os.mkdir(path)

    with open(f'{path}/log.txt', 'w') as f:
        for mode in ['default', 'diag', 'tied']:
            for pca_dim in [None, 9, 8, 7, 6]:
                print(f"--------------------mode: {mode}     PCA_dim: {pca_dim}--------------------\n")
                f.write(f"--------------------mode: {mode}     PCA_dim: {pca_dim}--------------------\n")

                if pca_dim is None:
                    DP = DTR
                else:
                    s, P = PCA_matrix(DTR, pca_dim)
                    DP = np.dot(P.T, DTR)   #project on lower space

                params, scores = MVG_k_fold_train(DP, LTR, K, TRAIN_SEED, mode)

                pl = binary_optimal_bayes_decision(scores, working_point)
                cm = get_confusion_matrix(pl.reshape((pl.size,)), labels_sh, 2)

                print("Confusion matrix:")
                print(cm)

                _, dcfn = bayes_risk(cm, working_point)
                mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), labels_sh, working_point)

                print(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n")
                f.write(f"Minimum DCF: {mindcf} Actual DCF: {dcfn}\n\n")

                # save scores and params
                np.save(f"{path}/scores_mvg_effpr_{eff_prior}_mode_{mode}_pcadim_{pca_dim}", scores)

                with open(f"{path}/params_mvg_effpr_{eff_prior}_mode_{mode}_pcadim_{pca_dim}", 'wb') as g:
                    pickle.dump(params, g)

                #bayes_error_plot(scores.reshape(scores.size,), labels_sh)



if __name__ == '__main__':
    train_mvg()