import sys
sys.path.append('.')
sys.path.append('./utils')

import os
import pickle
import numpy as np
from utils.dimensionality_reduction import PCA_matrix
from utils.data_utils import load_data, to_effective_prior, split_k_folds
from gmm import gmm_k_fold_train
from utils.plot_utils import bayes_error_plot
from utils.measurements import *

K = 5
TRAIN_SEED = 22

def train_gmm():
    DTR, LTR = load_data('dataset/Train.txt')
    working_point = (0.5, 1, 10)

    #s, P = PCA_matrix(DTR, 6)

    #DP = np.dot(P.T, DTR)

    eff_prior = to_effective_prior(working_point)
    
    folds, labels_folds = split_k_folds(DTR, LTR, K, TRAIN_SEED)
    labels_sh = np.concatenate(labels_folds)

    num_comps_target = 2
    num_comps_nontarget = 8
#    mode_target = 'diag'#'default'
#    mode_nontarget = 'tied'#'default'

    path = os.getcwd() + '/gmm/results'
    if not os.path.isdir(path):
        os.mkdir(path)

    with open(f'{path}/log.txt', 'a') as f:
        for pca_dim in [6, -1]:
            for mode_target in ['default', 'diag', 'tied']:
                for mode_nontarget in ['default', 'diag', 'tied']:

                    params, scores_dict = gmm_k_fold_train(DTR, LTR, K, TRAIN_SEED, pca_dim, num_comps_target, num_comps_nontarget, mode_target, mode_nontarget)

                    for i in range(len(params[1])): #target_gmm in gmm_params[1]:
                        for j in range(len(params[0])): #nontarget_gmm in gmm_params[0]:
                            if i == 0 and j == 0:
                                continue

                            print(f"----------------PCA_dim: {pca_dim}  target_mode: {mode_target} nontarget_mode: {mode_nontarget}  targetNumComps: {2**i}  nontargetNumComps: {2**j}----------------\n")
                            f.write(f"----------------PCA_dim: {pca_dim}  target_mode: {mode_target} nontarget_mode: {mode_nontarget}  targetNumComps: {2**i}  nontargetNumComps: {2**j}----------------\n")

                            scores = scores_dict[(2**j, 2**i)]

                            pl = binary_optimal_bayes_decision(scores, working_point)
                            cm = get_confusion_matrix(pl.reshape((pl.size,)), labels_sh, 2)

                            print(cm)

                            _, dcfn = bayes_risk(cm, working_point)
                            mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), labels_sh, working_point)

                            print(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n")
                            f.write(f"Minimum DCF: {mindcf} Actual DCF: {dcfn}\n\n")

                            # save scores and params
                            np.save(f"{path}/scores_gmm_effpr_{eff_prior}_targetmode_{mode_target}_nontargetmode_{mode_nontarget}_targetNumComps_{2**i}_nontargetNumComps_{2**j}_pcadim_{pca_dim}", scores)

                    with open(f"{path}/params_gmm_effpr_{eff_prior}_targetmode_{mode_target}_nontargetmode_{mode_nontarget}_pcaDim_{pca_dim}", 'wb') as g:
                        pickle.dump(params, g)

                    #bayes_error_plot(scores.reshape(scores.size,), labels_sh)

if __name__ == '__main__':
    train_gmm()