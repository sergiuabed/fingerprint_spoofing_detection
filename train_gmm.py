import numpy as np
from dimensionality_reduction import PCA_matrix
from data_utils import load_data, to_effective_prior, split_k_folds
from gmm import gmm_k_fold_train
from plot_utils import bayes_error_plot
from measurements import *

K = 5
TRAIN_SEED = 22

def train_gmm():
    DTR, LTR = load_data('dataset/Train.txt')
    working_point = (0.5, 1, 10)

    s, P = PCA_matrix(DTR, 6)

    DP = np.dot(P.T, DTR)

    eff_prior = to_effective_prior(working_point)
    
    folds, labels_folds = split_k_folds(DP, LTR, K, TRAIN_SEED)
    labels_sh = np.concatenate(labels_folds)

    num_comps_target = 1
    num_comps_nontarget = 2
    mode_target = 'diag'#'default'
    mode_nontarget = 'tied'#'default'

    # NOW PASSING DTR INSTEAD OF DP
    params, scores = gmm_k_fold_train(DP, LTR, K, TRAIN_SEED, num_comps_target, num_comps_nontarget, mode_target, mode_nontarget)
    
    pl = binary_optimal_bayes_decision(scores, working_point)
    cm = get_confusion_matrix(pl.reshape((pl.size,)), labels_sh, 2)

    print(cm)

    _, dcfn = bayes_risk(cm, working_point)
    mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), labels_sh, working_point)

    print(f"Minimum DCF: {mindcf}")
    print(f"Actual DCF: {dcfn}")
    bayes_error_plot(scores.reshape(scores.size,), labels_sh)

if __name__ == '__main__':
    train_gmm()