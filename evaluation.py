from utils.measurements import minimum_bayes_risk, binary_optimal_bayes_decision, get_confusion_matrix, bayes_risk
from utils.data_utils import load_data, to_effective_prior, datasetMean, z_norm, expand_features
from utils.dimensionality_reduction import PCA_matrix
from mvg.mvg import MVG_classifier_wrap
from logreg.logreg import logreg_prior_weighted_obj_wrap, logistic_regression_wrap
import os
import pickle
import numpy as np
import scipy.optimize

def mvg_evaluation(DTE, LTE, DTR, LTR, params_path, working_point):
    target_eff_prior = to_effective_prior((0.5, 1, 10))
    for mode in ['default']:
        for pca_dim in [7]:
            mu_tr = datasetMean(DTR)

            if pca_dim == -1:
                DTE -= mu_tr
            else:
                s, P = PCA_matrix(DTR, pca_dim)
                DTE_pr = np.dot(P.T, DTE - mu_tr)

            with open(f"{params_path}/params_mvg_effpr_{target_eff_prior}_mode_{mode}_pcadim_{pca_dim}", 'rb') as g:
                params = pickle.load(g)
            
            means = params[0]
            covariances = params[1]
            classifier = MVG_classifier_wrap(means, covariances)

            scores = classifier(DTE_pr)
            print(minimum_bayes_risk(scores.reshape((scores.size,)), LTE, working_point))
            #print(params)

def logreg_evaluation(DTE, LTE, DTR, LTR, working_point):
    '''
    Because I forgot to save the parameters during training for Logistic Regression,
    this function will also retrain the models and test them on the evaluation set.

    We will only retrain and test the quadratic logistic regression models, more specifically
    we will consider the 3 best PCA dimensions together with using or not z-normalization.

    Best configs: (pca_dim=6, znorm=False), (pca_dim=7, znorm=False), (pca_dim=9, znorm=False)
    '''
    expand_feat = True

    log_path = './evaluation_logs'
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    scores_path = f"{log_path}/quadrLogReg_scores"
    if not os.path.isdir(scores_path):
        os.mkdir(scores_path)

    target_eff_prior = to_effective_prior((0.5, 1, 10))
    # train on whole data
    with open(f'{log_path}/quadr_logreg_log.txt', 'w') as f:
        for apply_znorm in [False]: #[True, False]:
            for pca_dim in [-1, 7, 6]:
                for _lambda in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]:
                    print(f"--------------------apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  lambda: {_lambda}--------------------\n")
                    f.write(f"--------------------apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  lambda: {_lambda}--------------------\n")

                    # z-normalization
                    if apply_znorm is True:
                        DTR_znorm, _, _ = z_norm(DTR)
                        
                        mu_tr = datasetMean(DTR)
                        sd_tr = np.std(DTR - mu_tr, axis=1).reshape((DTR.shape[0], 1))

                        DTE_znorm = z_norm(DTE, mu_tr, sd_tr)

                    else:
                        mu = datasetMean(DTR)
                        DTR_znorm = DTR - mu
                        DTE_znorm = DTE - mu

                    # pca
                    if pca_dim < 0:
                        DTR_proj = DTR_znorm
                        DTE_proj = DTE_znorm
                    else:
                        mu_train = datasetMean(DTR_znorm)
                        s, P = PCA_matrix(DTR_znorm, pca_dim)
                        DTR_proj = np.dot(P.T, DTR_znorm - mu_train)   #project on lower space
                        DTE_proj = np.dot(P.T, DTE_znorm - mu_train)

                    if expand_feat is True:
                        DTR_proj = expand_features(DTR_proj)
                        DTE_proj = expand_features(DTE_proj)

                    logreg_obj = logreg_prior_weighted_obj_wrap(DTR_proj, LTR, _lambda, target_eff_prior)
                    solution = scipy.optimize.fmin_l_bfgs_b(logreg_obj,  x0=np.zeros(DTR_proj.shape[0] + 1), approx_grad=True)

                    w = np.array(solution[0][0:-1]).reshape(len(solution[0][0:-1]), 1)
                    b = solution[0][-1]

                    params = (w, b)

                    logreg_classifier = logistic_regression_wrap(w, b)

                    scores = logreg_classifier(DTE_proj) - np.log(target_eff_prior/(1-target_eff_prior))

                    pl = binary_optimal_bayes_decision(scores, working_point)
                    cm = get_confusion_matrix(pl.reshape((pl.size,)), LTE, 2)

                    print("Confusion matrix:")
                    print(cm)

                    _, dcfn = bayes_risk(cm, working_point)
                    mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), LTE, working_point)

                    print(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n")
                    f.write(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n\n")

                    # save scores and params
                    np.save(f"{scores_path}/scores_quadrlogreg_effpr_{target_eff_prior}_znorm_{apply_znorm}_pcadim_{pca_dim}_lambda_{_lambda}", scores)

def svm_evaluation(DTE, LTE, DTR, LTR, params_path, working_point):
    '''
    Best configs quadr poly:    (degree: 2 const: 1  kv: 0  apply_znorm: False  PCA_dim: 6)
                                (degree: 2 const: 1  kv: 0  apply_znorm: True  PCA_dim: -1)
    
    Best configs rbf:   (gamma: 0.001  kv: 0  apply_znorm: False  PCA_dim: -1)
                        (gamma: 0.001  kv: 0  apply_znorm: False  PCA_dim: 6)
    '''

    # z-normalization
#    if apply_znorm is True:
#        train_folds_znorm, mu, sd = z_norm(train_folds)
#        test_fold_znorm, _, _ = z_norm(test_fold, mu, sd)
#    else:
#        mu = datasetMean(train_folds)
#        train_folds_znorm = train_folds - mu
#        test_fold_znorm = test_fold - mu
#
#    # pca
#    if pca_dim < 0:
#        train_folds_proj = train_folds_znorm
#        test_fold_proj = test_fold_znorm
#    else:
#        mu_train = datasetMean(train_folds_znorm)
#        s, P = PCA_matrix(train_folds_znorm, pca_dim)
#        train_folds_proj = np.dot(P.T, train_folds_znorm - mu_train)   #project on lower space
#        test_fold_proj = np.dot(P.T, test_fold_znorm - mu_train)
    
    #TO BE COMPLETED

if __name__ == '__main__':
    DTR, LTR = load_data('./dataset/Train.txt')
    DTE, LTE = load_data('./dataset/Test.txt')

    #mvg_evaluation(DTE, LTE, DTR, LTR, 'mvg/results', (0.5, 1, 10))
    #logreg_evaluation(DTE, LTE, DTR, LTR, (0.5, 1, 10))
    