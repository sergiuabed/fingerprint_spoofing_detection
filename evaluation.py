from utils.measurements import minimum_bayes_risk, binary_optimal_bayes_decision, get_confusion_matrix, bayes_risk
from utils.data_utils import load_data, to_effective_prior, datasetMean, z_norm, expand_features, split_k_folds
from utils.dimensionality_reduction import PCA_matrix
from mvg.mvg import MVG_classifier_wrap
from logreg.logreg import logreg_prior_weighted_obj_wrap, logistic_regression_wrap
from svm.svm import kernel_polynomial_wrap, kernel_rbf_wrap, svm_dual_classifier_wrap
from gmm.gmm import gmm_classifier_wrap
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
    with open(f'{log_path}/quadr_logreg_log.txt', 'a') as f:
        for apply_znorm in [False]: #[True, False]:
            for pca_dim in [9]:#, 7, 6]:
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
                    np.save(f"{scores_path}/eval_scores_quadrlogreg_effpr_{target_eff_prior}_znorm_{apply_znorm}_pcadim_{pca_dim}_lambda_{_lambda}", scores)

def svm_evaluation(DTE, LTE, DTR, LTR, working_point):
    '''
    Best configs quadr poly:    (degree: 2 const: 1  kv: 0  apply_znorm: False  PCA_dim: 6)
                                (degree: 2 const: 1  kv: 0  apply_znorm: True  PCA_dim: -1)
    
    Best configs rbf:   (gamma: 0.001  kv: 0  apply_znorm: False  PCA_dim: -1)
                        (gamma: 0.001  kv: 0  apply_znorm: False  PCA_dim: 6)
    '''

    log_path = './evaluation_logs'
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    scores_path = f"{log_path}/svm_scores"
    if not os.path.isdir(scores_path):
        os.mkdir(scores_path)

    ## split DTR in the same way as during training
    #folds, labels_folds = split_k_folds(DTR, LTR, 5, 22)
    #DTR = np.hstack(folds)
    #LTR = np.concatenate(labels_folds)

    LTR[LTR == 0] = -1
    #LTE[LTE == 0] = -1

    target_eff_prior = to_effective_prior((0.5, 1, 10))

    poly_kernel = kernel_polynomial_wrap(1, 2, 0)
    rbf_kernel = kernel_rbf_wrap(0.001, 0)

    with open(f'{log_path}/svm_applicationEffPrior_{to_effective_prior(working_point)}_log.txt', 'w') as f:
        for pca_dim, apply_znorm, kernel, kernel_type in [(6, False, poly_kernel, 'poly'), (-1, True, poly_kernel, 'poly'), (-1, False, rbf_kernel, 'rbf'), (6, False, rbf_kernel, 'rbf')]:
            for c in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]:
                if kernel_type == 'poly':
                    print(f"--------------------degree: {2} const: {1}  kv: {0}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")
                    f.write(f"--------------------degree: {2} const: {1}  kv: {0}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")
                else:
                    print(f"--------------------gamma: {0.001}  kv: {0}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")
                    f.write(f"--------------------gamma: {0.001}  kv: {0}  apply_znorm: {apply_znorm}  PCA_dim: {pca_dim}  c: {c}--------------------\n")

                # z-normalization
                if apply_znorm is True:
                    DTR_znorm, mu_tr, sd_tr = z_norm(DTR)
                    DTE_znorm, _, _ = z_norm(DTE, mu_tr, sd_tr)
                else:
                    mu_tr = datasetMean(DTR)
                    DTR_znorm = DTR - mu_tr
                    DTE_znorm = DTE - mu_tr

                # pca
                if pca_dim < 0:
                    DTR_proj = DTR_znorm
                    DTE_proj = DTE_znorm
                else:
                    mu_tr = datasetMean(DTR_znorm)
                    s, P = PCA_matrix(DTR_znorm, pca_dim)
                    DTR_proj = np.dot(P.T, DTR_znorm - mu_tr)   #project on lower space
                    DTE_proj = np.dot(P.T, DTE_znorm - mu_tr)

                # HERE REPLACE WITH np.load(params_{}...)
                if kernel_type == 'poly':
                    _alphas = np.load(f"svm/results_poly/params_polySVM_effpr_{target_eff_prior}_degree_2_const_1_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}.npy")
                else:
                    _alphas = np.load(f"svm/results_rbf/params_rbfSVM_effpr_{target_eff_prior}_gamma_{0.001}_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}.npy")

                svm_dual_classifier, w_b1 = svm_dual_classifier_wrap(_alphas, DTR_proj, LTR, 0, kernel=kernel)
                scores = svm_dual_classifier(DTE_proj)

                pl = binary_optimal_bayes_decision(scores, working_point, svm_scores=True)
                cm = get_confusion_matrix(pl.reshape((pl.size,)), LTE, 2)

                print(cm)

                _, dcfn = bayes_risk(cm, working_point)
                mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), LTE, working_point, svm_scores=True)

                print(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n")
                f.write(f"Minimum DCF: {mindcf}   Actual DCF: {dcfn}\n\n")

                if kernel_type == 'poly':
                    np.save(f"{scores_path}/scores_polySVM_effpr_{to_effective_prior(working_point)}_degree_2_const_1_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}.npy", scores)
                else:
                    np.save(f"{scores_path}/scores_rbfSVM_effpr_{to_effective_prior(working_point)}_gamma_{0.001}_kv_0_znorm_{apply_znorm}_pcadim_{pca_dim}_c_{c}.npy", scores)

def gmm_evaluation(DTE, LTE, DTR, LTR, working_point, alt_working_point1, alt_working_point2):
    '''
    Best configs:   PCA_dim: 6  target_mode: default    nontarget_mode: default    targetNumComps: 1  nontargetNumComps: 8  0.2461  GOOD
                    PCA_dim: 6  target_mode: default    nontarget_mode: diag       targetNumComps: 1  nontargetNumComps: 8  0.2486
                    PCA_dim: 6  target_mode: diag       nontarget_mode: default    targetNumComps: 2  nontargetNumComps: 8  0.2449  GOOD
                    PCA_dim: 6  target_mode: diag       nontarget_mode: diag       targetNumComps: 2  nontargetNumComps: 8  0.2483
                    PCA_dim: 6  target_mode: tied       nontarget_mode: default    targetNumComps: 2  nontargetNumComps: 8  0.2462  GOOD
                    PCA_dim: 6  target_mode: tied       nontarget_mode: diag       targetNumComps: 2  nontargetNumComps: 8  0.2486
    '''

    log_path = './evaluation_logs'
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    scores_path = f"{log_path}/gmm_scores"
    if not os.path.isdir(scores_path):
        os.mkdir(scores_path)

#    params_path = 'gmm/results/params_gmm_effpr_0.09090909090909091_targetmode_default_nontargetmode_default_pcaDim_6'
#    with open(params_path, 'rb') as g:
#        params = pickle.load(g)
#
#    print(params[1][0])
#    print()
#    print(params[0][1])

    with open(f"{log_path}/gmm_log.txt", 'w') as f:
        pca_dim = 6
        #for mode_target in ['default', 'diag', 'tied']:
        #    for mode_nontarget in ['default', 'diag', 'tied']:
        for mode_target, mode_nontarget, targetNumComps, nontargetNumComps in [('default', 'default', 1, 8), ('diag', 'default', 2, 8), ('tied', 'default', 2, 8), ('default', 'diag', 1, 8), ('diag', 'diag', 2, 8), ('tied', 'diag', 2, 8)]:
            print(f"----------------PCA_dim: {pca_dim}  target_mode: {mode_target} nontarget_mode: {mode_nontarget}  targetNumComps: {targetNumComps}  nontargetNumComps: {nontargetNumComps}----------------\n")
            f.write(f"----------------PCA_dim: {pca_dim}  target_mode: {mode_target} nontarget_mode: {mode_nontarget}  targetNumComps: {targetNumComps}  nontargetNumComps: {nontargetNumComps}----------------\n")

            # apply pca
            mu_train = datasetMean(DTR)
            if pca_dim < 0:
                DTE_proj = DTE - mu_train
            else:
                s, P = PCA_matrix(DTR, pca_dim)
                DTE_proj = np.dot(P.T, DTE - mu_train)

            params_path = f"gmm/results/params_gmm_effpr_0.09090909090909091_targetmode_{mode_target}_nontargetmode_{mode_nontarget}_pcaDim_{pca_dim}"
            with open(params_path, 'rb') as g:
                gmm_params = pickle.load(g)

            gmm_target = gmm_params[1][int(np.log2(targetNumComps))]
            gmm_nontarget = gmm_params[0][int(np.log2(nontargetNumComps))]

            gmm_classifier = gmm_classifier_wrap({0: gmm_nontarget, 1: gmm_target})
            scores = gmm_classifier(DTE_proj)

            pl = binary_optimal_bayes_decision(scores, working_point)
            cm = get_confusion_matrix(pl.reshape((pl.size,)), LTE, 2)

            print(cm)

            _, dcfn = bayes_risk(cm, working_point)
            mindcf = minimum_bayes_risk(scores.reshape((scores.size,)), LTE, working_point)
            mindcf_alt1 = minimum_bayes_risk(scores.reshape((scores.size,)), LTE, alt_working_point1) #alt_working_point1
            mindcf_alt2 = minimum_bayes_risk(scores.reshape((scores.size,)), LTE, alt_working_point2)

            print(f"minDCF({to_effective_prior(working_point)}): {mindcf}   minDCF({to_effective_prior(alt_working_point1)}): {mindcf_alt1}   minDCF({to_effective_prior(alt_working_point2)}): {mindcf_alt2}\n")
            f.write(f"minDCF({to_effective_prior(working_point)}): {mindcf}   minDCF({to_effective_prior(alt_working_point1)}): {mindcf_alt1}   minDCF({to_effective_prior(alt_working_point2)}): {mindcf_alt2}\n")

            np.save(f"{scores_path}/scores_gmm_effpr_{to_effective_prior(working_point)}_targetmode_{mode_target}_nontargetmode_{mode_nontarget}_targetNumComps_{targetNumComps}_nontargetNumComps_{nontargetNumComps}_pcadim_{pca_dim}", scores)


if __name__ == '__main__':
    DTR, LTR = load_data('./dataset/Train.txt')
    DTE, LTE = load_data('./dataset/Test.txt')

    #mvg_evaluation(DTE, LTE, DTR, LTR, 'mvg/results', (0.5, 1, 10))
    #logreg_evaluation(DTE, LTE, DTR, LTR, (0.5, 1, 10))
    #svm_evaluation(DTE, LTE, DTR, LTR, (0.5, 1, 10))
    gmm_evaluation(DTE, LTE, DTR, LTR, (0.5, 1, 10), (0.5, 1, 1), (0.9, 1, 1))
    