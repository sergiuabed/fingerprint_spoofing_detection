import numpy as np
import scipy
from utils.data_utils import split_k_folds, load_data, to_effective_prior
from utils.plot_utils import bayes_error_plot
from logreg.logreg import logreg_prior_weighted_obj_wrap, logistic_regression_wrap

SEED = 22

def train_calibration_model(scores, eff_prior, K, dataset_file, seed, scores_to_calibrate=None):

    # 'scores' contains the scores ordered according to the
    # shuffle performed during training. To obtain the exact
    # same permutation, we use the same seed as the one used
    # the first time

    DTR, LTR = load_data(dataset_file)
    folds, labels_folds = split_k_folds(DTR, LTR, K, seed)

    labels = np.concatenate(labels_folds)

    calibration_seed  = 20

    # scores is 2-dimensional 
    score_folds, labels_folds = split_k_folds(scores, labels, K, calibration_seed)

    calibrated_scores_list = []
    for i in range(K):
        test_fold  = score_folds[i]
        test_labels = labels_folds[i]

        train_folds = np.hstack([score_folds[j] for j in range(K) if i != j])
        train_labels = np.concatenate([labels_folds[j] for j in range(K) if i != j])

        logreg_obj = logreg_prior_weighted_obj_wrap(train_folds, train_labels, 0, eff_prior)
        solution = scipy.optimize.fmin_l_bfgs_b(logreg_obj,  x0=np.zeros(train_folds.shape[0] + 1), approx_grad=True)

        w = np.array(solution[0][0:-1]).reshape(len(solution[0][0:-1]), 1)
        b = solution[0][-1]
        logreg_classifier = logistic_regression_wrap(w, b)

        calibrated_scores_list.append(logreg_classifier(test_fold) - np.log(eff_prior/(1-eff_prior)))

    calibrated_scores = np.hstack(calibrated_scores_list)

    if scores_to_calibrate is not None:
        print("Calibrating scores on evaluation dataset")

        logreg_obj = logreg_prior_weighted_obj_wrap(scores, labels, 0, eff_prior)
        solution = scipy.optimize.fmin_l_bfgs_b(logreg_obj,  x0=np.zeros(scores.shape[0] + 1), approx_grad=True)

        w = np.array(solution[0][0:-1]).reshape(len(solution[0][0:-1]), 1)
        b = solution[0][-1]
        logreg_classifier = logistic_regression_wrap(w, b)

        return logreg_classifier(scores_to_calibrate) - np.log(eff_prior/(1-eff_prior))

    return calibrated_scores

def calibration_on_eval_data():
    '''
    Train the calibrator on the validation samples scores given by the model.
    Then, use the calibrator to calibrate evaluation samples scores. 
    '''
    path = 'dataset/Train.txt'

    # load scores on training data on which we train the calibration model
    #training_scores = np.load("gmm/results/scores_gmm_effpr_0.09090909090909091_targetmode_diag_nontargetmode_default_targetNumComps_2_nontargetNumComps_8_pcadim_6.npy")
    #scores_to_calibrate = np.load("evaluation_logs/gmm_scores/scores_gmm_effpr_0.09090909090909091_targetmode_diag_nontargetmode_default_targetNumComps_2_nontargetNumComps_2_pcadim_6.npy")

    #training_scores = np.load("logreg/results/scores_quadrlogreg_effpr_0.09090909090909091_znorm_False_pcadim_6_lambda_0.1.npy")
    #scores_to_calibrate = np.load("evaluation_logs/quadrLogReg_scores/eval_scores_quadrlogreg_effpr_0.09090909090909091_znorm_False_pcadim_6_lambda_0.1.npy")

    training_scores = np.load("svm/results_poly/scores_polySVM_effpr_0.09090909090909091_degree_2_const_1_kv_0_znorm_False_pcadim_6_c_1.0.npy")
    scores_to_calibrate = np.load("evaluation_logs/svm_scores/scores_polySVM_effpr_0.09090909090909091_degree_2_const_1_kv_0_znorm_False_pcadim_6_c_1.0.npy")

    eff_prior = to_effective_prior((0.5, 1, 10))

    calibrated_scores = train_calibration_model(training_scores, eff_prior, 5, path, SEED, scores_to_calibrate=scores_to_calibrate)

    DTE, LTE = load_data("dataset/Test.txt")

    bayes_error_plot(scores_to_calibrate.reshape((scores_to_calibrate.size,)), LTE, title='QuadrLogReg before calibration', svm_scores=True)
    bayes_error_plot(calibrated_scores.reshape((calibrated_scores.size,)), LTE, title='QuadrLogReg after calibration', svm_scores=False)# for SVM after calibration, set "svm_scores=False"

def calibration_on_validation_data():
    path = 'dataset/Train.txt'

    #scores = np.load("mvg/results/scores_mvg_effpr_0.09090909090909091_mode_default_pcadim_-1.npy")
    #scores = np.load("svm/results_rbf/scores_rbfSVM_effpr_0.09090909090909091_gamma_0.001_kv_0_znorm_False_pcadim_6_c_100.0.npy")
    #scores = np.load("svm/results_poly/scores_polySVM_effpr_0.09090909090909091_degree_2_const_1_kv_0_znorm_False_pcadim_6_c_1.0.npy")
    #scores = np.load("logreg/results/scores_quadrlogreg_effpr_0.09090909090909091_znorm_False_pcadim_6_lambda_0.1.npy")
    scores = np.load("gmm/results/scores_gmm_effpr_0.09090909090909091_targetmode_diag_nontargetmode_default_targetNumComps_2_nontargetNumComps_8_pcadim_6.npy")

    eff_prior = to_effective_prior((0.5, 1, 10))

    calibrated_scores = train_calibration_model(scores, eff_prior, 5, path, SEED)
    #calibrated_scores -= np.log(eff_prior/(1-eff_prior))

    DTR, LTR = load_data(path)
    folds, labels_folds = split_k_folds(DTR, LTR, 5, SEED)
    labels = np.concatenate(labels_folds)

    DTE, LTE = load_data("dataset/Test.txt")

    #bayes_error_plot(scores.reshape((scores.size,)), labels, svm_scores=True) # FOR SVM ONLY, SET "svm_scores=True"
    bayes_error_plot(scores.reshape((scores.size,)), labels, title='GMM before calibration', svm_scores=False)

    score_folds, labels_folds = split_k_folds(scores, labels, 5, 20)
    labels = np.concatenate(labels_folds)

    bayes_error_plot(calibrated_scores.reshape((calibrated_scores.size,)), labels, title='GMM after calibration', svm_scores=False) # for SVM after calibration, set "svm_scores=False"

    print(scores.shape)
    print(calibrated_scores.shape)

if __name__ == '__main__':
    #calibration_on_validation_data()
    calibration_on_eval_data()