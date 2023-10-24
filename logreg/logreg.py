import numpy as np
import scipy.optimize
import sklearn
from utils.data_utils import split_k_folds, covarMat, datasetMean

def logreg_obj_wrap(DTR, LTR, _lambda):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        w = w.reshape(w.size, 1)

        z = np.array(LTR)
        z[z==0] = -1
        z = z.reshape(1, LTR.size)

        exp = np.dot(w.T, DTR) + b
        exp = -exp * z

        terms = np.logaddexp(np.zeros(DTR.shape[1]), exp)
        sum = terms.sum()
        
        reg = float(_lambda)/2 * np.dot(w.T, w)

        result = reg + sum/DTR.shape[1]

        return result.reshape(result.size,)

    return logreg_obj

def logreg_prior_weighted_obj_wrap(DTR, LTR, _lambda, prior):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        w = w.reshape(w.size, 1)

#        z = np.array(LTR)
#        z[z==0] = -1
#        z = z.reshape(1, LTR.size)

#        exp = np.dot(w.T, DTR) + b
#        exp = -exp * z

        n_t = LTR[LTR == 1].size

        exp_target = np.dot(w.T, DTR[:, LTR == 1]) + b
        exp_target = -exp_target * 1 # z = 1

        exp_nontarget = np.dot(w.T, DTR[:, LTR == 0]) + b
        exp_nontarget = -exp_nontarget * (-1) # z = -1

        terms_target = (prior / n_t) * np.logaddexp(np.zeros(DTR[:, LTR == 1].shape[1]), exp_target)
        terms_nontarget =  ((1-prior) / (LTR.size - n_t)) * np.logaddexp(np.zeros(DTR[:, LTR == 0].shape[1]), exp_nontarget)
        sum = terms_target.sum() + terms_nontarget.sum()
        
        reg = float(_lambda)/2 * np.dot(w.T, w)

        result = reg + sum

        return result.reshape(result.size,)

    return logreg_obj

def logistic_regression_wrap(w, b):
    def logistic_regression_classifier(x):
        s = np.dot(w.T, x) + b

        #preds = np.zeros(s.shape)
        #preds[s > 0] = 1

        #return preds.reshape(preds.size,)
        return s

    return logistic_regression_classifier

def logreg_accuracy(classifier, DTE, LTE):
    preds = classifier(DTE)

    correct_preds = np.zeros(preds.shape)
    correct_preds[preds == LTE] = 1

    return np.sum(correct_preds)/correct_preds.size

def logreg_k_fold_train(DTR, LTR, k, seed, _lambda, eff_prior):
    folds, labels_folds = split_k_folds(DTR, LTR, k, seed)

    test_scores = []
    for i in range(k):
        print(f"Fold {i}")
        test_fold = folds[i]
        test_labels = labels_folds[i]

        train_folds = np.hstack([folds[j] for j in range(k) if i != j])
        train_labels = np.concatenate([labels_folds[j] for j in range(k) if i != j])

        # train
        logreg_obj = logreg_prior_weighted_obj_wrap(train_folds, train_labels, _lambda, eff_prior)
        #logreg_obj = logreg_obj_wrap(train_folds, train_labels, _lambda)
        solution = scipy.optimize.fmin_l_bfgs_b(logreg_obj,  x0=np.zeros(train_folds.shape[0] + 1), approx_grad=True)

        w = np.array(solution[0][0:-1]).reshape(len(solution[0][0:-1]), 1)
        b = solution[0][-1]
        logreg_classifier = logistic_regression_wrap(w, b)

        # test
        #test_scores.append(logreg_classifier(test_fold))
        test_scores.append(logreg_classifier(test_fold) - np.log(eff_prior/(1-eff_prior)))
        
    scores = np.hstack(test_scores)
    print(scores.shape)

    # train on whole data
    logreg_obj = logreg_prior_weighted_obj_wrap(DTR, LTR, _lambda, eff_prior)
    #logreg_obj = logreg_obj_wrap(DTR, LTR, _lambda)
    solution = scipy.optimize.fmin_l_bfgs_b(logreg_obj,  x0=np.zeros(train_folds.shape[0] + 1), approx_grad=True)

    w = np.array(solution[0][0:-1]).reshape(len(solution[0][0:-1]), 1)
    b = solution[0][-1]

    params = (w, b)

    return params, scores   # params is the tuple (w, b)
                            # scores is a 1D array of scores for each sample in DTR (accordinng to k fold protocol)
