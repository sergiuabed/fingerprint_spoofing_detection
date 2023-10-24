import numpy as np
import scipy.optimize
import sklearn
from data_utils import split_k_folds, covarMat, datasetMean, get_empirical_prior

def kernel_polynomial_wrap(c, degree, k):
    '''
    returns a ready-to-use polynomial kernel function
    '''

    def kernel(X1, X2):
        res = (np.dot(X1.T, X2) + c) ** degree + k**2
        return res

    return kernel

def kernel_rbf_wrap(gamma, k):
    '''
    returns a ready-to-use Radial Basis Function kernel
    '''

    def kernel(X1, X2):
        res = np.zeros((X1.shape[1], X2.shape[1]))

        for i in range(X1.shape[1]):
            for j in range(X2.shape[1]):
                elem = X1[:,i:i+1] - X2[:, j:j+1]
                elem = elem ** 2
                elem = -gamma * np.sum(elem, axis=0)

                res[i, j] = np.exp(elem) + k**2

        return res
    
    return kernel

def svm_dual_obj_wrap(DTR, LTR, K, kernel=None):
    '''
    this is actually the negative of the svm dual objective. It's done this way because
    we need to look for the maximum and scipy.optimize.fmin_l_bfgs_b is a minimizer
    '''
    if kernel is None:
        embedding = K*np.ones((1, DTR.shape[1]), dtype=float)
        D = np.vstack((DTR, embedding))
    else:
        D = DTR
    
    zh = np.array(LTR).reshape(1,D.shape[1])
    zv = np.array(LTR).reshape(D.shape[1],1)

    if kernel is None:
        G = np.dot(D.T, D)
    else:
        G = kernel(D, D)

    H = G*zh
    H = H*zv

    def svm_dual_obj(_alphas):
        L = 0.5*_alphas.T.dot(H).dot(_alphas) - _alphas.T.dot(np.ones((_alphas.shape)))
        grad_l = H.dot(_alphas) - 1
        grad_l = grad_l.reshape((grad_l.size,))

        return (L.item(), grad_l)
    
    return svm_dual_obj

def svm_primal_obj_wrap(DTR, LTR, K, C):
    embedding = K*np.ones((1, DTR.shape[1]), dtype=float)
    D = np.vstack((DTR, embedding))
    z = np.array(LTR).reshape(1, D.shape[1])

    def svm_primal_obj(w_b):
        w_b = w_b.reshape(w_b.size, 1)
        P = w_b.T.dot(D)
        P = 1 - z * P
        P = C*np.maximum(P, 0).sum()
        J = 0.5 * w_b.T.dot(w_b) + P

        return J.item()
    
    return svm_primal_obj

def svm_dual_classifier_wrap(_alphas, DTR, LTR, K, kernel=None):
    if kernel is None:
        embedding = K*np.ones((1, DTR.shape[1]), dtype=float)
        D = np.vstack((DTR, embedding))
    else:
        D = DTR
    
    z = np.array(LTR).reshape(D.shape[1],1)

    if kernel is None:
        w_b = D.dot(z*_alphas)

        def svm_linear_classifier(x):
            w = w_b[0:-1]
            b = w_b[-1]*K

            s = w.T.dot(x) + b

            return s
        
        return svm_linear_classifier, w_b
    
    def svm_nonlinear_classifier(x):
        #D_filtered = D[_alphas.reshape((_alphas.size,)) != 0]

        #s = kernel(D.dot(z*_alphas), x)
        s = kernel(D, x)
        mul = z*_alphas
        s = mul.T.dot(s)

        return s

    return svm_nonlinear_classifier, None

def svm_primal_classifier_wrap(w_b, K):
    def svm_classifier(x):
        w = w_b[0:-1]
        b = w_b[-1]*K

        s = w.T.dot(x) + b

        return s
    
    return svm_classifier, w_b

def svm_accuracy(classifier, DTE, LTE):
    preds = classifier(DTE)

    correct_preds = np.zeros(preds.shape)
    correct_preds[preds == LTE] = 1

    return np.sum(correct_preds)/correct_preds.size

def duality_gap(primal_obj, dual_obj, w_b, _alphas):
    return primal_obj(w_b) + dual_obj(_alphas)[0]

def svm_k_fold_train(DTR, LTR, k, seed, kv, c, eff_prior, kernel=None):
    '''
    This function returns tuple (params, scores)
    
    IMPORTANT: 'params' contains the dual solution. To obtain the actual parameters, pass this 'params' to svm_dual_classifier_wrap
    together with TRAINING DATASET (DTR, LTR), NOT the test dataset!

    '''

    folds, labels_folds = split_k_folds(DTR, LTR, k, seed)

    test_scores = []
    for i in range(k):
        print(f"Fold {i}")
        test_fold = folds[i]
        test_labels = labels_folds[i]

        train_folds = np.hstack([folds[j] for j in range(k) if i != j])
        train_labels = np.concatenate([labels_folds[j] for j in range(k) if i != j])

        x0 = np.zeros((train_folds.shape[1]), dtype=float)
        
        _, priors = get_empirical_prior(train_labels)
        pi_t = priors[1]
        pi_n = priors[-1]

        c_t = c * eff_prior / pi_t
        c_n = c * eff_prior / pi_n

        bounds = [(0, c_t) if i == 1 else (0, c_n) for i in train_labels]

        # train
        # dual optimization
        svm_dual_obj = svm_dual_obj_wrap(train_folds, train_labels, kv, kernel=kernel)
        #solution = scipy.optimize.fmin_l_bfgs_b(svm_dual_obj, x0=x0, bounds=[(0, c) for _ in range(train_folds.shape[1])], factr=1.0)
        solution = scipy.optimize.fmin_l_bfgs_b(svm_dual_obj, x0=x0, bounds=bounds, factr=1.0)

        _alphas = np.array(solution[0]).reshape(len(solution[0]), 1)
        loss_dual = solution[1]
        svm_dual_classifier, w_b1 = svm_dual_classifier_wrap(_alphas, train_folds, train_labels, kv, kernel=kernel)

        #primal optimization
#        svm_primal_obj = svm_primal_obj_wrap(train_folds, train_labels, kv, c)
#        loss_primal = svm_primal_obj(w_b1)
#
#        gap = duality_gap(svm_primal_obj, svm_dual_obj, w_b1, _alphas)
#
#        print(f"K: {kv}")
#        print(f"    C: {c}")
#        #print(f"        Value of the objective at the minimum: {solution[1]}")
#        print(f"        Primal loss: {'{:e}'.format(loss_primal)}")
#        print(f"        Dual loss: {'{:e}'.format(-loss_dual)}")
#        print(f"        Duality gap: {'{:e}'.format(gap)}")

        # test
        test_scores.append(svm_dual_classifier(test_fold))

    scores = np.hstack(test_scores)
    print(scores.shape)

    # train on whole data
    # dual optimization
    x0 = np.zeros((DTR.shape[1]), dtype=float)

    _, priors = get_empirical_prior(LTR)
    pi_t = priors[1]
    pi_n = priors[-1]

    c_t = c * eff_prior / pi_t
    c_n = c * eff_prior / pi_n

    bounds = [(0, c_t) if i == 1 else (0, c_n) for i in LTR]

    svm_dual_obj = svm_dual_obj_wrap(DTR, LTR, kv)
    #solution = scipy.optimize.fmin_l_bfgs_b(svm_dual_obj, x0=x0, bounds=[(0, c) for _ in range(DTR.shape[1])], factr=1.0)
    solution = scipy.optimize.fmin_l_bfgs_b(svm_dual_obj, x0=x0, bounds=bounds, factr=1.0)

    _alphas = np.array(solution[0]).reshape(len(solution[0]), 1)
    svm_dual_classifier, w_b1 = svm_dual_classifier_wrap(_alphas, DTR, LTR, kv)

    params = _alphas

    return params, scores   # params contains the dual solution. To obtain the actual parameters, pass this 'params' to svm_dual_classifier_wrap
                            # scores is a 1D array of scores for each sample in DTR (accordinng to k fold protocol)
