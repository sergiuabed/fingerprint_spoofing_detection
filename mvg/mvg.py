import numpy as np
import math
import scipy
from utils.data_utils import split_k_folds, covarMat, datasetMean
from utils.dimensionality_reduction import PCA_matrix

def logpdf_GAU_ND(X, mu, C):
    firstTerm = (-1)*np.log(2*math.pi)*0.5*C.shape[0]

    sign, detC = np.linalg.slogdet(C)

    secondTerm = (-1)*0.5*detC

    i = 0
    Y = 0  # just to define the scope of Y outside the loop
    while i < X.shape[1]:
        x = X[:, i].reshape(X.shape[0],1)
        # subtract the mean from the sample
        x_centered = x-mu.reshape(mu.size, 1)

        invC = np.linalg.inv(C)

        thirdTerm = np.dot(x_centered.T, invC)
        thirdTerm = np.dot(thirdTerm, x_centered)
        thirdTerm = (-1)*0.5*thirdTerm

        y = firstTerm + secondTerm + thirdTerm
        if i == 0:
            Y = y
        else:
            Y = np.hstack([Y, y])

        i += 1

    return Y

def MVG_parameters(DTR, LTR, mode='default'):
    '''
    mode assumes one of three possible values: 'default', 'tied' and 'diag'
    '''

    labels = np.unique(LTR)
    means = {}
    covariances = {}

    if mode == 'tied':
        cov_sum = np.zeros((DTR.shape[0], DTR.shape[0]))
        for l in labels:
            X = DTR[:, LTR == l]
            mean = datasetMean(X)   # "broadcasting ready"
            cov_sum += X.shape[1] * covarMat(X)

            means[l] = mean
        
        cov_avg = cov_sum / DTR.shape[1]
        covariances = [cov_avg for l in labels]
    else:
        for l in labels:
            X = DTR[:, LTR == l]
            mean = datasetMean(X)   # "broadcasting ready"
            cov = covarMat(X)

            if mode == 'diag':
                cov = cov * np.eye(cov.shape[0])

            means[l] = mean
            covariances[l] = cov

    return [means, covariances], labels

def MVG_classifier_wrap(means, covariances):

    def MVG_classifier_logDomain(D):

        loglikelihoods_target = logpdf_GAU_ND(D, means[1], covariances[1])
        loglikelihoods_nontarget = logpdf_GAU_ND(D, means[0], covariances[0])

        #threshold = -np.log (effective_prior / (1 - effective_prior))
        
        llr = loglikelihoods_target - loglikelihoods_nontarget

        return llr
    
    return MVG_classifier_logDomain

def MVG_k_fold_train(DTR, LTR, k, seed, pca_dim, mode='default'):
    folds, labels_folds = split_k_folds(DTR, LTR, k, seed)

    test_scores = []
    for i in range(k):
        #print(f"Fold {i}")
        test_fold = folds[i]
        test_labels = labels_folds[i]

        train_folds = np.hstack([folds[j] for j in range(k) if i != j])
        train_labels = np.concatenate([labels_folds[j] for j in range(k) if i != j])

        # apply PCA
        mu_train = datasetMean(train_folds)
        if pca_dim < 0:
            train_folds_proj = train_folds - mu_train
            test_fold_proj = test_fold - mu_train
        else:
            s, P = PCA_matrix(train_folds, pca_dim)
            train_folds_proj = np.dot(P.T, train_folds - mu_train)   #project on lower space
            test_fold_proj = np.dot(P.T, test_fold - mu_train)

        # train
        params, keys = MVG_parameters(train_folds_proj, train_labels, mode) # 'keys' is a list of non-repeating labels
        means = params[0]

        covariances = params[1]

        mvg_classifier = MVG_classifier_wrap(means, covariances)

        # test
        test_scores.append(mvg_classifier(test_fold_proj))
        
    scores = np.hstack(test_scores)

    # train on whole data

    # apply PCA
    mu = datasetMean(DTR)
    if pca_dim < 0:
        DP = DTR - mu
    else:
        s, P = PCA_matrix(DTR, pca_dim)
        DP = np.dot(P.T, DTR - mu)   #project on lower space
    
    params, keys = MVG_parameters(DP, LTR, mode)

    return params, scores   # params is a list of 2 dictionaries (one consisting of means and the other of variance matrices)
                            # scores is a 1D array of scores for each sample in DTR (accordinng to k fold protocol)
