import numpy as np
import math
import scipy
import sys
from utils.data_utils import datasetMean, covarMat, split_k_folds
from utils.dimensionality_reduction import PCA_matrix
from utils.measurements import *

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

def logpdf_GMM(X, gmm):
    priors = [gmm[c][0] for c in range(len(gmm))]
    gaussian_densities = [logpdf_GAU_ND(X, gmm[c][1], gmm[c][2]) for c in range(len(gmm))]

    priors = np.array(priors).reshape((len(priors), 1))
    log_priors = np.log(priors)

    S = np.vstack(gaussian_densities)
    S += log_priors # broadcasting

    logdens = scipy.special.logsumexp(S, axis=0)

    return logdens.reshape(1, logdens.size)
    
def gmm_estimation(X, init_gmm, mode='default'):
    stop_condition = False
    gmm = init_gmm
    num_iters = 0

    psi = 0.01
    while not stop_condition:
        #print(f"Iteration {num_iters}")
        priors = [gmm[c][0] for c in range(len(gmm))]
        gaussian_densities = [logpdf_GAU_ND(X, gmm[c][1], gmm[c][2]) for c in range(len(gmm))]

        priors = np.array(priors).reshape((len(priors), 1))
        log_priors = np.log(priors)

        S = np.vstack(gaussian_densities)
        S += log_priors # broadcasting

        # at this point, S is joint-loglikelihood

        marginal_llh = scipy.special.logsumexp(S, axis=0)
        marginal_llh = marginal_llh.reshape((1, marginal_llh.shape[0]))

        log_responsibilities = S - marginal_llh

        responsibilities = np.exp(log_responsibilities)

        zero_stat = responsibilities.sum(1)
        first_stat = X.dot(responsibilities.T)

        means = first_stat / zero_stat.reshape(1, zero_stat.size)
        weights = zero_stat / zero_stat.sum(0)

        second_stat = [] # np.zeros((responsibilities.shape[1], responsibilities.shape[1]))
        covars = []

        for c in range(responsibilities.shape[0]):
            rc = responsibilities[c].reshape((1, len(responsibilities[c]))) # responsabilities of each sample under component (cluster) 'c'
            term = X * rc
            term = term.dot(X.T)

            second_stat.append(term)

            m = means[:, c].reshape((X.shape[0], 1))
            mm = m.dot(m.T)
            cov = term / zero_stat[c] - mm

            covars.append(cov)

            #X_centered = X - m
            #cov = np.dot(X_centered * rc, X_centered.T) / zero_stat[c]
            #covars.append(cov)

        if mode == 'tied':
            covars_sum = np.zeros(covars[0].shape)
            for c in range(responsibilities.shape[0]):
                covars_sum += zero_stat[c] * covars[c]

            covars_avg = covars_sum / zero_stat.sum()
            U, s, _ = np.linalg.svd(covars_avg)
            s[s<psi] = psi
            covars_avg = np.dot(U, s.reshape(s.size, 1)*U.T)

            covars = [covars_avg for _ in range(responsibilities.shape[0])]
            new_gmm = [(weights[c].item(), means[:, c].reshape(means.shape[0], 1), covars[c]) for c in range(responsibilities.shape[0])]

        if mode == 'diag':
            diag_covars = [covars[c] * np.eye(covars[0].shape[0]) for c in range(responsibilities.shape[0])]
            for i in range(len(diag_covars)):
                U, s, _ = np.linalg.svd(diag_covars[i])
                s[s<psi] = psi
                diag_covars[i] = np.dot(U, s.reshape(s.size, 1)*U.T)
            covars = diag_covars
            new_gmm = [(weights[c].item(), means[:, c].reshape(means.shape[0], 1), diag_covars[c]) for c in range(responsibilities.shape[0])]
        else:
            for i in range(len(covars)):
                U, s, _ = np.linalg.svd(covars[i])
                s[s<psi] = psi
                covars[i] = np.dot(U, s.reshape(s.size, 1)*U.T)

            new_gmm = [(weights[c].item(), means[:, c].reshape(means.shape[0], 1), covars[c]) for c in range(responsibilities.shape[0])]


        llh = marginal_llh.sum(axis=1) # log-likelihood of the samples in X under current gmm
        new_llh = logpdf_GMM(X, new_gmm).sum() # log-likelihood of the samples in X under newly estimated gmm

        #print(f"llh = {llh}")
        #print(f"new_llh = {new_llh}")

        delta_llh = new_llh - llh
        if (delta_llh / responsibilities.sum()) <= 1e-6:

            if delta_llh < 0 :
                print(f"Shape of mean: {init_gmm[0][1].shape}")
                sys.exit("LLH DECREASED!!! SOMETHING IS WRONG!")
                return None, None
            
            stop_condition = True
        
        gmm = new_gmm
        num_iters += 1

    return new_gmm, new_llh / responsibilities.sum()

def lbg_algorithm(X, init_gmm, alpha, num_components, mode="default"):
    '''
    'mode' can assume 1 of 3 values: 'default', 'tied', 'diag'
    '''
    gmm = init_gmm
    avg_llh = None

    gmm_list = [init_gmm] # insert in this list all candidate gmm
                  # the position (index) of an item in this list has this relation:
                        # item at ith position has a number of clusters equal to 2**i

    for i in range(int(np.log2(num_components))):
        print(f"Iteration {i} of LBG")
        new_gmm = []
        for comp in gmm:
            print(f"Shape of comp mean: {comp[1].shape}")
            U, s, Vh = np.linalg.svd(comp[2])
            d = U[:, 0:1] * s[0]**0.5 * alpha

            new_comp1 = (0.5 * comp[0], comp[1].reshape(comp[1].shape[0],1) + d, comp[2])
            new_comp2 = (0.5 * comp[0], comp[1].reshape(comp[1].shape[0],1) - d, comp[2])

            new_gmm.append(new_comp1)
            new_gmm.append(new_comp2)

            #print(f"Shape of mean in new_comp1: {new_comp1[1].shape}")
        new_gmm, avg_llh = gmm_estimation(X, new_gmm, mode)
        gmm = new_gmm
        gmm_list.append(gmm)

    #return gmm, avg_llh
    return gmm_list

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
        
        cov_avg = cov_sum / X.shape[1]
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

def gmm_classifier_wrap(gmms):

    def gmm_classifier(X):

        loglikelihoods_target = logpdf_GMM(X, gmms[1])
        loglikelihoods_nontarget = logpdf_GMM(X, gmms[0])

        
        llr = loglikelihoods_target - loglikelihoods_nontarget
        
        return llr
    
    return gmm_classifier

def gmm_classifier_train(DTR, LTR, num_comps_target, num_comps_nontarget, mode_target='default', mode_nontarget='default'):
    # start by fitting a MVG on each class
    # each MVG corresponds to a GMM, not to a component of a GMM!!
#    [means, covariances], labels = MVG_parameters(DTR, LTR, mode) # no duplicates in 'labels'
#
#    init_gmms = {l: [(1, means[l], covariances[l])] for l in labels}

    [means1, covariances1], labels = MVG_parameters(DTR, LTR, mode_target) # no duplicates in 'labels'
    [means0, covariances0], labels = MVG_parameters(DTR, LTR, mode_nontarget) # no duplicates in 'labels'

    init_gmms = {0: [(1, means0[0], covariances0[0])], 1: [(1, means1[1], covariances1[1])]}

    num_comps = {0: num_comps_nontarget, 1: num_comps_target}
    mode = {0: mode_nontarget, 1: mode_target}
    final_gmms = {}
    for l in labels:
        DTR_class_l = DTR[:, LTR == int(l)]
        #new_gmm, _ = lbg_algorithm(DTR_class_l, init_gmms[l], 0.1, num_comps[l], mode[l])
        print(f"Class {l} density estimation\n")
        new_gmm = lbg_algorithm(DTR_class_l, init_gmms[l], 0.1, num_comps[l], mode[l])

        final_gmms[l] = new_gmm

    return final_gmms

def gmm_k_fold_train(DTR, LTR, k, seed, pca_dim, num_comps_target, num_comps_nontarget, mode_target='default', mode_nontarget='default'):
    '''
        Returns a tuple of 2:
        - "gmm_params": a dictionary of lists, each list storing different gmms for a certain class

            ex: in the binary case, gmm_params = {0: [gmm0, gmm1, gmm2, ...], 1: [gmm0, gmm1, gmm2, ...]}
                gmmi represents a GMM and it is a list of tuples, where each tuple contains the parametersfor a cluster (component)
                for a cluster (component) of a GMM. the 'i' in 'gmmi' indicates that the nr of clusters in gmmi is 2**i

        - "test_dict": a dictionary of arrays, each array containing the scores for each sample of all folds

            ex: test_dict = {(j, i): 'array of scores'}
                j = nr of clusters for non-target class
                i = nr of clusters for target class
    '''

    folds, labels_folds = split_k_folds(DTR, LTR, k, seed)

    #test_scores = []
    test_dict = {} # the keys of this dict are tuples (j, i), where i and j indicate which config of gmm to choose from target and nontarget, respectively
    #test_dict = {(j, i): [] for i in range(int(np.log2(num_comps_target)) + 1) for j in range(int(np.log2(num_comps_nontarget)) + 1)}

    for i in range(int(np.log2(num_comps_target)) + 1):
        for j in range(int(np.log2(num_comps_nontarget)) + 1):
            if i == 0 and j == 0:
                continue
            test_dict[(2**j, 2**i)] = []


    for i in range(k):
        print(f"Fold {i}")
        test_fold = folds[i]
        test_labels = labels_folds[i]

        train_folds = np.hstack([folds[j] for j in range(k) if i != j])
        train_labels = np.concatenate([labels_folds[j] for j in range(k) if i != j])

        # apply pca
        mu_train = datasetMean(train_folds)
        if pca_dim < 0:
            train_folds_proj = train_folds - mu_train
            test_fold_proj = test_fold - mu_train
        else:
            s, P = PCA_matrix(train_folds, pca_dim)
            train_folds_proj = np.dot(P.T, train_folds - mu_train)   #project on lower space
            test_fold_proj = np.dot(P.T, test_fold - mu_train)

        # train
        gmm_params = gmm_classifier_train(train_folds_proj, train_labels, num_comps_target, num_comps_nontarget, mode_target, mode_nontarget)
        # gmm_params is a dictionary of lists, each list storing possible gmm configurations for a class

        for i in range(len(gmm_params[1])): #target_gmm in gmm_params[1]:
            for j in range(len(gmm_params[0])): #nontarget_gmm in gmm_params[0]:
                if i == 0 and j == 0:
                    continue

                gmm_classifier = gmm_classifier_wrap({0: gmm_params[0][j], 1: gmm_params[1][i]})

                # test
                #test_scores.append(gmm_classifier(test_fold))
                test_dict[(2**j, 2**i)].append(gmm_classifier(test_fold_proj))
        
    
    #scores = np.hstack(test_scores)
    #print(scores.shape)
    for k in test_dict.keys():
        l = test_dict[k]
        test_dict[k] = np.hstack(l)
    # at this point, each entry has an array of scores for all folds

    # train on whole data
    # apply PCA
    mu = datasetMean(DTR)
    if pca_dim < 0:
        DP = DTR - mu
    else:
        s, P = PCA_matrix(DTR, pca_dim)
        DP = np.dot(P.T, DTR - mu)   #project on lower space

    gmm_params = gmm_classifier_train(DP, LTR, num_comps_target, num_comps_nontarget, mode_target, mode_nontarget)

    return gmm_params, test_dict    # gmm_params is a dictionary of lists, each list storing possible gmm configurations for a class