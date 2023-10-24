import numpy as np

def get_confusion_matrix(predicted_labels, labels, nr_classes):
    confusion_matrix = np.zeros((nr_classes, nr_classes))
    for i in range(nr_classes):
        for j in range(nr_classes):
            match = predicted_labels[labels == j] # labels==j -> select items on indices corresponding to samples belonging to class j
            confusion_matrix[i, j] = sum((match == i).astype(int))

    return confusion_matrix

def binary_optimal_bayes_decision(llr, working_point, svm_scores=False):
    '''
    This assigns labels using the theoretical optimal threshold

    Set 'svm_scores=True' when passing scores from SVM models
    '''
    prior1, C_fn, C_fp = working_point

    prior0 = 1 - prior1
    th = -np.log(prior1 * C_fn / (prior0 * C_fp))

    predicted_labels = np.zeros(llr.shape)

    if svm_scores:
        predicted_labels[llr > 0] = 1
    else:
        predicted_labels[llr > th] = 1

    return predicted_labels

def bayes_risk(cm, working_point):
    prior1, C_fn, C_fp = working_point
    prior0 = 1 - prior1

    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[1, 0] + cm[0, 0])

    DCF_u = prior1 * C_fn * FNR + prior0 * C_fp * FPR
    DCF_n = DCF_u / min(prior1 * C_fn, prior0 * C_fp)

    return (DCF_u, DCF_n)

def minimum_bayes_risk(llr, labels, working_point, svm_scores=False):
    '''
    Set 'svm_scores=True' when passing scores from SVM models
    '''

    ths = np.linspace(start=min(llr), stop=max(llr), num=7000)
    minDCF = float("inf")
    for th in ths:
        predicted_labels = np.zeros(llr.shape)
        if svm_scores:
            #predicted_labels[llr > 0] = 1
            predicted_labels[llr > th] = 1
        else:
            predicted_labels[llr > th] = 1

        cm = get_confusion_matrix(predicted_labels, labels, 2)

        DCF_u, DCF_n = bayes_risk(cm, working_point)
        #print(f"    {DCF_n}")
        
        if minDCF > DCF_n:
            minDCF = DCF_n
    
    return minDCF

