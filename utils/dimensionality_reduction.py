import numpy as np
import scipy
import matplotlib.pyplot as plt

def PCA_matrix(dataMat, m):
    mu=dataMat.mean(1)
    dataMatCentered=dataMat-mu.reshape(mu.size,1) #through broadcasting, mu is subracted from each column of dataMatCentered

    N=dataMatCentered.shape[1]
    covarianceMat=np.dot(dataMatCentered,dataMatCentered.T)/N
    s, U=np.linalg.eigh(covarianceMat)
    P=U[:, ::-1][:, 0:m]

    s_decr = s[::-1]

    return s_decr, P

def PCA_variance_plot(dataMat):
    s, _ = PCA_matrix(dataMat, dataMat.shape[1])

    variances = [s[:m].sum() / s.sum() for m in range(0, dataMat.shape[0]+1)]
    print(variances)

    plt.figure()
    plt.title("PCA explained variance")
    plt.xlabel("PCA dimension")
    plt.ylabel("fraction of explained dimension")

    plt.xticks(range(dataMat.shape[0]+1))
    plt.yticks([0.1*n for n in range(11)])
    plt.plot(range(dataMat.shape[0]+1), variances)
    plt.grid()
    plt.show()

def LDA_matrix(dataMat, labels, m):

    mean=dataMat.mean(1)
    mean=mean.reshape(mean.size, 1)
    Sb=np.zeros((mean.shape[0], mean.shape[0]), dtype=np.float32)
    Sw=np.zeros((mean.shape[0], mean.shape[0]), dtype=np.float32)

    for i in range(0,3):
        data_labeled=dataMat[:, labels==i]
        
        mean_labeled=data_labeled.mean(1)
        mean_labeled=mean_labeled.reshape(mean_labeled.size, 1) #reshape the mean as a (2D) column vector

        nc=data_labeled.shape[1]    #nr of samples in the current class
        N=dataMat.shape[1]

        #compute between class covariance matrix Sb
        e=mean_labeled-mean
        term=np.dot(e, e.T)
        term=term*nc
        term=term/N
        Sb=Sb+term

        #compute within class covariance matrix Sw
        data_labeled_centered=data_labeled-mean_labeled     #recall: mean_labeled is already shaped as a column vector, so broadcasting occurs as wanted
        covariance_mat=np.dot(data_labeled_centered, data_labeled_centered.T)
        covariance_mat=covariance_mat/N
        Sw=Sw+covariance_mat

    s, U=scipy.linalg.eigh(Sb, Sw)
    W=U[:, ::-1][:, 0:m]

    return W