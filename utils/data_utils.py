import numpy as np

def load_data(dataset_file):
    list_vectors=[]
    labels=[]
    nr_lines=0
    with open(dataset_file, 'r') as f:
        for line in f:
            sample=line.strip().split(",")
            measurements=np.array([float(i) for i in sample[0:-1]]).reshape(len(sample[0:-1]),1)
            label=int(sample[-1])

            list_vectors.append(measurements)
            labels.append(label)
            nr_lines+=1

    array_lables=np.array(labels)
    data_matrix=np.hstack(list_vectors)

    return (data_matrix, array_lables)

def split_k_folds(data, labels, k, seed):
    data_labels = np.vstack([data, labels.reshape(1, labels.size)])

    np.random.seed(seed) # to make sure we get the same shuffle output when using the same seed
    np.random.shuffle(data_labels.T)
    #idx = np.random.permutation(data.shape[1])

    data_sh = data_labels[:-1, :]
    labels_sh = data_labels[-1, :]

    #data_sh = data[:, idx]
    #labels_sh = labels[idx]

    fold_size = data_sh.shape[1] // 5 + 1
    folds = [data_sh[:, i*fold_size : (i+1)*fold_size] for i in range(k)]
    labels_folds = [labels_sh[i*fold_size : (i+1)*fold_size] for i in range(k)]

#    for f in folds:
#        print(f"fold size: {f.shape}")

    return folds, labels_folds

def covarMat(X):
    mu = X.mean(axis=1)  # mean of the dataset
    mu = mu.reshape(mu.size, 1)
    # mu is suptracted from each column(sample) of X through broadcasting
    X_centered = X-mu

    covMat = np.dot(X_centered, X_centered.T) / X.shape[1]

    return covMat

def datasetMean(X):
    mu = X.mean(axis=1)
    mu = mu.reshape(mu.size, 1) # making it 2-dimensional so it can be used for broadcasting
    return mu

def to_effective_prior(working_point):
    prior = working_point[0]
    Cfn = working_point[1]
    Cfp = working_point[2]

    eff_prior = prior * Cfn / (prior * Cfn + (1 - prior) * Cfp)

    return eff_prior

def get_empirical_prior(LTR):
    classes = np.unique(LTR)
    nr_samples = {}
    priors = {}
    for c in classes:
        c = int(c)
        nr_samples[c] = sum(np.ones(LTR.size)[LTR == c])
        priors[c] = nr_samples[c] / LTR.size

    return nr_samples, priors

#def z_norm

def expand_features(DTR):
    expanded_samples = []
    for s in DTR.T:
        x = s.reshape((s.size, 1))
        m = x.dot(x.T)

        vect_xx = np.vstack([m[:, i:i+1] for i in range(m.shape[1])])

        expanded_x = np.vstack([vect_xx, x])
        expanded_samples.append(expanded_x)

    expanded_DTR = np.hstack(expanded_samples)

    return expanded_DTR

