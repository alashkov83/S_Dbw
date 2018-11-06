# -*- coding: utf-8 -*-
"""Created by lashkov on 31.10.18"""

import math

import numpy as np
from scipy.spatial.distance import cdist


def filter_noise_lab(X, Label):
    """

    :param Label:
    :param X:
    :return:
    """
    filterLabel = Label[Label != -1]
    filterXYZ = X[Label != -1]
    return filterLabel, filterXYZ


def bind_noise_lab(X, labels, metric):
    labels = labels.copy()
    if -1 not in set(labels):
        return labels
    if len(set(labels)) == 1 and -1 in set(labels):
        raise ValueError('Labels contain noise point only')
    label_id = []
    label_new = []
    for i in range(len(labels)):
        if labels[i] == -1:
            point = np.array([X[i]])
            dist = cdist(X[labels != -1], point, metric=metric)
            lid = np.where(np.all(X == X[labels != -1][np.argmin(dist), :], axis=1))[0][0]
            label_id.append(i)
            label_new.append(labels[lid])
    labels[np.array(label_id)] = np.array(label_new)
    return labels


def sep_noise_lab(labels):
    labels = labels.copy()
    max_label = np.max(labels)
    j = max_label + 1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = j
            j += 1
    return labels


def comb_noise_lab(labels):
    labels = labels.copy()
    max_label = np.max(labels)
    j = max_label + 1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = j
    return labels


def center_of_mass(str_nparray):
    mass_sum = str_nparray.shape[0]
    dim = str_nparray.shape[1]
    center_m = str_nparray.sum(axis=0) / mass_sum
    center_m.shape = (-1, dim)
    return center_m


def get_center_id(X, labels, metric):
    center_id = []
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(n_clusters):
        c_mass = center_of_mass(X[labels == i])
        dist = cdist(X[labels == i], c_mass, metric=metric)
        center_id.append(np.where(np.all(X == X[labels == i][np.argmin(dist), :], axis=1))[0][0])
    return np.array(center_id)


def density(X, centers_id, labels, stdev, density_list):
    """
    compute the density of one or two cluster(depend on density_list)
    """
    density = 0
    centers_id1 = centers_id[density_list[0]]
    if len(density_list) == 2:
        centers_id2 = centers_id[density_list[1]]
        center_v = (X[centers_id1] + X[centers_id2]) / 2
    else:
        center_v = X[centers_id1]
    for i in density_list:
        temp = X[labels == i]
        for j in temp:
            if np.linalg.norm(j - center_v) <= stdev:
                density += 1
    return density


def Dens_bw(X, centers_id, labels, stdev, k):
    density_list = []
    result = 0
    for i in range(k):
        density_list.append(density(X, centers_id, labels, stdev, density_list=[i]))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            result += density(X, centers_id, labels, stdev, [i, j]) / max(density_list[i], density_list[j])
    return result / (k * (k - 1))


def Scat(X, k, labels):
    theta_s = np.std(X, axis=0)
    theta_s_2norm = math.sqrt(np.dot(theta_s.T, theta_s))
    sum_theta_2norm = 0
    for i in range(k):
        matrix_data_i = X[labels == i]
        theta_i = np.std(matrix_data_i, axis=0)
        sum_theta_2norm += math.sqrt(np.dot(theta_i.T, theta_i))
    return sum_theta_2norm / (theta_s_2norm * k)


def S_Dbw(X, labels, centers_id=None, alg_noise='comb', metric='euclidean'):
    """
    Compute the S_Dbw validity index
    S_Dbw validity index is defined by equation:
    S_Dbw = scatt + dens
    where scatt - means average scattering for clusters and dens - inter-cluster density.
    Lower value -> better clustering.

    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample (-1 - for noise).
    centers_id : array-like, shape (``n_samples``,)
        The center_id of each cluster's center. If None - cluster's center calculate automatically.
    alg_noise : str,
        Algorithm for recording noise points.
        'comb' - combining all noise points into one cluster (default)
        'sep' - definition of each noise point as a separate cluster
        'bind' -  binding of each noise point to the cluster nearest from it
        'filter' - filtering noise points
    metric : str,
        The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’. Default is ‘euclidean’.

    Returns
    -------
    score : float
        The resulting S_DBw score.

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, “Clustering validity assess-
        ment: Finding the optimal partitioning of a data set,” in
        ICDM, Washington, DC, USA, 2001, pp. 187–194.
        <https://pdfs.semanticscholar.org/dc44/df745fbf5794066557e52074d127b31248b2.pdf>
    .. [2] Understanding of Internal Clustering Validation Measures
        <http://datamining.rutgers.edu/publication/internalmeasures.pdf>
    """
    if len(set(labels)) < 2 or len(set(labels)) > len(X) - 1:
        raise ValueError("No. of unique labels must be > 1 and < n_samples")
    if alg_noise == 'sep':
        labels = sep_noise_lab(labels)
    elif alg_noise == 'bind':
        labels = bind_noise_lab(X, labels, metric=metric)
    elif alg_noise == 'comb':
        labels = comb_noise_lab(labels)
    elif alg_noise == 'filter':
        labels, X = filter_noise_lab(X, labels)
    if centers_id is None:
        centers_id = get_center_id(X, labels, metric=metric)
    k = len(centers_id)
    if k < 2:
        raise ValueError('Only one cluster!')
    stdev = 0
    for i in range(k):
        std_matrix_i = np.std(X[labels == i], axis=0)
        stdev += math.sqrt(np.dot(std_matrix_i.T, std_matrix_i))
    stdev = math.sqrt(stdev) / k
    sdbw = Dens_bw(X, centers_id, labels, stdev, k) + Scat(X, k, labels)
    return sdbw
