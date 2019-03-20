# -*- coding: utf-8 -*-
"""Created by lashkov on 31.10.18"""

import math

import numpy as np
from scipy.spatial.distance import cdist


def calc_nearest_points(X, labels, unique_labels, centroids, metric):
    """
    Calculation of coordinates of clusters points closest to their geometric centers

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    unique_labels : array-like (n_clusters,),
        Unique labels of clusters (k > 0)
    centroids : dict-like, 
        Key: cluster index, Value: n_features-dimensional data point
    metric : str,
        The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’. Default is ‘euclidean’.

    Returns
    -------
    centroids : dict-like, 
        Key: cluster index, Value: n_features-dimensional data point
    """
    centroids_p = dict()
    for i in unique_labels:
        dist = cdist(X[labels == i], np.array(centroids[i], ndmin=2), metric=metric)
        centroids_p[i] = X[labels == i][np.argmin(dist)]
    
    return centroids_p


def calc_centroids(X, unique_labels, labels, centr):
    """
    Calculation of coordinates of centroids

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    unique_labels : array-like (n_clusters,),
        Unique labels of clusters (k > 0)
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    centr : str,
        Cluster center calculation method (mean (default) or median)

    Returns
    -------
    centroids : dict-like, 
        Key: cluster index, Value: n_features-dimensional data point
    """
    centers = dict()
    for i in unique_labels:
        if centr == "mean":
            centers[i] = np.mean(X[labels == i], axis=0)
        elif centr == "median":
            centers[i] =  np.median(X[labels == i], axis=0)
    return centers


def filter_noise_lab(X, labels):
    """
    Filter noise points

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)

    Returns
    -------
    filterLabel : array-like, shape (n_samples,)
        Filtered predicted labels for each sample.
    filterXYZ : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point. Data points which label = -1 was removed.
    """
    filterLabel = labels[labels != -1]
    filterXYZ = X[labels != -1]
    return filterLabel, filterXYZ


def bind_noise_lab(X, labels, metric):
    """
    Bind noise points to nearest cluster

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)
    metric : str,
        The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’. Default is ‘euclidean’.

    Returns
    -------
    labels : array-like, shape (n_samples,)
        Modified predicted labels for each sample. to a single data point.
        Data points which label = -1 was bound to nearest clusters.
    """

    labels = labels.copy()
    if -1 not in set(labels):
        return labels
    if len(set(labels)) == 1 and -1 in set(labels):
        raise ValueError('Labels contains noise point only')
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
    """
    Definition of each noise point as a separate cluster

    Parameters
    ----------
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)

    Returns
    -------
    labels : array-like, shape (n_samples,)
        Modified predicted labels for each sample. to a single data point.
        Each data points which label = -1 was defined as a separate cluster.
    """
    labels = labels.copy()
    max_label = np.max(labels)
    j = max_label + 1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = j
            j += 1
    return labels


def comb_noise_lab(labels):
    """
    Combining all noise points into one cluster

    Parameters
    ----------
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.  (-1 - for noise)

    Returns
    -------
    labels : array-like, shape (n_samples,)
        Modified predicted labels for each sample. to a single data point.
        All data points which label = -1 was combined into a one cluster.
    """
    labels = labels.copy()
    max_label = np.max(labels)
    j = max_label + 1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = j
    return labels


def centroid_distance(X, unique_labels, centroids, metric):
    """
    Calculation of distances between cluster centers given by: (Dmax/Dmin) * sum{forall i in 1:|C|} 1 /( sum{forall j in 1:|C|} ||vi - vj|| )

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    unique_labels : array-like (n_clusters,),
        Unique labels of clusters (k > 0)
    centroids : dict-like, 
        Key: cluster index, Value: n_features-dimensional data point
    metric : str,
        The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
        ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,
        ‘yule’. Default is ‘euclidean’.

    Returns
    -------
    centroids : dict-like, 
        Key: cluster index, Value: n_features-dimensional data point
    """
    sum_dist = 0
    max_dist = 0
    min_dist = 1e6
    for i in unique_labels:
        module_dist = 0
        for j in unique_labels:
            dist = cdist(np.array(centroids[j], ndmin=2), np.array(centroids[i], ndmin=2), metric=metric)[0,0]
            module_dist += dist

            if dist > max_dist:
                max_dist = dist

            if dist < min_dist and dist > 0:
                min_dist = dist

        sum_dist += 1/module_dist

    distance = (max_dist/min_dist)*sum_dist

    return distance

def calc_density(X, centroids, labels, stdev, clusters_list, method, density_dict=None, lambd=0.7):
    """
    Compute the density of one or two cluster(depend on cluster_list)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    centroids : dict-like, 
        Key: cluster index, Value: n_features-dimensional data point
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    stdev : float,
        Average standard deviation of clusters (for Halkidi method [1])
    clusters_list : list,
        List contains one or two No. of cluster
    method : str,
        S_Dbw calc method:
        'Halkidi' - original paper [1]
        'Kim' - see [2]
        'Tong' - see [3]
    density_dict : dict(),
        List contains density of each cluster for calculate muij [3]
    lambd : float,
        Lambda coefficient is a positive constant between 0 and 1 (default: 0.7, see [3]

    Returns
    -------
    score : float
        Density of one or two cluster

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, “Clustering validity assessment: Finding the optimal partitioning
        of a data set,” in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    .. [2] Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD’2003, Seoul, Korea, April 30–May 2,
        2003, LNAI 2637, 602–608
    .. [3] Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
    """

    density = 0
    center_p1 = centroids[clusters_list[0]]
    if len(clusters_list) == 2:
        center_p2 = centroids[clusters_list[1]]
        if method == 'Kim' or method == 'Tong':
            sigmai = np.std(X[labels == clusters_list[0]], axis=0)
            sigmaj = np.std(X[labels == clusters_list[1]], axis=0)
            sigmaij = (sigmai + sigmaj) / 2
            ni = X[labels == clusters_list[0]].shape[0]
            nj = X[labels == clusters_list[1]].shape[0]
            nij = ni + nj
            if method == 'Tong':
                center_v = lambd * (center_p1 * nj + center_p2 * ni) / nij + \
                           (1 - lambd) * ((center_p1 * density_dict[clusters_list[0]] +
                                           center_p2 * density_dict[clusters_list[1]]) /
                                          (density_dict[clusters_list[0]] + density_dict[clusters_list[1]]))
            else:
                center_v = (center_p1 + center_p2) / 2
        else:
            center_v = (center_p1 + center_p2) / 2
    else:
        center_v = center_p1
        if method == 'Kim' or method == 'Tong':
            sigmaij = np.std(X[labels == clusters_list[0]], axis=0)
            nij = X[labels == clusters_list[0]].shape[0]
    if method == 'Halkidi':
        for i in clusters_list:
            temp = X[labels == i]
            for j in temp:
                if np.linalg.norm(j - center_v) <= stdev:
                    density += 1
    elif method == 'Kim' or method == 'Tong':
        CI = 1.96 * sigmaij / math.sqrt(nij)
        for i in clusters_list:
            temp = X[labels == i]
            for j in temp:
                if np.all(np.abs(j - center_v) <= CI):
                    density += 1

    return density


def Dens_bw(X, centroids, labels, unique_labels, method='Halkidi'):
    """
    Compute Inter-cluster Density (ID) - It evaluates the average density in the region among clusters in relation
    with the density of the clusters. The goal is the density among clusters to be significant low in comparison with
    the density in the considered clusters [1].

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    centroids : dict-like, 
        Key: cluster index, Value: n_features-dimensional data point
    unique_labels : array-like (n_clusters,)
        Unique labels of clusters (k > 0)
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    method : str,
        S_Dbw calc method:
        'Halkidi' - original paper [1]
        'Kim' - see [2]
        'Tong' - see [3]

    Returns
    -------
    score : float
        Inter-cluster Density

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, “Clustering validity assessment: Finding the optimal partitioning
        of a data set,” in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    .. [2] Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD’2003, Seoul, Korea, April 30–May 2,
        2003, LNAI 2637, 602–608
    .. [3] Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
    """
    k = np.size(unique_labels)
    density_dict = dict()
    result = 0
    stdev = 0
    if method == 'Halkidi':
        for i in unique_labels:
            std_matrix_i = np.std(X[labels == i], axis=0)
            stdev += math.sqrt(np.dot(std_matrix_i.T, std_matrix_i))
        stdev = math.sqrt(stdev) / k
    for i in unique_labels:
        density_dict[i] = calc_density(X, centroids, labels, stdev, [i], method)
    
    count_zeros = 0
    for cluster in density_dict.keys():
        if density_dict[cluster] == 0:
            count_zeros += 1 
        
        if count_zeros > 1:
            raise ValueError('The density for two or more clusters to equal zero.')
    
    for i in unique_labels:
        for j in unique_labels:
            if i == j:
                continue
            result += calc_density(X, centroids, labels, stdev, [i, j], method, density_dict) / max(density_dict[i], density_dict[j])
    
    return result / (k * (k - 1))


def Scat(X, unique_labels, labels, method):
    """
    Calculate intra-cluster variance (Average scattering for clusters).
    Lower value -> better clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    unique_labels : array-like (n_clusters,)
        Unique labels of clusters (k > 0)
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.
    method : str,
        S_Dbw calc method:
        'Halkidi' - original paper [1]
        'Kim' - see [2]
        'Tong' - see [3]

    Returns
    -------
    score : float
        Average scattering for clusters

    References:
    -----------
    .. [1] M. Halkidi and M. Vazirgiannis, “Clustering validity assessment: Finding the optimal partitioning
        of a data set,” in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    .. [2] Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD’2003, Seoul, Korea, April 30–May 2,
     2003, LNAI 2637, 602–608
    .. [3] Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
    """
    k = np.size(unique_labels)
    theta_s = np.std(X, axis=0)
    theta_s_2norm = math.sqrt(np.dot(theta_s.T, theta_s))
    sum_theta_2norm = 0
    if method == 'Halkidi':
        for i in unique_labels:
            theta_i = np.std(X[labels == i], axis=0)
            sum_theta_2norm += math.sqrt(np.dot(theta_i.T, theta_i))
        result = sum_theta_2norm / (theta_s_2norm * k)
    else:
        n = len(labels)
        for i in unique_labels:
            ni = X[labels == i].shape[0]
            theta_i = np.std(X[labels == i], axis=0)
            sum_theta_2norm += ((n - ni) / n) * math.sqrt(np.dot(theta_i.T, theta_i))
        result = sum_theta_2norm / (theta_s_2norm * k)
        if method == 'Tong':
            result = sum_theta_2norm / (theta_s_2norm * (k - 1))
    return result


def S_Dbw(X, labels, centers_id=None, method='Tong', alg_noise='comb',
          centr='mean', nearest_centr=True, metric='euclidean'):
    """
    Compute the S_Dbw validity index
    S_Dbw validity index is defined by equation:
    S_Dbw = scatt + dens
    where scatt - means average scattering for clusters and dens - inter-cluster density.
    Lower value -> better clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample (-1 - for noise).
    centers_id : array-like, shape (n_samples,)
        The center_id of each cluster's center. If None - cluster's center calculate automatically.
    method : str,
        S_Dbw calc method:
        'Halkidi' - original paper [1]
        'Kim' - see [2]
        'Tong' - see [3]
    alg_noise : str,
        Algorithm for recording noise points.
        'comb' - combining all noise points into one cluster (default)
        'sep' - definition of each noise point as a separate cluster
        'bind' -  binding of each noise point to the cluster nearest from it
        'filter' - filtering noise points
    centr : str,
        cluster center calculation method (mean (default) or median)
    nearest_centr : bool,
        The centroid corresponds to the cluster point closest to the geometric center (default: True).
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
    .. [1] M. Halkidi and M. Vazirgiannis, “Clustering validity assessment: Finding the optimal partitioning
        of a data set,” in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    .. [2] Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD’2003, Seoul, Korea, April 30–May 2,
        2003, LNAI 2637, 602–608
    .. [3] Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
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
    unique_labels = np.unique(labels)

    if centers_id:
        centroids = dict()
        for index, label in enumerate(unique_labels):
            centroids[label] = X[centers_id[index]]
    else:
        centroids = calc_centroids(X, unique_labels, labels, centr)
        if nearest_centr:
            centroids = calc_nearest_points(X, labels, unique_labels, centroids, metric)
    if np.size(unique_labels) < 2:
        raise ValueError('Only one cluster!')
    sdbw = Dens_bw(X, centroids, labels, unique_labels, method) + Scat(X, unique_labels, labels, method)
    return sdbw



def SD(X, labels, centers_id=None, centr='mean', nearest_centr=True, metric='euclidean'):
    """
    Compute the SD validity index
    SD validity index is defined by equation:
    S_Dbw = k*scatt + distance
    where scatt - means average scattering for clusters and distance - distances between cluster centers.
    Lower value -> better clustering.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample (-1 - for noise).
    centers_id : array-like, shape (n_samples,)
        The center_id of each cluster's center. If None - cluster's center calculate automatically.
    centr : str,
        cluster center calculation method (mean (default) or median)
    nearest_centr : bool,
        The centroid corresponds to the cluster point closest to the geometric center (default: True).
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
    .. [4] M. Halkidi and M. Vazirgiannis, “Clustering validity assessment: Finding the optimal partitioning
        of a data set,” in ICDM, Washington, DC, USA, 2001, pp. 187–194.
    """

    if len(set(labels)) < 2 or len(set(labels)) > len(X) - 1:
        raise ValueError("No. of unique labels must be > 1 and < n_samples")
    unique_labels = np.unique(labels)

    if centers_id:
        centroids = dict()
        for index, label in enumerate(unique_labels):
            centroids[label] = X[centers_id[index]]
    else:
        centroids = calc_centroids(X, unique_labels, labels, centr)
        if nearest_centr:
            centroids = calc_nearest_points(X, labels, unique_labels, centroids, metric)
    if np.size(unique_labels) < 2:
        raise ValueError('Only one cluster!')
    
    sd = Scat(X, unique_labels, labels, 'Halkidi')*centroid_distance(X, unique_labels, centroids, metric) + centroid_distance(X, unique_labels, centroids, metric)

    return sd
