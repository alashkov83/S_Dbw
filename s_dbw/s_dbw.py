#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created by lashkov on 31.10.18"""

import numpy as np
import math
from scipy.spatial.distance import cdist


def bind_noise_lab(X, labels):
    labels = labels.copy()
    label_id = []
    label_new = []
    for i in range(len(labels)):
        if labels[i] == -1:
            point = np.array([X[i]])
            dist = cdist(X[labels != -1], point)
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

def center_of_mass(str_nparray):
    mass_sum = str_nparray.shape[0]
    dim = str_nparray.shape[1]
    center_m = str_nparray.sum(axis=0) / mass_sum
    center_m.shape = (-1, dim)
    return center_m


def get_center_id(data, labels):
    center_id = []
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(n_clusters):
        c_mass = center_of_mass(data[labels == i])
        dist = cdist(data[labels == i], c_mass)
        center_id.append(np.where(np.all(data == data[labels == i][np.argmin(dist), :], axis=1))[0][0])
    return np.array(center_id)


def density(data, centers_id, labels, stdev, density_list):
    """
    compute the density of one or two cluster(depend on density_list)
    """
    density = 0
    centers_id1 = centers_id[density_list[0]]
    if len(density_list) == 2:
        centers_id2 = centers_id[density_list[1]]
        center_v = (data[centers_id1] + data[centers_id2]) / 2
    else:
        center_v = data[centers_id1]
    for i in density_list:
        temp = data[labels == i]
        for j in temp:
            if np.linalg.norm(j - center_v) <= stdev:
                density += 1
    return density


def Dens_bw(data, centers_id, labels, stdev, k):
    density_list = []
    result = 0
    for i in range(k):
        density_list.append(density(data, centers_id, labels, stdev, density_list=[i]))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            result += density(data, centers_id, labels, stdev, [i, j]) / max(density_list[i], density_list[j])
    return result / (k * (k - 1))


def Scat(data, k, labels):
    theta_s = np.std(data, axis=0)
    theta_s_2norm = math.sqrt(np.dot(theta_s.T, theta_s))
    sum_theta_2norm = 0
    for i in range(k):
        matrix_data_i = data[labels == i]
        theta_i = np.std(matrix_data_i, axis=0)
        sum_theta_2norm += math.sqrt(np.dot(theta_i.T, theta_i))
    return sum_theta_2norm / (theta_s_2norm * k)


def S_Dbw(X, labels, centers_id=None, noise=False, alg_noise='sep'):
    """
    X --> raw data
    data_cluster --> The category that represents each piece of data(the number of category should begin 0)
    centers_id --> the center_id of each cluster's center
    """
    if len(set(labels)) < 2:
        raise ValueError("No. of unique labels must be > 1")
    if noise:
        if alg_noise == 'sep':
            labels = sep_noise_lab(labels)
        elif alg_noise == 'bind':
            labels = bind_noise_lab(X, labels)
    if centers_id is None:
        centers_id = get_center_id(X, labels)
    k = len(centers_id)
    stdev = 0
    for i in range(k):
        std_matrix_i = np.std(X[labels == i], axis=0)
        stdev += math.sqrt(np.dot(std_matrix_i.T, std_matrix_i))
    stdev = math.sqrt(stdev) / k
    sdbw = Dens_bw(X, centers_id, labels, stdev, k) + Scat(X, k, labels)
    return sdbw


if __name__ == '__main__':
    from sklearn.cluster import DBSCAN
    from sklearn.datasets.samples_generator import make_blobs

    n_samples = 1500
    centers = [[1, 1], [-2, -2], [3, 3]]
    X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.4,
                                random_state=0)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X = np.dot(X, transformation)

    db = DBSCAN(eps=0.14).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(S_Dbw(X, labels, noise=True, alg_noise='bind'))