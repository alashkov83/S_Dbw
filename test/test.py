import unittest
import numpy as np
from sklearn.cluster import DBSCAN
# from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
epsilon = 0.000000000000001

# 1 - SIMPLE DATA TESTS
simple_data = np.array([[1, 2, 1], [0, 1, 4], [3, 3, 3], [2, 2, 2]])
simple_data_cluster = np.array([1, 0, 1, 2])  # The category represents each piece of data belongs
simple_centers_id = np.array([1, 0, 3])  # the cluster's num is 3

# 2 - ANISO BLOB DATA TEST
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


class MyTestCase(unittest.TestCase):

    def test_simplenewversion(self):
        from s_dbw import S_Dbw
        value = S_Dbw(simple_data, simple_data_cluster)  # 0.2886751345948128
        self.assertTrue(0.2886751345948128 - epsilon < value < 0.2886751345948128 + epsilon,
                        msg='test 1 = {:.16f}, must be 0.2886751345948128'.format(value))

    def test_simpleeq(self):
        from original_sdbw import S_Dbw
        a = S_Dbw(simple_data, simple_data_cluster, simple_centers_id)
        value_old = a.S_Dbw_result()  # 0.2886751345948128
        from s_dbw import S_Dbw
        value = S_Dbw(simple_data, simple_data_cluster)  # 0.2886751345948128
        self.assertEqual(value_old, value,
                         msg='test 2 = {:.16f}, must be 0.2886751345948128'.format(value))

    def test_anisodbnewversioncombnoise(self):
        from s_dbw import S_Dbw
        value = S_Dbw(X, labels)  # 1.4045566925764599
        self.assertTrue(1.4045566925764599 - epsilon < value < 1.4045566925764599 + epsilon,
                        msg='test 3 = {:.16f}, must be 1.4045566925764599'.format(value))

    def test_anisodbnewversionbindnoise(self):
        from s_dbw import S_Dbw
        value = S_Dbw(X, labels, alg_noise='bind')  # 1.2233006502166595
        self.assertTrue(1.2233006502166595 - epsilon < value < 1.2233006502166595 + epsilon,
                        msg='test 4 = {:.16f}, must be 1.2233006502166595'.format(value))

    def test_anisodbnewversionsepnoise(self):
        from s_dbw import S_Dbw
        value = S_Dbw(X, labels, alg_noise='sep')  # 0.3844372683801507
        self.assertTrue(0.3844372683801507 - epsilon < value < 0.3844372683801507 + epsilon,
                        msg='test 5 = {:.16f}, must be 0.3844372683801507'.format(value))


if __name__ == '__main__':
    unittest.main()
