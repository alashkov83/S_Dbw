# S_Dbw
Compute the S_Dbw validity index  
S_Dbw validity index is defined by equation:  
```S_Dbw = scatt + dens```  
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
[1] M. Halkidi and M. Vazirgiannis, “Clustering validity assessment:
    Finding the optimal partitioning of a data set,” in
    ICDM, Washington, DC, USA, 2001, pp. 187–194.
    <https://pdfs.semanticscholar.org/dc44/df745fbf5794066557e52074d127b31248b2.pdf>  
[2] Understanding of Internal Clustering Validation Measures  
    <http://datamining.rutgers.edu/publication/internalmeasures.pdf>
