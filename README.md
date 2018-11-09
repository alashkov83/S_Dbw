# S_Dbw
Compute the S_Dbw validity index  
S_Dbw validity index is defined by equation:
##### S_Dbw = Scatt + Dens_bw
where Scatt - means average scattering for clusters  and Dens_bw - inter-cluster density.  
**Lower value -> better clustering.**

Installation
------------

```shell
pip install --upgrade s-dbw
```

Usage
-----
```python
from s_dbw import S_Dbw
score = S_Dbw(X, labels, centers_id=None, method='Tong', alg_noise='bind',
centr='mean', nearest_centr=True, metric='euclidean')

```
### Parameters:
* X : array-like, shape (n_samples, n_features)  
    List of n_features-dimensional data points. Each row corresponds to a single data point.  
* labels : array-like, shape (n_samples,)  
    Predicted labels for each sample (-1 - for noise).  
* centers_id : array-like, shape (n_samples,)  
    The center_id of each cluster's center. If None - cluster's center calculate automatically.  
* method : str,  
    S_Dbw calc method:  
    'Halkidi' - original paper \[1]  
    'Kim' - see \[2]  
    'Tong' - see \[3]  
* alg_noise : str,  
    Algorithm for recording noise points.  
    'comb' - combining all noise points into one cluster (default)  
    'sep' - definition of each noise point as a separate cluster  
    'bind' -  binding of each noise point to the cluster nearest from it  
    'filter' - filtering noise points  
* centr : str,  
    cluster center calculation method (mean (default) or median)  
* nearest_centr : bool,  
    The centroid corresponds to the cluster point closest to the geometric center (default: True).  
* metric : str,  
    The distance metric, can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,  
    ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,  
    ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’,‘yule’.  
    Default is ‘euclidean’.  

### Returns
score : float  
    The resulting S_DBw score.  

References:
-----------
1. M. Halkidi and M. Vazirgiannis, “Clustering validity assessment: Finding the optimal partitioning of a data set,” in ICDM, Washington, DC, USA, 2001, pp. 187–194.
2. Youngok Kim and Soowon Lee. A clustering validity assessment Index. PAKDD’2003, Seoul, Korea, April 30–May 2, 2003, LNAI 2637, 602–608
3. Tong, J. & Tan, H. J. Electron.(China) (2009) 26: 258. https://doi.org/10.1007/s11767-007-0151-8
