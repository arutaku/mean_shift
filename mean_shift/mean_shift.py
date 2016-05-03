#!/usr/bin/env python

"""Mean shift clustering algorithm.

This is a parallel implementation of the Mean Shift algorithm allowing flat, gaussian
and custom kernels, that are not allowed in the scikit-learn implementation.

Parallelization is supported thanks to JobLib.

For more information:
    https://en.wikipedia.org/wiki/Mean_shift
    https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/cluster/mean_shift_.py

Contributors:
    Alberto Rubio <alberto.rubio.munoz@gmail.com>

Under GPL v3
"""

import numpy as np
from sklearn.neighbors import BallTree
from sklearn.utils import extmath
from joblib import Parallel, delayed

from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import euclidean_distances


def gaussian_kernel(x, points, bandwidth):
    """Return the evalutation of points in x using a gaussian kernel for the provided bandwidth."""
    distances = euclidean_distances(points, x.reshape(1,2))
    weights = np.exp(-1 * (distances ** 2 / bandwidth ** 2))
    return np.sum(points * weights, axis=0) / np.sum(weights)

def flat_kernel(x, points, bandwidth):
    """Return the evalutation of points in x using a flat kernel."""
    return np.mean(points, axis=0)

def _iter(X,
          weighted_mean,
          kernel_update_function,
          bandwidth,
          ball_tree,
          stop_thresh,
          max_iter):
    """Return the cluster center and the within points visited while iterated from the seed
    to the centroid. This code has been isolated to be executed in parallel using JobLib."""
    visited_points = set()
    completed_iterations = 0
    while True:
        within_idx = ball_tree.query_radius([weighted_mean], bandwidth*3)[0]
        [visited_points.add(x) for x in within_idx]
        points_within = X[within_idx]
        old_mean = weighted_mean  # save the old mean
        weighted_mean = kernel_update_function(old_mean, points_within, bandwidth)
        converged = extmath.norm(weighted_mean - old_mean) < stop_thresh
        if converged or completed_iterations == max_iter:
            return weighted_mean, visited_points
        completed_iterations += 1


class MeanShift(object):
    """Mean shift clustering using a flat, gaussian or custom kernel.

    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given region.

    Parameters
    ----------
    kernel_func : function, default gaussian kernel
        The function that implements the desired kernel.

    bandwidth : float, optional
        Bandwidth used in the provided kernel.

        If you do not know which one to choose, see sklearn.cluster.estimate_bandwidth.

    seeds : array, shape=[n_samples, n_features], optional
        Seeds used to initialize kernels. If not set, the seeds are the provided X points.

    max_iter : int, default 300
        Maximum number of allowed interations from each seed to the convergence point.
        In order to avoid infinite loops or convergence issues.

    n_jobs : int, default 1
        The number of jobs to use for the computation.
        This works by computing each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        Thus for n_jobs = -2, all CPUs but one are used.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
                
    cluster_centers_by_popularity_ : array, [n_clusters, n_features + 1 (% of paths in this cluster)])
        An array of coordinates of cluster centers and the number of points
        associated to this centroid. The array is sorted in descending order of n_points_in_cluster.
                
    labels_ :
        Labels of each point. Associated to the index of cluster_centers_
        (cluster_centers_by_popularity is sorted after being computed, so index may not match).
    """

    def __init__(self,
                 kernel_func,
                 bandwidth,
                 seeds=None,
                 max_iter=300,
                 n_jobs=1):
        self.kernel_func = kernel_func
        if bandwidth is None:
            raise ValueError('See bandwidth estimator')
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Mean-shift implementation that allows custom kernels and parallelization.

        Parameters
        ----------
            X: array, [n_points, n_features]
                Input data to be clustered.
        """
        n_points, n_features = X.shape
        stop_thresh = 1e-3 * self.bandwidth # When mean has converged
        centroids = []
        ball_tree = BallTree(X)             # Efficiently look up nearby points
        if self.seeds is None:
            self.seeds = X
        n_seeds = self.seeds.shape[0]
        if self.n_jobs == 1:
            # When parallelization is not required avoid the usage of JobLib
            for weighted_mean in self.seeds:
                centroids.append(_iter(X, weighted_mean, self.kernel_func,
                                       self.bandwidth, ball_tree, stop_thresh, self.max_iter))
        else:
            centroids = Parallel(n_jobs=self.n_jobs)(
            delayed(_iter)(X, weighted_mean, self.kernel_func,
                           self.bandwidth, ball_tree, stop_thresh, self.max_iter
                ) for weighted_mean in self.seeds)
        # Find centroids that converged to (almost) the same point
        arr_centers = np.array([x[0] for x in centroids])
        Z = linkage(arr_centers)
        clusters = fcluster(Z, self.bandwidth, criterion='distance')

        n_points = X.shape[0]
        final_clusters = []
        points_2_cluster = np.zeros((n_points, 0))
        for centroid_idx in {x for x in clusters}:
            # Determine the centroid coordinates averaging close centroids
            # that belong to the same cluster
            final_clusters.append((np.mean(arr_centers[clusters == centroid_idx], 0),
                                   float(sum(clusters == centroid_idx)) / n_seeds))
            points_2_cluster = np.hstack([points_2_cluster, np.zeros((n_points, 1))])
            # For every "centroid" gathered to this centroid_idx
            for within_pts in [curr_centroid[1] for in_centroid, curr_centroid in\
                    zip(clusters == centroid_idx, centroids) if in_centroid]:
                # For each within point, count how many times belonged to a path that converged
                # to this centroid. It will be useful when computing final labels
                for pt in within_pts:
                    points_2_cluster[pt, -1] += 1
        # Formatting attributes
        self.cluster_centers_by_popularity_ = np.array(sorted(final_clusters, key=lambda x: -x[1]))
        self.cluster_centers_ = np.array([x[0] for x in final_clusters])
        self.labels_ = np.argmax(points_2_cluster, axis=1)

    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X)
        return self.labels_
