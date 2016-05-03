# mean_shift

Mean Shift [1] parallel implementation allowing different kernels (even custom ones) under GPLv3.

The main motivation of this implementation is the need to use a gaussian kernel while using the scikit-learn implementation of Mean Shift (only flat kernel was available) and take advantage of the multiple cores of my computer allowing parallel execution when dealing with big datasets like [2], while reproducing the experiment described in [3].

## Features

 - scikit-learn interface
 - Kernel flexibility:
   - Predefined flat and gaussian kernels
   - Custom kernels are also allowed
 - Parallel execution using JobLib
 
## Examples

The following images show the output produced by the below example:

<img src="https://raw.githubusercontent.com/arutaku/mean_shift/master/images/figure_1.png">
<img src="https://raw.githubusercontent.com/arutaku/mean_shift/master/images/figure_2.png">
<img src="https://raw.githubusercontent.com/arutaku/mean_shift/master/images/figure_3.png">

Sometimes it cannot find the a priori number of clusters. But in this case, even I would also have failed miserably!

<img src="https://raw.githubusercontent.com/arutaku/mean_shift/master/images/figure_4.png">

## [TODO] Installation

Make sure you have numpy, scikit-learn and joblib installed. Then do the following:

```
 git clone git://github.com/arutaku/mean_shift.git
 cd mean_shift
 sudo python setup.py install
```

## Usage

Matplotlib is also needed to run the example in order to show the results.

```python
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

import mean_shift as ms

# Dataset generation
n_clusters = 7
X, y = datasets.make_blobs(n_samples=2000, centers=n_clusters, cluster_std=np.random.normal(1, .3, n_clusters))
# Only take 100 seeds from the whole dataset
seeds = X[np.random.randint(X.shape[0], size=100), :]

# Model fitting
model = ms.MeanShift(kernel_func=ms.gaussian_kernel, bandwidth=1, seeds=seeds, n_jobs=-1)
y_prime = model.fit_predict(X)

# Plotting stuff
n_clusters_found = np.unique(y_prime).size
print('Expected {} clusters, found {}'.format(n_clusters, n_clusters_found))
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].set_title('A priori known number of clusters {}'.format(n_clusters))
ax[0].scatter(X[:,0], X[:,1], c=y, cmap='spring')
ax[1].set_title('Data without labels passed to Mean Shift')
ax[1].scatter(X[:,0], X[:,1])
ax[2].set_title('Clusters found ({} out of {})'.format(n_clusters_found, n_clusters))
ax[2].scatter(X[:,0], X[:,1], c=y_prime, cmap='rainbow')
plt.axis('equal')
plt.show()
```

## References

1. Fukunaga, Keinosuke; Larry D. Hostetler (January 1975). "The Estimation of the Gradient of a Density Function, with Applications in Pattern Recognition". IEEE Transactions on Information Theory (IEEE) 21 (1): 32–40.
2. Hatem Mousselly-Sergieh, Daniel Watzinger, Bastian Huber, Mario Döller, Elöd Egyed-Zsigmond, and Harald Kosch. 2014. World-wide scale geotagged image dataset for automatic image annotation and reverse geotagging. In Proceedings of the 5th ACM Multimedia Systems Conference (MMSys '14). ACM, New York, NY, USA, 47-52.
3. David J. Crandall, Lars Backstrom, Daniel Huttenlocher, and Jon Kleinberg. 2009. Mapping the world's photos. In Proceedings of the 18th international conference on World wide web (WWW '09). ACM, New York, NY, USA, 761-770.
4. Cheng, Yizong (August 1995). "Mean Shift, Mode Seeking, and Clustering". IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE) 17 (8): 790–799.
5. Comaniciu, Dorin; Peter Meer (May 2002). "Mean Shift: A Robust Approach Toward Feature Space Analysis". IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE) 24 (5): 603–619.
6. Aliyari Ghassabeh, Youness (2015-03-01). "A sufficient condition for the convergence of the mean shift algorithm with Gaussian kernel". Journal of Multivariate Analysis 135: 1–10.
