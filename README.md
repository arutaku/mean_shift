# mean_shift
Multikernel and parallel implementation in python of the popular density based clustering algorithm: Mean Shift

The main motivation of this implementation is the need to use a gaussian kernel while using the scikit-learn implementation of Mean Shift (only the flat kernel is available). There are some implementations of Mean Shift gaussian kernels, but I have had some performance issues using them.

## Features
 - scikit-learn interface
 - Kernel flexibility:
   - Predefined flat and gaussian kernels
   - Custom kernels are also allowed
 - Parallel execution using JobLib
