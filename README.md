# FRKF

This is a Python library of data assimilation and optimization methods developed to solve large-scale state-parameter estimation problems. Data assimilation approach has been widely used in geophysics, hydrology and numerical weather forecast to improve the forecast from a numerical model based on data sets collected in real time. Conventional approach like Kalman filter is computatioanlly prohibitive for large problems. The FRKF library takes advantage of techniques such as Principle Component Analysis (PCA), Fast Linear Algebra, and utilizing cross-covariance matrices to develop Fast and Robust data assimilation approach that is scalable for large-scale problems. The functions and examples are produced based on collabrated work on data assimialtion documented in the papers listed in the Reference section.

# This is an ongoing project. The library is incomplete.

This library includes the following Fast and Robust Kalman Filter algorithms:

- SpecKF
- MCSKF
- CSKF
- RFC
- HiKF
- EnKF
- KF

|  Method  |  Assumptions                 |  Jacobian matrix|  Covariance matrix |   
| -------: |:----------------------------:|:------------------------: |:--------------|
|  KF      | only for small problem       | mn forward run| m^2 operations |
|  SpecKF  | n<< m, approximate uncertainty| p forward run|  O(m) operations|
|  CSKF    |    smooth problem, n<<m            | r forward run | O(m) operations|
|  MCSKF    |    smooth problem           | r forward run | O(m) operations|
|  FRC    | Small number of controllers    |n forward run |  O(m) operations |
|  HiKF    | fast data acquisition/n<<m| no forward run| O(m) operations |
|          | random-walk forward model    | |  |
|  EnKF    | monte carlo based approach   | r forward run | O(m) operations|
