# FRKF

This is a Python library of data assimilation and optimization methods developed to solve large-scale state-parameter estimation problems. Data assimilation approach has been widely used in geophysics, hydrology and numerical weather forecast to improve the forecast from a numerical model based on data sets collected in real time. Conventional approach like Kalman filter is computatioanlly prohibitive for large problems. The FRKF library takes advantage of techniques such as Principle Component Analysis (PCA), Fast Linear Algebra, and utilizing cross-covariance matrices to develop Fast and Robust data assimilation approach that is scalable for large-scale problems. The functions and examples are produced based on collabrated work on data assimialtion documented in the papers listed in the Reference section.

# This is an ongoing project! Needs to be updated.

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



<img width="540" alt="screen shot 2018-02-07 at 11 54 43 am" src="https://user-images.githubusercontent.com/7990350/35938453-fd317d06-0bfd-11e8-93a0-475e2219617e.png">



## Quick Start Guide


## Bathymetry example


![image](https://user-images.githubusercontent.com/7990350/35986974-652596d4-0caf-11e8-9596-68ba2792f349.png)


![image](https://user-images.githubusercontent.com/7990350/35987094-b237f980-0caf-11e8-94e2-1d60c2a0b363.png)

