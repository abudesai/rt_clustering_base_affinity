Affinity Propagation Model build in Sklearn for Clustering - Base problem category as per Ready Tensor specifications.

- sklearn
- python
- pandas
- numpy
- scikit-optimize
- docker
- clustering

This is a Clustering Model that uses Affinity Propagation implemented through Sklearn.

AffinityPropagation creates clusters by sending messages between pairs of samples until convergence. A dataset is then described using a small number of exemplars, which are identified as those most representative of other samples. The messages sent between pairs represent the suitability for one sample to be the exemplar of the other, which is updated in response to the values from other pairs. This updating happens iteratively until convergence, at which point the final exemplars are chosen, and hence the final clustering is given.

The data preprocessing step includes:

- for numerical variables
  - Standard scale data

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as Computer Activity, Heart Disease, Satellite, Statlog, Tripadvisor, and White Wine.

This Clustering Model is written using Python as its programming language. ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT.
