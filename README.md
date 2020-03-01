
Metric Learning with Submodular Functions

## Abstract

  Most of the metric learning mainly focuses on using single feature weights with $L_p$ norms, or the pair of features with Mahalanobis distances to learn the similarities between the samples, while ignoring the potential value of higher-order interactions in the feature space. In this paper, we investigate the possibility of learning weights to coalitions of features whose cardinality can be greater than two, with the help of set-functions. With the more particular property of submodular set-function, we propose to define a metric for continuous features based on Lovasz extension of submodular functions, and then present a dedicated metric learning approach. According to the submodular constraints, it naturally leads to a higher complexity price so that we use the $\xi$-additive fuzzy measure to decrease this complexity, by reducing the order of interactions that are taken into account. This approach finally gives a computationally, feasible problem. Experiments on various datasets show the effectiveness of the approach.<br>

## Configuration

Python 3.6

  numpy 1.14.3 <br>
  cvxopt 1.2.0 <br>
  scikit-learn 0.19.1 <br>
