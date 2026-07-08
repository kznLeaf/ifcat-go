## ifcat-go

Go implementation of the Isolation Forest algorithm with support for Categorical data.

This is basically a golang implementation of the algorithm in [Extending Isolation Forest to support non-numerical data](https://github.com/SinaDBMS/IsolationForest) from Sina Barghidarian. Text features are are not supported for now.

## Experimental results

In order to measure the performance of each algorithm, on a single dataset, we run it 10 times with different initial states and report the average of AUC.

### Car Evaluation

- Source: https://archive.ics.uci.edu/dataset/19/car+evaluation
- Instances: 1275(class acc and class good are excluded)
  - class labels: unacc(normal), vgood(anomaly). Train the model on unacc, then evaluate it on the full dataset.
  - attributes: 0 Num, 6 Cat.
- Average AUC: 0.999452

### KDD Cup 1999 (10% subset)

- Source: https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data
- Instances: 494020
  - class labels: normal(97277), rest(396743). Train the model on normal, then evaluate on the full dataset.
  - attributes: 33 Num, 7 Cat.
- Average AUC: 0.951648
