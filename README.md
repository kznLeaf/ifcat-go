## ifcat-go

Go implementation of the Isolation Forest algorithm with support for Categorical data.

This is basically a golang implementation of the algorithm in [Extending Isolation Forest to support non-numerical data](https://github.com/SinaDBMS/IsolationForest) from Sina Barghidarian. Text features are are not supported for now.

## Installation

Go 1.22+

```sh
go get -u github.com/kznLeaf/ifcat-go
```

## Example

Take car_evaluation for example:

```go
	const (
		treeCount        int     = 100
		subsamplingSize  int     = 256
		anomalyThreshold float64 = 0.5
	)

	// Convert the dataset into a slice of []ifcat.Vector.
	// For example, [vhigh,vhigh,2,2,small,low,unacc] is encoded as [3 3 2 2 0 0].
	// Note: It is recommended to normalize numerical data to the [0, 1] range.
	normalDataset, anomalyDataset := parseCarEvaluationData("./car.data")

	// Define the schema for the forest. Each forest has its local schema.
	// Fields must be added in the exact same order as they appear in the dataset.
	// Currently supported data types:
    // 
	// - TypeNumerical:   Continuous numeric values.
	// - TypeCategorical: Discrete values from a finite set.
	// - TypeBool:        A specialized categorical type, applicable when values are restricted to 0 or 1.
    // 
    // Mixed types are also supported.
	f := ifcat.Forest{}
    f.RegisterFields([]ifcat.AttributeMeta{
		{"buying", ifcat.TypeCategorical},
		{"maint", ifcat.TypeCategorical},
		{"doors", ifcat.TypeCategorical},
		{"persons", ifcat.TypeCategorical},
		{"lug_boot", ifcat.TypeCategorical},
		{"safety", ifcat.TypeCategorical},
	})

	// Initialize the forest with the number of trees, subsampling size, and anomaly threshold.
	f.Init(treeCount, subsamplingSize, anomalyThreshold)

	// The anomaly threshold can be adjusted at any time.
	f.SetAnomalyThreshold(0.5)

	// Train the Isolation Forest model on the dataset.
	f.Train(normalDataset)

	// Calculate scores
	anomalyScores := make([]float64, len(anomalyDataset))
	normalScores := make([]float64, len(normalDataset))

   	if !f.Trained() {
		return
	}
	// Calculate anomaly scores for both evaluation and baseline datasets.
	for i, data := range anomalyDataset {
		anomalyScores[i], _ = f.AnomalyScore(data)
	}
	for i, data := range normalDataset {
		normalScores[i], _ = f.AnomalyScore(data)
	}

   	// Predict if the data is a anomaly point.
   	predict, err := f.Predict(anomalyDataset[0])
	if err != nil {
		panic(err)
    }
```

## Experimental results

In order to measure the performance of each algorithm, on a single dataset, we run it 10 times with different initial states and report the average of ROC-AUC.

### Car Evaluation

- Source: https://archive.ics.uci.edu/dataset/19/car+evaluation
- Instances: 1,275(class acc and class good are excluded)
  - class labels: unacc(normal, 70.023%), vgood(anomaly, 3.762%). Train the model on unacc, then evaluate it on the full dataset.
  - attributes: 0 Num, 6 Cat.
- Average ROC-AUC: 0.9994

### KDD Cup 1999 (10% subset)

- Source: https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data
- Instances: 494,020
  - class labels: normal(97,277), rest(396,743). Train the model on normal, then evaluate on the full dataset.
  - attributes: 33 Num, 7 Cat.
- Average ROC-AUC: 0.9516

### Mushroom

- Source: https://archive.ics.uci.edu/dataset/73/mushroom
- Instances: 8,124
  - class labels: e(4,208), p(3,916). Train the model on "e", then evaluate on the full dataset. The missing value in stalk-root is marked as "?".
  - attributes: 0 Num, 22 Cat.
- Average ROC-AUC: 0.9094
