package ifcat_test

import (
	"bufio"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/kznLeaf/ifcat-go"
)

// -------------------------------------------------------------------

func TestForest_AnomalyScore_TwoNumericalVariables(t *testing.T) {
	forest := newTwoNumericalVariablesForest(t)

	x := ifcat.Vector{-2.0, -2.0}
	want := 0.3436

	got := forest.AnomalyScore(x)

	if math.Abs(got-want) > 0.01 {
		t.Errorf("AnomalyScore() = %v, want %v", got, want)
	}
}

func newTwoNumericalVariablesForest(t *testing.T) ifcat.Forest {
	t.Helper()

	const (
		nInliers         int     = 240
		nOutliers        int     = 40
		treeCount        int     = 100
		subsamplingSize  int     = 100
		anomalyThreshold float64 = 0.5
	)

	data := generateNumericalData(nInliers, nOutliers)

	forest := ifcat.Forest{}

	forest.AddField("X", ifcat.TypeNumerical)
	forest.AddField("Y", ifcat.TypeNumerical)

	forest.NewForest(treeCount, subsamplingSize, anomalyThreshold)
	forest.Train(data)

	return forest
}

func generateNumericalData(nInliers int, nOutliers int) []ifcat.Vector {
	totalSamples := nInliers + nOutliers

	r := rand.New(rand.NewPCG(0, 0))
	X := make([][]float64, 0, totalSamples)

	// (Spherical) 240 inliers
	// 0.3 * randn(240, 2) + [-2, -2]
	for range nInliers {
		z1 := r.NormFloat64()
		z2 := r.NormFloat64()

		x := 0.3*z1 - 2.0
		y := 0.3*z2 - 2.0

		X = append(X, []float64{x, y})
	}

	// 40 random outliers
	// uniform(low=-4, high=4, size=(40, 2))
	for range nOutliers {
		x := r.Float64()*8.0 - 4.0
		y := r.Float64()*8.0 - 4.0

		X = append(X, []float64{x, y})
	}

	vectors := make([]ifcat.Vector, len(X))
	for i, row := range X {
		vectors[i] = ifcat.Vector(row)
	}

	return vectors
}

// -------------------------------------------------------------------

func TestForest_AnomalyScore_CategoricalVariables(t *testing.T) {
	path := "./testdata/car_evaluation/car.data"
	// Train forest based on car_evaluation normalDataset
	// The normalDataset returned here is used for both training and later testing
	f, normalDataset, anomalyDataset := newCategoricalVariablesForest(t, path)

	anomalyScores := make([]float64, len(anomalyDataset))
	normalScores := make([]float64, len(normalDataset))

	for i, data := range anomalyDataset {
		anomalyScores[i] = f.AnomalyScore(data)
	}

	for i, data := range normalDataset {
		normalScores[i] = f.AnomalyScore(data)
	}

	// save scores in a csv file
	writeCSV(normalScores, anomalyScores)

	// Prediction stage: iterate on whole dataset, compute TP, FP and FN
	// var TP, FP, FN int
	//
	// for i := range normalDataset {
	// 	score := f.AnomalyScore(normalDataset[i])
	// 	t.Logf("score: %v\n", score)
	// 	predictIsVgood := f.Predict(normalDataset[i])
	// 	actualIsVgood := anomalyDataset[i]
	// 	if predictIsVgood && actualIsVgood {
	// 		TP += 1
	// 	}
	// 	if predictIsVgood && !actualIsVgood {
	// 		FP += 1
	// 	}
	// 	if !predictIsVgood && actualIsVgood {
	// 		FN += 1
	// 	}
	// }
	//
	// t.Logf("TP: %v, FP: %v, FN: %v\n", TP, FP, FN)
	//
	// precision := float64(TP) / (float64(TP) + float64(FP))
	// recall := float64(TP) / (float64(TP) + float64((FN)))
	// f1Score := 2 * precision * recall / (precision + recall)
	//
	// t.Logf("Precision: %v\n", precision)
	// t.Logf("Recall: %v\n", recall)
	// t.Logf("F1 Score: %v\n", f1Score)
}

func newCategoricalVariablesForest(t *testing.T, path string) (ifcat.Forest, []ifcat.Vector, []ifcat.Vector) {
	t.Helper()

	const (
		treeCount        int     = 100
		subsamplingSize  int     = 256
		anomalyThreshold float64 = 0.5
	)

	normalDataset, anomalyDataset := parseCarEvaluationData(path)
	// for i := range 20 {
	// 	t.Log(dataset[i])
	// }

	f := ifcat.Forest{}
	f.AddField("buying", ifcat.TypeCategorical)
	f.AddField("maint", ifcat.TypeCategorical)
	f.AddField("doors", ifcat.TypeCategorical)
	f.AddField("persons", ifcat.TypeCategorical)
	f.AddField("lug_boot", ifcat.TypeCategorical)
	f.AddField("safety", ifcat.TypeCategorical)

	f.NewForest(treeCount, subsamplingSize, anomalyThreshold)
	f.Train(normalDataset)

	return f, normalDataset, anomalyDataset
}

func parseCarEvaluationData(path string) ([]ifcat.Vector, []ifcat.Vector) {
	normalDataset := make([]ifcat.Vector, 0, 2000)
	anomalyDataset := make([]ifcat.Vector, 0, 100)

	buyingOrMaint := map[string]float64{
		"low":   0,
		"med":   1,
		"high":  2,
		"vhigh": 3,
	}
	lugBoot := map[string]float64{
		"small": 0,
		"med":   1,
		"big":   2,
	}
	safety := map[string]float64{
		"low":  0,
		"med":  1,
		"high": 2,
	}
	// read and parse data into []Vector
	err := forEachLine(path, func(line string) {
		atts := strings.Split(line, ",")
		class := atts[6]
		if class == "acc" || class == "good" {
			return
		}
		instance := make(ifcat.Vector, 6)
		for i := range 6 {
			att := strings.TrimSpace(atts[i])

			switch i {
			case 0, 1:
				instance[i] = mustLookup(buyingOrMaint, att)
			case 2, 3:
				// `more` and `5more` is marked as number `0` here
				if num, err := strconv.Atoi(att); err == nil {
					instance[i] = float64(num)
				} else {
					instance[i] = 0
				}
			case 4:
				instance[i] = mustLookup(lugBoot, att)
			case 5:
				instance[i] = mustLookup(safety, att)
			}
		}

		if class == "unacc" {
			normalDataset = append(normalDataset, instance)
		} else {
			anomalyDataset = append(anomalyDataset, instance)
		}
	})
	if err != nil {
		panic(err)
	}
	return normalDataset, anomalyDataset
}

func forEachLine(path string, callback func(line string)) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		callback(scanner.Text())
	}

	return scanner.Err()
}

func mustLookup(m map[string]float64, key string) float64 {
	v, ok := m[key]
	if !ok {
		panic("unknown category: " + key)
	}
	return v
}

// writeCSV save scores in a csv file so we can analyze it later using a python script
func writeCSV(nomalScores []float64, anomalyScores []float64) {
	file, err := os.Create("scores.csv")
	if err != nil {
		panic("can not create csv file")
	}
	defer file.Close()

	fmt.Fprintln(file, "score,label")

	for _, score := range anomalyScores {
		fmt.Fprintf(file, "%f,1\n", score)
	}

	for _, score := range nomalScores {
		fmt.Fprintf(file, "%f,0\n", score)
	}
}
