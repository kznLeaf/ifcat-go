package ifcat_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/kznLeaf/ifcat-go"
)

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
		nInliers        int     = 240
		nOutliers       int     = 40
		treeCount       int     = 100
		subsamplingSize int     = 100
		anomalyRatio    float64 = 0.5
	)

	data := generateNumericalData(nInliers, nOutliers)

	forest := ifcat.Forest{}

	forest.AddField("X", ifcat.TypeNumerical)
	forest.AddField("Y", ifcat.TypeNumerical)

	forest.NewForest(treeCount, subsamplingSize, anomalyRatio)
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
