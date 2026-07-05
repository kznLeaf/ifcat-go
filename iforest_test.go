package ifcat_test

import (
	"ifcat-go"
	"math"
	"math/rand/v2"
	"testing"
)

var f *ifcat.Forest

const (
	nInliers        int     = 240
	nOutliers       int     = 40
	treeCount       int     = 100
	subsamplingSize int     = 100
	anomalyRatio    float64 = 0.5
)

func init() {
	ifcat.AddField("X", ifcat.TypeNumerical)
	ifcat.AddField("Y", ifcat.TypeNumerical)

	rawData := generateNumericalData()
	f = ifcat.NewForest(treeCount, subsamplingSize, anomalyRatio)
	f.Train(rawData)
}

func TestForest_AnomalyScore(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for receiver constructor.
		t               int
		subsamplingSize int
		// Named input parameters for target function.
		x    ifcat.Vector
		want float64
	}{
		{
			name:            "Two numerical varibles",
			t:               treeCount,
			subsamplingSize: subsamplingSize,
			x:               ifcat.Vector{-2.0, -2.0},
			want:            0.3436, // 0.3436 is the result computed by sklearn
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := f.AnomalyScore(tt.x)
			// For two-dimensional numerical data,
			// the error between our algorithm and sklearn is less than 0.01.
			if math.Abs(got-tt.want) > 0.01 {
				t.Errorf("AnomalyScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func generateNumericalData() []ifcat.Vector {
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
