package ifcat

import (
	"math"
	"math/rand/v2"
)

// Euler is an Euler's constant as described in algorithm specification
const Euler float64 = 0.5772156649

type Forest struct {
	trees           []Tree
	subsamplingSize int
	treeCount       int
	heightLimit     int
	anomalyRatio    float64
}

func (f *Forest) SetAnomalyThreshold(a float64) {
	f.anomalyRatio = a
}

// NewForest initializes and returns an empty Forest ready for training.
//
// The t parameter specifies the total number of isolation trees to be created
// within the forest.
//
// The subsamplingSize determines the number of samples drawn to train each base tree.
// A value of 256 is recommended for most datasets:
//
//	subsamplingSize := 256
//
// The anomalyRatio defines the decision threshold for the final anomaly score.
// If a calculated score is smaller than this ratio, the instance is likely
// to be classified as a normal data point.
func NewForest(t int, subsamplingSize int, anomalyRatio float64) *Forest {
	// Initialize Forest
	heightLimit := math.Ceil(math.Log2(float64(subsamplingSize)))
	trees := make([]Tree, t)

	f := &Forest{
		trees:           trees,
		subsamplingSize: subsamplingSize,
		treeCount:       t,
		heightLimit:     int(heightLimit),
		anomalyRatio:    anomalyRatio,
	}
	return f
}

// Train creates the collection of trees in the forest.
func (f *Forest) Train(trainSet []Vector) {
	n := len(trainSet)

	for i := range f.treeCount {
		indices := rand.Perm(n)
		samples := make([]Vector, f.subsamplingSize)
		for i := range f.subsamplingSize {
			samples[i] = trainSet[indices[i]]
		}
		f.trees[i] = *NewTree(samples, f.heightLimit)
	}
}

// AnomalyScore computes the average path length of x from the ensemble of trees,
// then normalize it to a range between 0 and 1.
//
//   - if instances have anomaly score close to 1, then they are anomalies.
//   - if instances have anomaly score close to 0, then they are inliers.
func (f *Forest) AnomalyScore(x Vector) float64 {
	plSum := 0.0
	for _, tree := range f.trees {
		// TODO: accelerate using goroutines
		root := tree.Root
		plSum += pathLength(x, root, 0)
	}
	avg := plSum / float64(len(f.trees))
	exponent := -1 * avg / averagePathLength(f.subsamplingSize)
	s := math.Pow(2, exponent)
	// TODO: remove panic
	if s < 0 || s > 1 {
		panic("invalid anomaly score")
	}
	return s
}

// pathLength computes the path length on one tree
//
//	x: an instance
//	t: an iTree
//	e: current path length
func pathLength(x Vector, t *Node, e float64) float64 {
	// external node
	if t.Left == nil && t.Right == nil {
		return e + averagePathLength(t.Size)
	}
	// inNode
	att := t.SplitAtt
	attIdx := globalSchema[att]
	xattv := x[attIdx]

	switch att.Type {
	case TypeCategorical:
		{
			for _, v := range t.SplitValue {
				if xattv == v {
					return pathLength(x, t.Left, e+1)
				} else {
					return pathLength(x, t.Right, e+1)
				}
			}
		}
	case TypeNumerical:
		{
			if xattv < t.SplitValue[0] {
				return pathLength(x, t.Left, e+1)
			} else {
				return pathLength(x, t.Right, e+1)
			}
		}
	case TypeBool:
		{
			if xattv == 1 {
				return pathLength(x, t.Left, e+1)
			} else {
				return pathLength(x, t.Right, e+1)
			}
		}
	}
	// TODO: remove panic
	panic("Unreachable!")
}

// averagePathLength is the same as c(n) in algorithm description
func averagePathLength(n int) float64 {
	if n > 2 {
		return 2*harmonic(n-1) - 2*float64(n-1)/float64(n)
	}
	if n == 2 {
		return 1
	}
	return 0
}

func harmonic(i int) float64 {
	return math.Log(float64(i)) + Euler
}
