package ifcat

import (
	"math"
	"math/rand"
)

// Euler is an Euler's constant as described in algorithm specification
const Euler float64 = 0.5772156649

type Forest struct {
	Trees           []Tree
	SubsamplingSize int
	TreeCount       int
	HeightLimit     int
	AnomalyRatio    float64
}

// NewForest returns a set of iTrees
//
//	t: number of trees
func NewForest(t int, subsamplingSize int) *Forest {
	// Initialize Forest
	heightLimit := math.Ceil(math.Log2(float64(subsamplingSize)))
	trees := make([]Tree, 0, t)

	f := &Forest{
		Trees:           trees,
		SubsamplingSize: subsamplingSize,
		TreeCount:       t,
		HeightLimit:     int(heightLimit),
		AnomalyRatio:    0.5,
	}
	return f
}

// Train creates the collection of trees in the forest.
// The total number of `trainSet` must be SubsamplingSize * TreeCount since
// `trainSet` represents the subsampled dataset instead of the whole dataset.
func (f *Forest) Train(trainSet []Vector) {
	n := len(trainSet)
	if n < f.SubsamplingSize*f.TreeCount {
		return
	}

	indices := rand.Perm(n)

	for i := 0; i < n; i += f.SubsamplingSize {
		end := min(i+f.SubsamplingSize, n)
		batchIndices := indices[i:end]
		batchData := make([]Vector, len(batchIndices))
		for j, idx := range batchIndices {
			batchData[j] = trainSet[idx]
		}
		newTree := NewTree(batchData, f.HeightLimit)
		f.Trees = append(f.Trees, *newTree)
	}
}

// AnomalyScore computes the average path length of x from the ensemble of trees,
// then normalize it in (0, 1).
//   - if instances have anomaly score close to 1, then they are anomalies.
//   - if instance have anomaly score close to 0, then they are inliers.
func (f *Forest) AnomalyScore(x Vector) float64 {
	plSum := 0.0
	for _, tree := range f.Trees {
		root := tree.Root
		plSum += pathLength(x, root, 0)
	}
	avg := plSum / float64(len(f.Trees))
	exponent := -1 * avg / averagePathLength(f.SubsamplingSize)
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
	xattv := x[att.Name].Value

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

// averagePathLength is the same as c(n)
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
