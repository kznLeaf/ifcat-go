package ifcat

import (
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
)

// Euler is an Euler's constant as described in algorithm specification
const Euler float64 = 0.5772156649

type Forest struct {
	localSchema

	trees           []Tree
	subsamplingSize int
	treeCount       int
	heightLimit     int
	scoreThreshold  float64
	// TODO: add a field `trained bool`
}

type localSchema struct {
	// nameToIdx maps each attribute metadata to its slice index
	nameToIdx map[AttributeMeta]int
	// idxToName provides an lookup to efficiently select
	// a random attribute during node splitting
	idxToName map[int]AttributeMeta
}

func (f *Forest) SetAnomalyThreshold(a float64) {
	f.scoreThreshold = a
}

// AddField appends one field to the local schema of the current forest.
func (f *Forest) AddField(name string, attrType AttributeType) {
	if f.nameToIdx == nil {
		f.nameToIdx = make(Schema)
	}

	if f.idxToName == nil {
		f.idxToName = make(map[int]AttributeMeta)
	}

	meta := AttributeMeta{
		Name: name,
		Type: attrType,
	}

	if _, exists := f.nameToIdx[meta]; exists {
		return
	}

	nextIndex := len(f.nameToIdx)
	f.nameToIdx[meta] = nextIndex
	f.idxToName[nextIndex] = meta
}

// RegisterFields registers multiple attributes into the forest configuration.
// Calling RegisterFields multiple times will overwrite any existing field configurations.
//
// The order in which fields are added must strictly match the column
// ordering of the input dataset Vectors, as the underlying model relies on
// fixed index mapping to determine the specific data type.
func (f *Forest) RegisterFields(fields []AttributeMeta) {
	if f.nameToIdx == nil {
		f.nameToIdx = make(Schema)
	}

	if f.idxToName == nil {
		f.idxToName = make(map[int]AttributeMeta)
	}

	for i, attr := range fields {
		if _, exists := f.nameToIdx[attr]; exists {
			continue
		}
		f.nameToIdx[attr] = i
		f.idxToName[i] = attr
	}
}

// Init initializes the isolation forest. Calling this method is a prerequisite for executing the Train method.
//
// The t parameter specifies the total number of isolation trees to be created
// within the forest.
//
// The subsamplingSize determines the number of samples drawn to train each base tree.
// A value of 256 is recommended for most datasets:
//
//	subsamplingSize := 256
//
// The threshold defines the decision threshold for the final anomaly score.
// If a calculated score is smaller than this value, the instance is likely
// to be classified as a normal data point.
func (f *Forest) Init(t int, subsamplingSize int, threshold float64) {
	// Initialize Forest
	heightLimit := math.Ceil(math.Log2(float64(subsamplingSize)))

	f.trees = make([]Tree, t)
	// Link each tree back to the forest so it can access the localSchema.
	for i := range t {
		f.trees[i] = Tree{Forest: f}
	}

	f.subsamplingSize = subsamplingSize
	f.treeCount = t
	f.heightLimit = int(heightLimit)
	f.scoreThreshold = threshold
}

// Train builds the Isolation Forest model using the provided trainSet.
// AddFields and Init must be called prior to invoking this function.
func (f *Forest) Train(trainSet []Vector) {
	n := len(trainSet)

	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	var wg sync.WaitGroup

	for i := range f.treeCount {
		wg.Add(1)
		sem <- struct{}{}

		go func(treeIndex int) {
			defer wg.Done()
			defer func() { <-sem }()

			indices := rand.Perm(n)
			samples := make([]Vector, f.subsamplingSize)
			for i := range f.subsamplingSize {
				samples[i] = trainSet[indices[i]]
			}
			f.trees[i].Build(samples)
		}(i)
	}
	wg.Wait()
}

// AnomalyScore computes the average path length of x from the ensemble of trees,
// then normalize it to a range between 0 and 1.
//
// If instances have anomaly score close to 1, then they are anomalies.
// If instances have anomaly score close to zero, then they are inliers.
func (f *Forest) AnomalyScore(x Vector) float64 {
	plSum := 0.0
	for _, tree := range f.trees {
		root := tree.root
		plSum += f.pathLength(x, root, 0)
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

// Predict predicts if instance x is an anomaly point.
func (f *Forest) Predict(x Vector) bool {
	return f.AnomalyScore(x) > f.scoreThreshold
}

// pathLength computes the path length on one tree
//
//	x: an instance
//	t: an iTree
//	e: current path length
func (f *Forest) pathLength(x Vector, t *Node, e float64) float64 {
	// external node
	if t.Left == nil && t.Right == nil {
		return e + averagePathLength(t.Size)
	}
	if t.Left == nil || t.Right == nil {
		panic("invalid tree: internal node has only one child")
	}
	// inNode
	att := t.SplitAtt
	// access localSchema from f
	attIdx, ok := f.nameToIdx[att]
	if !ok {
		panic("unknown split attribute")
	}
	xattv := x[attIdx]

	switch att.Type {
	case TypeCategorical:
		{
			if slices.Contains(t.SplitValue, xattv) {
				return f.pathLength(x, t.Left, e+1)
			} else {
				return f.pathLength(x, t.Right, e+1)
			}
		}
	case TypeNumerical:
		{
			if xattv < t.SplitValue[0] {
				return f.pathLength(x, t.Left, e+1)
			} else {
				return f.pathLength(x, t.Right, e+1)
			}
		}
	case TypeBool:
		{
			if xattv == 1 {
				return f.pathLength(x, t.Left, e+1)
			} else {
				return f.pathLength(x, t.Right, e+1)
			}
		}
	default:
		panic("unknown attribute type")
	}
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
