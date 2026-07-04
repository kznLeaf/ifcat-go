package ifcat

import "math/rand"

// MaxDepth limits the depth of trees, is computed during initialization based
// on the subsampling size
var MaxDepth int

// Tree is base structure for the iTree
type Tree struct {
	Root *Node
}

// Node is a structure for iNode
type Node struct {
	// nodes with no children are called external nodes or leaf nodes
	Left  *Node
	Right *Node
	// SplitAtt is used in prediction stage.
	// Note SplitValue has been included in SplitAtt.
	SplitAtt   Attribute
	SplitValue []float64
	Size       int
	// TODO: Should I add `isExternal` attribute? it may be faster than `left == nil && right == nil`
}

// BuildTree builds a itree. Return the root node.
//
//	X: input dataset
//	e: current tree height
//	l: height limit
func (t *Tree) BuildTree(X []Vector, e int, l int) *Node {
	if e >= l || len(X) <= 1 {
		exNode := &Node{Size: len(X)}
		return exNode
	}
	q := randAtt()
	Xl, Xr := filter(X, q)
	inNode := &Node{
		Left:     t.BuildTree(Xl, e+1, l),
		Right:    t.BuildTree(Xr, e+1, l),
		SplitAtt: q,
	}
	return inNode
}

// randAtt randomly select an attribute q from Q
func randAtt() Attribute {
	return Q[rand.Intn(len(Q))]
}

// filter filters the dataset X based on the conditional expression.
// If the data meets the condition, it would be put into Xl, or else Xr.
func filter(X []Vector, q Attribute) ([]Vector, []Vector) {
	attName := q.Name
	attType := q.Type

	Xl := make([]Vector, 0, len(X))
	Xr := make([]Vector, 0, len(X))

	switch attType {
	case TypeCategorical:
		{
			counts := make(map[float64]struct{})
			for _, v := range X {
				val := v[attName].Value
				if _, ok := counts[val]; !ok {
					counts[val] = struct{}{}
				}
			}
			var size int
			// the size of subset is in [1, n - 1]
			if len(counts) == 0 {
				size = 1
			} else {
				size = rand.Intn(len(counts)-1) + 1
			}

			subset := randSubset(counts, size)

			// if qv is in subset p, then put into Xl, else Xr
			for _, data := range X {
				qv := data[attName].Value
				for _, v := range subset {
					if qv == v {
						Xl = append(Xl, data)
					} else {
						Xr = append(Xr, data)
					}
				}
			}
		}
	case TypeNumerical:
		{
			min := X[0][attName].Value
			max := X[0][attName].Value
			// determine max and min by iterating X
			for _, v := range X {
				val := v[attName].Value
				if val < min {
					min = val
				}
				if val > max {
					max = val
				}
			}
			// randomly select a split point in [min, max]
			p := min + rand.Float64()*(max-min)

			for _, data := range X {
				qv := data[attName].Value
				if qv < p {
					Xl = append(Xl, data)
				} else {
					Xr = append(Xr, data)
				}
			}
		}
	case TypeBool:
		for _, data := range X {
			qv := data[attName].Value
			if qv == 1 {
				Xl = append(Xl, data)
			} else {
				Xr = append(Xr, data)
			}
		}
	}

	return Xl, Xr
}

// randSubset select a subset using Fisher-Yates shuffle
func randSubset(fullSet map[float64]struct{}, size int) []float64 {
	keys := make([]float64, 0, len(fullSet))
	for k := range fullSet {
		keys = append(keys, k)
	}
	for i := range size {
		j := rand.Intn(len(keys)-i) + 1
		keys[i], keys[j] = keys[j], keys[i]
	}
	return keys[:size]
}
