package ifcat

import "math/rand"

// Tree is base structure for the iTree
type Tree struct {
	*Forest
	root *Node
}

// Node is a structure for iNode
type Node struct {
	Left  *Node
	Right *Node
	// SplitAtt and SplitValue is used in prediction stage.
	// Only inNodes have these two fields.
	SplitAtt   AttributeMeta
	SplitValue []float64
	// Only exNodes have `Size` field.
	// nodes with no children are called external nodes or leaf nodes.
	Size int
}

func (t *Tree) Build(X []Vector) {
	t.root = buildNode(X, 0, t.heightLimit, t.localSchema)
}

// buildNode builds an itree. Returns the root node.
//
//	X: input dataset
//	e: current tree height
//	l: height limit
func buildNode(X []Vector, e int, l int, ls localSchema) *Node {
	if e >= l || len(X) <= 1 {
		exNode := &Node{Size: len(X)}
		return exNode
	}
	q := randAtt(ls.IdxToName)
	Xl, Xr, splitValue := filter(X, q, ls)
	inNode := &Node{
		Left:       buildNode(Xl, e+1, l, ls),
		Right:      buildNode(Xr, e+1, l, ls),
		SplitAtt:   q,
		SplitValue: splitValue,
	}
	return inNode
}

// randAtt randomly select an attribute q from Q
func randAtt(m map[int]AttributeMeta) AttributeMeta {
	return m[rand.Intn(len(m))]
}

// filter filters the dataset X based on the conditional expression.
// If the data meets the condition, it would be put into Xl, else Xr.
//
// returns the sub-dataset for left tree, right tree and the SplitValue on current inNode.
func filter(X []Vector, q AttributeMeta, ls localSchema) ([]Vector, []Vector, []float64) {
	attIdx := ls.NameToIdx[q]
	attType := q.Type

	Xl := make([]Vector, 0, len(X))
	Xr := make([]Vector, 0, len(X))

	splitValue := []float64{}

	switch attType {
	case TypeCategorical:
		{
			counts := make(map[float64]struct{})
			for _, v := range X {
				val := v[attIdx]
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
			splitValue = subset

			// if qv is in subset p, then put in Xl, else Xr
			for _, data := range X {
				qv := data[attIdx]
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
			min := X[0][attIdx]
			max := X[0][attIdx]
			// determine max and min by iterating X
			for _, v := range X {
				val := v[attIdx]
				if val < min {
					min = val
				}
				if val > max {
					max = val
				}
			}
			// randomly select a split point in [min, max]
			p := min + rand.Float64()*(max-min)
			splitValue = append(splitValue, p)

			for _, data := range X {
				qv := data[attIdx]
				if qv < p {
					Xl = append(Xl, data)
				} else {
					Xr = append(Xr, data)
				}
			}
		}
	case TypeBool:
		for _, data := range X {
			qv := data[attIdx]
			if qv == 1 {
				Xl = append(Xl, data)
			} else {
				Xr = append(Xr, data)
			}
		}
	}

	return Xl, Xr, splitValue
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
