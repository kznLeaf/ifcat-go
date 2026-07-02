package ifcat

// MaxDepth limits the depth of trees, is computed during initialization based
// on the subsampling size
var MaxDepth int

// Tree is base structure for the iTree
type Tree struct {
	Root *Node
}

// Node is a structure for iNode
type Node struct {
	Left      *Node
	Right     *Node
	Split     float64
	Attribute int
	C         float64
	Size      int
	External  bool
}
