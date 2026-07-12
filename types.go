package ifcat

import "errors"

type AttributeType byte

const (
	TypeNumerical   AttributeType = iota // 0: numerical
	TypeCategorical                      // 1: categorical
	TypeBool                             // 2: Bool
)

// Vector represents one sample
type Vector []float64

type AttributeMeta struct {
	Name string
	Type AttributeType
}

// Schema defines the attributes for the data
type Schema map[AttributeMeta]int

var ErrModelNotTrained = errors.New("model not trained")
