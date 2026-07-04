package ifcat

type AttributeType byte

const (
	TypeNumerical   AttributeType = iota // 0: numerical
	TypeCategorical                      // 1: categorical
	TypeBool                             // 2: Bool
)

type Operator byte

const (
	Less    Operator = 0
	Equal   Operator = 1
	Greater Operator = 2
)

var AttNameType = map[string]AttributeType{
	"rooted": TypeCategorical,
}

// Q is a list of attribute in X
// TODO: init Q
var Q []Attribute

type Attribute struct {
	Name string
	Type AttributeType
	// Value can represent both categorical and numerical values.
	Value float64
}

// Vector each input data is consisted of several features.
type Vector map[string]Attribute
