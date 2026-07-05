package ifcat

type AttributeType byte

const (
	TypeNumerical   AttributeType = iota // 0: numerical
	TypeCategorical                      // 1: categorical
	TypeBool                             // 2: Bool
)

var AttNameType = map[string]AttributeType{
	"rooted": TypeCategorical,
}

// Q is a list of attribute in X. It is only used in randAtt()
// TODO: init Q
// Q's Name and Type is useful. Value field is not used.
var Q []Attribute

type Attribute struct {
	Name string
	Type AttributeType
	// Value can represent both categorical and numerical values.
	Value float64
}

// Vector each input data is consisted of several features.
type Vector map[string]Attribute
