package ifcat

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

// GlobalSchema defines the field types and index positions for each data.
var GlobalSchema Schema = map[AttributeMeta]int{
	{Name: "X", Type: TypeNumerical}: 0,
	{Name: "Y", Type: TypeNumerical}: 1,
}
var GlobalSchemaIdxToName map[int]AttributeMeta

// init initialize
func init() {
	GlobalSchemaIdxToName = make(map[int]AttributeMeta)
	for att, idx := range GlobalSchema {
		GlobalSchemaIdxToName[idx] = att
	}
}

// type Dataset struct {
// 	Schema  Schema
// 	Vectors []Vector
// }
