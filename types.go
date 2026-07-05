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

// globalSchema maps each attribute metadata to its corresponding slice index.
var globalSchema Schema = map[AttributeMeta]int{}

// globalSchemaIdxToName provides an lookup to efficiently select
// a random attribute during node splitting
var globalSchemaIdxToName = map[int]AttributeMeta{}

// AddField adds one field to globalSchema. This method is not thread safe.
func AddField(name string, attrType AttributeType) {
	meta := AttributeMeta{
		Name: name,
		Type: attrType,
	}

	if _, exists := globalSchema[meta]; exists {
		return
	}

	nextIndex := len(globalSchema)
	globalSchema[meta] = nextIndex
	globalSchemaIdxToName[nextIndex] = meta
}
