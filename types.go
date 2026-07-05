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

// globalSchema defines the field types and index positions for each data.
var globalSchema Schema = map[AttributeMeta]int{}

// globalSchemaIdxToName is used in randomly selecting an attribute q from all attributes
var globalSchemaIdxToName map[int]AttributeMeta

// init initializes GlobalSchemaIdxToName
func init() {
	globalSchemaIdxToName = make(map[int]AttributeMeta)
	for att, idx := range globalSchema {
		globalSchemaIdxToName[idx] = att
	}
}

// AddField adds one field to globalSchema. This method is not thread safe.
func AddField(name string, attrType AttributeType) {
	meta := AttributeMeta{
		Name: name,
		Type: attrType,
	}

	if _, exists := globalSchema[meta]; !exists {
		return
	}

	nextIndex := len(globalSchema)
	globalSchema[meta] = nextIndex
}
