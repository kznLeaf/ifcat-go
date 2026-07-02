package ifcat

type FeatureType byte

const (
	TypeNumerical   FeatureType = iota // 0: numerical
	TypeCategorical                    // 1: categorical
	TypeText                           // 2: text
)

type Feature struct {
	Type FeatureType

	NumValue float64
	CatValue int
	StrValue string
}

// Vector input data
type Vector []Feature

func NewNumerical(v float64) Feature {
	return Feature{Type: TypeNumerical, NumValue: v}
}

func NewCategorical(id int) Feature {
	return Feature{Type: TypeCategorical, CatValue: id}
}

func NewText(v string) Feature {
	return Feature{Type: TypeText, StrValue: v}
}
