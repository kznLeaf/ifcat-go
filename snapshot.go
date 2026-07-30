package ifcat

import (
	"encoding/gob"
	"errors"
	"os"
)

// LocalSchemaSnapshot mirrors localSchema with exported fields.
type LocalSchemaSnapshot struct {
	NameToIdx Schema
	IdxToName map[int]AttributeMeta
}

// TreeSnapshot mirrors Tree without the circular *Forest pointer.
type TreeSnapshot struct {
	Root *Node
}

// ForestSnapshot mirrors Forest with all exported fields.
type ForestSnapshot struct {
	LocalSchema     LocalSchemaSnapshot
	Trees           []TreeSnapshot
	SubsamplingSize int
	TreeCount       int
	HeightLimit     int
	ScoreThreshold  float64
	Trained         bool
}

func forestToSnapshot(f *Forest) ForestSnapshot {
	trees := make([]TreeSnapshot, len(f.trees))
	for i, tree := range f.trees {
		trees[i] = TreeSnapshot{Root: tree.root}
	}

	return ForestSnapshot{
		LocalSchema: LocalSchemaSnapshot{
			NameToIdx: f.nameToIdx,
			IdxToName: f.idxToName,
		},
		Trees:           trees,
		SubsamplingSize: f.subsamplingSize,
		TreeCount:       f.treeCount,
		HeightLimit:     f.heightLimit,
		ScoreThreshold:  f.scoreThreshold,
		Trained:         f.trained,
	}
}

func snapshotToForest(s ForestSnapshot) *Forest {
	f := &Forest{
		localSchema: localSchema{
			nameToIdx: s.LocalSchema.NameToIdx,
			idxToName: s.LocalSchema.IdxToName,
		},
		subsamplingSize: s.SubsamplingSize,
		treeCount:       s.TreeCount,
		heightLimit:     s.HeightLimit,
		scoreThreshold:  s.ScoreThreshold,
		trained:         s.Trained,
	}

	f.trees = make([]Tree, len(s.Trees))
	for i, ts := range s.Trees {
		f.trees[i] = Tree{Forest: f, root: ts.Root}
	}

	return f
}

// SaveForestGob serializes a Forest instance into a gob file.
func SaveForestGob(path string, f *Forest) error {
	if f == nil {
		return errors.New("forest is nil")
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return gob.NewEncoder(file).Encode(forestToSnapshot(f))
}

// LoadForestGob deserializes a Forest instance from a gob file.
func LoadForestGob(path string) (*Forest, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var snap ForestSnapshot
	if err := gob.NewDecoder(file).Decode(&snap); err != nil {
		return nil, err
	}

	return snapshotToForest(snap), nil
}
