package ifcat_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/kznLeaf/ifcat-go"
)

func TestForestGobRoundTrip(t *testing.T) {
	path := "./testdata/car_evaluation/car.data"
	original, normalDataset, anomalyDataset := newCategoricalVariablesForest(path)

	if !original.Trained() {
		t.Fatal("expected trained forest")
	}

	// create gob file in a temp dir which is automatically removed when the test completes.
	gobPath := filepath.Join(t.TempDir(), "forest.gob")
	if err := ifcat.SaveForestGob(gobPath, &original); err != nil {
		t.Fatalf("SaveForestGob: %v", err)
	}

	loaded, err := ifcat.LoadForestGob(gobPath)
	if err != nil {
		t.Fatalf("LoadForestGob: %v", err)
	}

	if !loaded.Trained() {
		t.Fatal("loaded forest is not trained")
	}

	testVectors := []ifcat.Vector{
		normalDataset[0],
		anomalyDataset[0],
	}

	for i, vector := range testVectors {
		origScore, err := original.AnomalyScore(vector)
		if err != nil {
			t.Fatalf("original AnomalyScore[%d]: %v", i, err)
		}

		loadedScore, err := loaded.AnomalyScore(vector)
		if err != nil {
			t.Fatalf("loaded AnomalyScore[%d]: %v", i, err)
		}

		if math.Abs(origScore-loadedScore) > 1e-12 {
			t.Fatalf("AnomalyScore mismatch at %d: got %v, want %v", i, loadedScore, origScore)
		}

		origPredict, err := original.Predict(vector)
		if err != nil {
			t.Fatalf("original Predict[%d]: %v", i, err)
		}

		loadedPredict, err := loaded.Predict(vector)
		if err != nil {
			t.Fatalf("loaded Predict[%d]: %v", i, err)
		}

		if origPredict != loadedPredict {
			t.Fatalf("Predict mismatch at %d: got %v, want %v", i, loadedPredict, origPredict)
		}
	}
}

func TestSaveForestGobNilForest(t *testing.T) {
	err := ifcat.SaveForestGob(filepath.Join(t.TempDir(), "forest.gob"), nil)
	if err == nil {
		t.Fatal("expected error for nil forest")
	}
}

func TestLoadForestGobMissingFile(t *testing.T) {
	_, err := ifcat.LoadForestGob(filepath.Join(t.TempDir(), "missing.gob"))
	if err == nil {
		t.Fatal("expected error for missing file")
	}
	if !os.IsNotExist(err) {
		t.Fatalf("expected file not exist error, got: %v", err)
	}
}
