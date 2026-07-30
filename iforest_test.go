package ifcat_test

import (
	"archive/zip"
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/kznLeaf/ifcat-go"
)

// ---------------------- Car Evaluation --------------------------

func BenchmarkCarScore(b *testing.B) {
	path := "./testdata/car_evaluation/car.data"

	f, _, anomalyDataset := newCategoricalVariablesForest(path)

	b.ResetTimer()
	for range b.N {
		f.AnomalyScore(anomalyDataset[0])
	}
}

func BenchmarkCarTrain(b *testing.B) {
	const (
		treeCount        int     = 100
		subsamplingSize  int     = 256
		anomalyThreshold float64 = 0.5
	)

	normalDataset, _ := parseCarEvaluationData("./testdata/car_evaluation/car.data")

	f := ifcat.Forest{}

	f.RegisterFields([]ifcat.AttributeMeta{
		{"buying", ifcat.TypeCategorical},
		{"maint", ifcat.TypeCategorical},
		{"doors", ifcat.TypeCategorical},
		{"persons", ifcat.TypeCategorical},
		{"lug_boot", ifcat.TypeCategorical},
		{"safety", ifcat.TypeCategorical},
	})

	f.Init(treeCount, subsamplingSize, anomalyThreshold)

	b.ResetTimer()
	for range b.N {
		f.Train(normalDataset)
	}
}

func TestForest_AnomalyScore_CategoricalVariables(t *testing.T) {
	path := "./testdata/car_evaluation/car.data"
	// Train forest based on car_evaluation normalDataset
	f, normalDataset, anomalyDataset := newCategoricalVariablesForest(path)

	anomalyScores := make([]float64, len(anomalyDataset))
	normalScores := make([]float64, len(normalDataset))

	for i, data := range anomalyDataset {
		anomalyScores[i], _ = f.AnomalyScore(data)
	}

	for i, data := range normalDataset {
		normalScores[i], _ = f.AnomalyScore(data)
	}

	// save scores in a csv file
	writeCSV(normalScores, anomalyScores)
}

func newCategoricalVariablesForest(path string) (ifcat.Forest, []ifcat.Vector, []ifcat.Vector) {

	const (
		treeCount        int     = 100
		subsamplingSize  int     = 256
		anomalyThreshold float64 = 0.5
	)

	normalDataset, anomalyDataset := parseCarEvaluationData(path)

	f := ifcat.Forest{}
	f.AddField("buying", ifcat.TypeCategorical)
	f.AddField("maint", ifcat.TypeCategorical)
	f.AddField("doors", ifcat.TypeCategorical)
	f.AddField("persons", ifcat.TypeCategorical)
	f.AddField("lug_boot", ifcat.TypeCategorical)
	f.AddField("safety", ifcat.TypeCategorical)

	f.Init(treeCount, subsamplingSize, anomalyThreshold)
	f.Train(normalDataset)

	return f, normalDataset, anomalyDataset
}

func parseCarEvaluationData(path string) ([]ifcat.Vector, []ifcat.Vector) {
	normalDataset := make([]ifcat.Vector, 0, 2000)
	anomalyDataset := make([]ifcat.Vector, 0, 100)

	buyingOrMaint := map[string]float64{
		"low":   0,
		"med":   1,
		"high":  2,
		"vhigh": 3,
	}
	lugBoot := map[string]float64{
		"small": 0,
		"med":   1,
		"big":   2,
	}
	safety := map[string]float64{
		"low":  0,
		"med":  1,
		"high": 2,
	}
	// read and parse data into []Vector
	err := forEachLine(path, func(line string) {
		atts := strings.Split(line, ",")
		class := atts[6]
		if class == "acc" || class == "good" {
			return
		}
		instance := make(ifcat.Vector, 6)
		for i := range 6 {
			att := strings.TrimSpace(atts[i])

			switch i {
			case 0, 1:
				instance[i] = mustLookup(buyingOrMaint, att)
			case 2, 3:
				// `more` and `5more` is marked as number `0` here
				if num, err := strconv.Atoi(att); err == nil {
					instance[i] = float64(num)
				} else {
					instance[i] = 0
				}
			case 4:
				instance[i] = mustLookup(lugBoot, att)
			case 5:
				instance[i] = mustLookup(safety, att)
			}
		}

		if class == "unacc" {
			normalDataset = append(normalDataset, instance)
		} else {
			anomalyDataset = append(anomalyDataset, instance)
		}
	})
	if err != nil {
		panic(err)
	}
	return normalDataset, anomalyDataset
}

// --------------------------- KDD 1999 -------------------------------------

const (
	kdd99ZipPath = "testdata/kdd99/kddcup.data_10_percent.zip"
	kdd99TxtPath = "testdata/kdd99/kddcup.data_10_percent.txt"
	kdd99TxtName = "kddcup.data_10_percent.txt"
)

var (
	kdd99Once      sync.Once
	kdd99EnsureErr error
)

func ensureKDD99Data(t *testing.T) string {
	t.Helper()

	kdd99Once.Do(func() {
		if _, err := os.Stat(kdd99TxtPath); err == nil {
			return
		}
		kdd99EnsureErr = extractKDD99Zip()
	})
	if kdd99EnsureErr != nil {
		t.Fatalf("prepare kdd99 testdata: %v", kdd99EnsureErr)
	}
	return kdd99TxtPath
}

func extractKDD99Zip() error {
	r, err := zip.OpenReader(kdd99ZipPath)
	if err != nil {
		return fmt.Errorf("open zip %q: %w", kdd99ZipPath, err)
	}
	defer r.Close()

	if len(r.File) != 1 {
		return fmt.Errorf("expected 1 file in %q, got %d", kdd99ZipPath, len(r.File))
	}

	entry := r.File[0]
	if entry.Name != kdd99TxtName {
		return fmt.Errorf("expected %q in zip, got %q", kdd99TxtName, entry.Name)
	}

	if err := os.MkdirAll(filepath.Dir(kdd99TxtPath), 0o755); err != nil {
		return fmt.Errorf("create directory %q: %w", filepath.Dir(kdd99TxtPath), err)
	}

	rc, err := entry.Open()
	if err != nil {
		return fmt.Errorf("open zip entry %q: %w", entry.Name, err)
	}
	defer rc.Close()

	out, err := os.OpenFile(kdd99TxtPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("create %q: %w", kdd99TxtPath, err)
	}
	defer out.Close()

	written, err := io.Copy(out, rc)
	if err != nil {
		os.Remove(kdd99TxtPath)
		return fmt.Errorf("extract %q: %w", kdd99TxtPath, err)
	}

	if uint64(written) != entry.UncompressedSize64 {
		os.Remove(kdd99TxtPath)
		return fmt.Errorf("extracted %q size mismatch: got %d bytes, want %d", kdd99TxtPath, written, entry.UncompressedSize64)
	}

	return nil
}

func TestKdd99_10(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping KDD99 integration test in short mode")
	}

	path := ensureKDD99Data(t)
	// Train forest based on full dataset
	f, normalDataset, anomalyDataset := newKDDCupForest(t, path)

	anomalyScores := make([]float64, len(anomalyDataset))
	normalScores := make([]float64, len(normalDataset))

	t.Logf("anomaly samples: %v, normal samples: %v", len(anomalyDataset), len(normalDataset))

	for i, data := range anomalyDataset {
		anomalyScores[i], _ = f.AnomalyScore(data)
	}

	for i, data := range normalDataset {
		normalScores[i], _ = f.AnomalyScore(data)
	}

	// save scores in a csv file
	writeCSV(normalScores, anomalyScores)
}

func newKDDCupForest(t *testing.T, path string) (ifcat.Forest, []ifcat.Vector, []ifcat.Vector) {
	t.Helper()

	const (
		treeCount        int     = 100
		subsamplingSize  int     = 256
		anomalyThreshold float64 = 0.5
	)

	normalDataset, anomalyDataset := parseKDDCupData(path)

	// fullDataset := make([]ifcat.Vector, 0, len(normalDataset)+len(anomalyDataset))
	// fullDataset = append(fullDataset, normalDataset...)
	// fullDataset = append(fullDataset, anomalyDataset...)

	f := ifcat.Forest{}

	f.RegisterFields([]ifcat.AttributeMeta{
		{"duration", ifcat.TypeNumerical},
		{"protocol_type", ifcat.TypeCategorical},
		{"service", ifcat.TypeCategorical},
		{"flag", ifcat.TypeCategorical},
		{"src_bytes", ifcat.TypeNumerical},
		{"dst_bytes", ifcat.TypeNumerical},
		{"land", ifcat.TypeBool},
		{"wrong_fragment", ifcat.TypeNumerical},
		{"urgent", ifcat.TypeNumerical},
		{"hot", ifcat.TypeNumerical},
		{"num_failed_logins", ifcat.TypeNumerical},
		{"logged_in", ifcat.TypeBool},
		{"num_compromised", ifcat.TypeNumerical},
		{"root_shell", ifcat.TypeNumerical},
		{"su_attempted", ifcat.TypeNumerical},
		{"num_root", ifcat.TypeNumerical},
		{"num_file_creations", ifcat.TypeNumerical},
		{"num_shells", ifcat.TypeNumerical},
		{"num_access_files", ifcat.TypeNumerical},
		{"num_outbound_cmds", ifcat.TypeNumerical},
		{"is_host_login", ifcat.TypeBool},
		{"is_guest_login", ifcat.TypeBool},
		{"count", ifcat.TypeNumerical},
		{"srv_count", ifcat.TypeNumerical},
		{"serror_rate", ifcat.TypeNumerical},
		{"srv_serror_rate", ifcat.TypeNumerical},
		{"rerror_rate", ifcat.TypeNumerical},
		{"srv_rerror_rate", ifcat.TypeNumerical},
		{"same_srv_rate", ifcat.TypeNumerical},
		{"diff_srv_rate", ifcat.TypeNumerical},
		{"srv_diff_host_rate", ifcat.TypeNumerical},
		{"dst_host_count", ifcat.TypeNumerical},
		{"dst_host_srv_count", ifcat.TypeNumerical},
		{"dst_host_same_srv_rate", ifcat.TypeNumerical},
		{"dst_host_diff_srv_rate", ifcat.TypeNumerical},
		{"dst_host_same_src_port_rate", ifcat.TypeNumerical},
		{"dst_host_srv_diff_host_rate", ifcat.TypeNumerical},
		{"dst_host_serror_rate", ifcat.TypeNumerical},
		{"dst_host_srv_serror_rate", ifcat.TypeNumerical},
		{"dst_host_rerror_rate", ifcat.TypeNumerical},
		{"dst_host_srv_rerror_rate", ifcat.TypeNumerical},
	})

	f.Init(treeCount, subsamplingSize, anomalyThreshold)
	f.Train(normalDataset)

	return f, normalDataset, anomalyDataset
}

// parseKDDCupData reads KDDCup99 records, normalizes continuous features to [0,1],
// encodes symbolic features as categorical values. Only "normal" label is treated
// as nomral.
//
// Returns normal dateset and anomaly dataset.
func parseKDDCupData(path string) ([]ifcat.Vector, []ifcat.Vector) {
	const featureCount = 41

	normalDataset := make([]ifcat.Vector, 0, 10000)
	anomalyDataset := make([]ifcat.Vector, 0, 10000)

	// KDDCup99 symbolic fields
	symbolicIdx := map[int]struct{}{
		1:  {}, // protocol_type
		2:  {}, // service
		3:  {}, // flag
		6:  {}, // land
		11: {}, // logged_in
		20: {}, // is_host_login
		21: {}, // is_guest_login
	}

	isSymbolic := func(idx int) bool {
		_, ok := symbolicIdx[idx]
		return ok
	}

	type rawRow struct {
		values []string
		label  string
	}

	rows := make([]rawRow, 0, 10000)

	minVals := make([]float64, featureCount)
	maxVals := make([]float64, featureCount)
	initialized := make([]bool, featureCount)

	// each categorical attribute holds its own map
	categoricalEncoders := make(map[int]map[string]float64)

	// ---------- numerical min/max ----------
	err := forEachLine(path, func(line string) {
		line = strings.TrimSpace(line)
		if line == "" {
			return
		}

		atts := strings.Split(line, ",")
		if len(atts) != featureCount+1 {
			panic(fmt.Sprintf("invalid KDD row: got %d columns, want %d, line=%q",
				len(atts), featureCount+1, line))
		}

		for i := range atts {
			atts[i] = strings.TrimSpace(atts[i])
		}

		label := strings.TrimSuffix(atts[featureCount], ".")
		rowValues := atts[:featureCount]

		rows = append(rows, rawRow{
			values: rowValues,
			label:  label,
		})

		for i := range featureCount {
			if isSymbolic(i) {
				continue
			}

			v, err := strconv.ParseFloat(rowValues[i], 64)
			if err != nil {
				panic(fmt.Sprintf("invalid numerical value at col %d: %q, line=%q",
					i, rowValues[i], line))
			}

			if !initialized[i] {
				minVals[i] = v
				maxVals[i] = v
				initialized[i] = true
				continue
			}

			if v < minVals[i] {
				minVals[i] = v
			}
			if v > maxVals[i] {
				maxVals[i] = v
			}
		}
	})
	if err != nil {
		panic(err)
	}

	// ---------- normalize continous data; encode categorical data ----------
	for _, row := range rows {
		instance := make(ifcat.Vector, featureCount)

		for i := range featureCount {
			raw := row.values[i]

			if isSymbolic(i) {
				encoder := categoricalEncoders[i]
				if encoder == nil {
					encoder = make(map[string]float64)
					categoricalEncoders[i] = encoder
				}

				code, ok := encoder[raw]
				if !ok {
					code = float64(len(encoder))
					encoder[raw] = code
				}

				instance[i] = code
				continue
			}

			v, err := strconv.ParseFloat(raw, 64)
			if err != nil {
				panic(fmt.Sprintf("invalid numerical value at col %d: %q", i, raw))
			}

			minVal := minVals[i]
			maxVal := maxVals[i]

			// normalize to [0, 1]
			if maxVal == minVal {
				instance[i] = 0
			} else {
				instance[i] = (v - minVal) / (maxVal - minVal)
			}
		}

		if row.label == "normal" {
			normalDataset = append(normalDataset, instance)
		} else {
			anomalyDataset = append(anomalyDataset, instance)
		}
	}

	return normalDataset, anomalyDataset
}

// -------------------------- Mushroom -------------------------------------

func TestMushroom(t *testing.T) {
	path := "./testdata/mushroom/agaricus-lepiota.data"
	// Train forest based on car_evaluation normalDataset
	f, normalDataset, anomalyDataset := newMushroomForest(t, path)

	anomalyScores := make([]float64, len(anomalyDataset))
	normalScores := make([]float64, len(normalDataset))

	for i, data := range anomalyDataset {
		anomalyScores[i], _ = f.AnomalyScore(data)
	}

	for i, data := range normalDataset {
		normalScores[i], _ = f.AnomalyScore(data)
	}

	// save scores in a csv file
	writeCSV(normalScores, anomalyScores)
}

func newMushroomForest(t *testing.T, path string) (ifcat.Forest, []ifcat.Vector, []ifcat.Vector) {
	t.Helper()

	const (
		treeCount        int     = 100
		subsamplingSize  int     = 256
		anomalyThreshold float64 = 0.5
	)

	normalDataset, anomalyDataset := parseMushroomData(path)

	f := ifcat.Forest{}

	f.RegisterFields([]ifcat.AttributeMeta{
		{"cap_shape", ifcat.TypeCategorical},
		{"cap_surface", ifcat.TypeCategorical},
		{"cap_color", ifcat.TypeCategorical},
		{"bruises", ifcat.TypeCategorical},
		{"odor", ifcat.TypeCategorical},
		{"gill_attachment", ifcat.TypeCategorical},
		{"gill_spacing", ifcat.TypeCategorical},
		{"gill_size", ifcat.TypeCategorical},
		{"gill_color", ifcat.TypeCategorical},
		{"stalk_shape", ifcat.TypeCategorical},
		{"stalk_root", ifcat.TypeCategorical},
		{"stalk_surface_above_ring", ifcat.TypeCategorical},
		{"stalk_surface_below_ring", ifcat.TypeCategorical},
		{"stalk_color_above_ring", ifcat.TypeCategorical},
		{"stalk_color_below_ring", ifcat.TypeCategorical},
		{"veil_type", ifcat.TypeCategorical},
		{"veil_color", ifcat.TypeCategorical},
		{"ring_number", ifcat.TypeCategorical},
		{"ring_type", ifcat.TypeCategorical},
		{"spore_print_color", ifcat.TypeCategorical},
		{"population", ifcat.TypeCategorical},
		{"habitat", ifcat.TypeCategorical},
	})

	f.Init(treeCount, subsamplingSize, anomalyThreshold)
	f.Train(normalDataset)

	return f, normalDataset, anomalyDataset
}

func parseMushroomData(path string) ([]ifcat.Vector, []ifcat.Vector) {
	// total 8124 samples
	normalDataset := make([]ifcat.Vector, 0, 4500)
	anomalyDataset := make([]ifcat.Vector, 0, 4000)

	featureMaps := []map[string]float64{
		0:  {"b": 0, "c": 1, "x": 2, "f": 3, "k": 4, "s": 5},                                                   // cap-shape
		1:  {"f": 0, "g": 1, "y": 2, "s": 3},                                                                   // cap-surface
		2:  {"n": 0, "b": 1, "c": 2, "g": 3, "r": 4, "p": 5, "u": 6, "e": 7, "w": 8, "y": 9},                   // cap-color
		3:  {"t": 0, "f": 1},                                                                                   // bruises?
		4:  {"a": 0, "l": 1, "c": 2, "y": 3, "f": 4, "m": 5, "n": 6, "p": 7, "s": 8},                           // odor
		5:  {"a": 0, "d": 1, "f": 2, "n": 3},                                                                   // gill-attachment
		6:  {"c": 0, "w": 1, "d": 2},                                                                           // gill-spacing
		7:  {"b": 0, "n": 1},                                                                                   // gill-size
		8:  {"k": 0, "n": 1, "b": 2, "h": 3, "g": 4, "r": 5, "o": 6, "p": 7, "u": 8, "e": 9, "w": 10, "y": 11}, // gill-color
		9:  {"e": 0, "t": 1},                                                                                   // stalk-shape
		10: {"b": 0, "c": 1, "u": 2, "e": 3, "z": 4, "r": 5, "?": 6},                                           // stalk-root
		11: {"f": 0, "y": 1, "k": 2, "s": 3},                                                                   // stalk-surface-above-ring
		12: {"f": 0, "y": 1, "k": 2, "s": 3},                                                                   // stalk-surface-below-ring
		13: {"n": 0, "b": 1, "c": 2, "g": 3, "o": 4, "p": 5, "e": 6, "w": 7, "y": 8},                           // stalk-color-above-ring
		14: {"n": 0, "b": 1, "c": 2, "g": 3, "o": 4, "p": 5, "e": 6, "w": 7, "y": 8},                           // stalk-color-below-ring
		15: {"p": 0, "u": 1},                                                                                   // veil-type
		16: {"n": 0, "o": 1, "w": 2, "y": 3},                                                                   // veil-color
		17: {"n": 0, "o": 1, "t": 2},                                                                           // ring-number
		18: {"c": 0, "e": 1, "f": 2, "l": 3, "n": 4, "p": 5, "s": 6, "z": 7},                                   // ring-type
		19: {"k": 0, "n": 1, "b": 2, "h": 3, "r": 4, "o": 5, "u": 6, "w": 7, "y": 8},                           // spore-print-color
		20: {"a": 0, "c": 1, "n": 2, "s": 3, "v": 4, "y": 5},                                                   // population
		21: {"g": 0, "l": 1, "m": 2, "p": 3, "u": 4, "w": 5, "d": 6},                                           // habitat
	}

	err := forEachLine(path, func(line string) {
		atts := strings.Split(line, ",")

		class := strings.TrimSpace(atts[0])

		instance := make(ifcat.Vector, 22)

		for i := range 22 {
			att := strings.TrimSpace(atts[i+1])

			if mapping, exists := featureMaps[i][att]; exists {
				instance[i] = mapping
			} else {
				panic("invalid attribute")
			}
		}

		// anomaly: poisonous
		if class == "e" {
			normalDataset = append(normalDataset, instance)
		} else {
			anomalyDataset = append(anomalyDataset, instance)
		}
	})

	if err != nil {
		panic(err)
	}

	return normalDataset, anomalyDataset
}

// -------------------------------------------------------------------

func forEachLine(path string, callback func(line string)) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		callback(scanner.Text())
	}

	return scanner.Err()
}

func mustLookup(m map[string]float64, key string) float64 {
	v, ok := m[key]
	if !ok {
		panic("unknown category: " + key)
	}
	return v
}

// writeCSV save scores in a csv file so we can analyze it later using a python script
func writeCSV(nomalScores []float64, anomalyScores []float64) {
	file, err := os.Create("scores.csv")
	if err != nil {
		panic("can not create csv file")
	}
	defer file.Close()

	fmt.Fprintln(file, "score,label")

	for _, score := range anomalyScores {
		fmt.Fprintf(file, "%f,1\n", score)
	}

	for _, score := range nomalScores {
		fmt.Fprintf(file, "%f,0\n", score)
	}
}
