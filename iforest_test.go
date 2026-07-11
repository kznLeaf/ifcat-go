package ifcat_test

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/kznLeaf/ifcat-go"
)

// ---------------------- Car Evaluation ---------------------------------

func BenchmarkCarScore(b *testing.B) {
	path := "./testdata/car_evaluation/car.data"

	f, _, anomalyDataset := newCategoricalVariablesForest(path)

	b.ResetTimer()
	for range b.N {
		// 6000 ns/op
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
	f.AddField("buying", ifcat.TypeCategorical)
	f.AddField("maint", ifcat.TypeCategorical)
	f.AddField("doors", ifcat.TypeCategorical)
	f.AddField("persons", ifcat.TypeCategorical)
	f.AddField("lug_boot", ifcat.TypeCategorical)
	f.AddField("safety", ifcat.TypeCategorical)

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
		anomalyScores[i] = f.AnomalyScore(data)
	}

	for i, data := range normalDataset {
		normalScores[i] = f.AnomalyScore(data)
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
	// for i := range 20 {
	// 	t.Log(dataset[i])
	// }

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

// -------------------------------------------------------------------

func TestKdd99_10(t *testing.T) {
	path := "./testdata/kdd99/kddcup.data_10_percent.txt"
	// Train forest based on full dataset
	f, normalDataset, anomalyDataset := newKDDCupForest(t, path)

	anomalyScores := make([]float64, len(anomalyDataset))
	normalScores := make([]float64, len(normalDataset))

	t.Logf("anomaly samples: %v, normal samples: %v", len(anomalyDataset), len(normalDataset))

	for i, data := range anomalyDataset {
		anomalyScores[i] = f.AnomalyScore(data)
	}

	for i, data := range normalDataset {
		normalScores[i] = f.AnomalyScore(data)
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

	f.AddField("duration", ifcat.TypeNumerical)
	f.AddField("protocol_type", ifcat.TypeCategorical)
	f.AddField("service", ifcat.TypeCategorical)
	f.AddField("flag", ifcat.TypeCategorical)
	f.AddField("src_bytes", ifcat.TypeNumerical)
	f.AddField("dst_bytes", ifcat.TypeNumerical)
	f.AddField("land", ifcat.TypeBool)
	f.AddField("wrong_fragment", ifcat.TypeNumerical)
	f.AddField("urgent", ifcat.TypeNumerical)
	f.AddField("hot", ifcat.TypeNumerical)
	f.AddField("num_failed_logins", ifcat.TypeNumerical)
	f.AddField("logged_in", ifcat.TypeBool)
	f.AddField("num_compromised", ifcat.TypeNumerical)
	f.AddField("root_shell", ifcat.TypeNumerical)
	f.AddField("su_attempted", ifcat.TypeNumerical)
	f.AddField("num_root", ifcat.TypeNumerical)
	f.AddField("num_file_creations", ifcat.TypeNumerical)
	f.AddField("num_shells", ifcat.TypeNumerical)
	f.AddField("num_access_files", ifcat.TypeNumerical)
	f.AddField("num_outbound_cmds", ifcat.TypeNumerical)
	f.AddField("is_host_login", ifcat.TypeBool)
	f.AddField("is_guest_login", ifcat.TypeBool)
	f.AddField("count", ifcat.TypeNumerical)
	f.AddField("srv_count", ifcat.TypeNumerical)
	f.AddField("serror_rate", ifcat.TypeNumerical)
	f.AddField("srv_serror_rate", ifcat.TypeNumerical)
	f.AddField("rerror_rate", ifcat.TypeNumerical)
	f.AddField("srv_rerror_rate", ifcat.TypeNumerical)
	f.AddField("same_srv_rate", ifcat.TypeNumerical)
	f.AddField("diff_srv_rate", ifcat.TypeNumerical)
	f.AddField("srv_diff_host_rate", ifcat.TypeNumerical)
	f.AddField("dst_host_count", ifcat.TypeNumerical)
	f.AddField("dst_host_srv_count", ifcat.TypeNumerical)
	f.AddField("dst_host_same_srv_rate", ifcat.TypeNumerical)
	f.AddField("dst_host_diff_srv_rate", ifcat.TypeNumerical)
	f.AddField("dst_host_same_src_port_rate", ifcat.TypeNumerical)
	f.AddField("dst_host_srv_diff_host_rate", ifcat.TypeNumerical)
	f.AddField("dst_host_serror_rate", ifcat.TypeNumerical)
	f.AddField("dst_host_srv_serror_rate", ifcat.TypeNumerical)
	f.AddField("dst_host_rerror_rate", ifcat.TypeNumerical)
	f.AddField("dst_host_srv_rerror_rate", ifcat.TypeNumerical)

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

// -------------------------------------------------------------------

func TestMushroom(t *testing.T) {
	path := "./testdata/mushroom/agaricus-lepiota.data"
	// Train forest based on car_evaluation normalDataset
	f, normalDataset, anomalyDataset := newMushroomForest(t, path)

	anomalyScores := make([]float64, len(anomalyDataset))
	normalScores := make([]float64, len(normalDataset))

	for i, data := range anomalyDataset {
		anomalyScores[i] = f.AnomalyScore(data)
	}

	for i, data := range normalDataset {
		normalScores[i] = f.AnomalyScore(data)
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

	f.AddField("cap_shape", ifcat.TypeCategorical)
	f.AddField("cap_surface", ifcat.TypeCategorical)
	f.AddField("cap_color", ifcat.TypeCategorical)
	f.AddField("bruises", ifcat.TypeCategorical)
	f.AddField("odor", ifcat.TypeCategorical)
	f.AddField("gill_attachment", ifcat.TypeCategorical)
	f.AddField("gill_spacing", ifcat.TypeCategorical)
	f.AddField("gill_size", ifcat.TypeCategorical)
	f.AddField("gill_color", ifcat.TypeCategorical)
	f.AddField("stalk_shape", ifcat.TypeCategorical)
	f.AddField("stalk_root", ifcat.TypeCategorical)
	f.AddField("stalk_surface_above_ring", ifcat.TypeCategorical)
	f.AddField("stalk_surface_below_ring", ifcat.TypeCategorical)
	f.AddField("stalk_color_above_ring", ifcat.TypeCategorical)
	f.AddField("stalk_color_below_ring", ifcat.TypeCategorical)
	f.AddField("veil_type", ifcat.TypeCategorical)
	f.AddField("veil_color", ifcat.TypeCategorical)
	f.AddField("ring_number", ifcat.TypeCategorical)
	f.AddField("ring_type", ifcat.TypeCategorical)
	f.AddField("spore_print_color", ifcat.TypeCategorical)
	f.AddField("population", ifcat.TypeCategorical)
	f.AddField("habitat", ifcat.TypeCategorical)

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
