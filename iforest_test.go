package ifcat_test

import (
	"ifcat-go"
	"testing"
)

// Initialize test dataset
func init() {

}

func TestForest_AnomalyScore(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for receiver constructor.
		t               int
		subsamplingSize int
		// Named input parameters for target function.
		x    ifcat.Vector
		want float64
	}{
		{
			name:            "Numbertical varibles",
			t:               100,
			subsamplingSize: 100,

			want: 0.3833,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := ifcat.NewForest(tt.t, tt.subsamplingSize)
			got := f.AnomalyScore(tt.x)
			// TODO: update the condition below to compare got with tt.want.
			if true {
				t.Errorf("AnomalyScore() = %v, want %v", got, tt.want)
			}
		})
	}
}
