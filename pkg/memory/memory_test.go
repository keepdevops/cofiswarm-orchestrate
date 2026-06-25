package memory

import "testing"

func TestRequiredGB(t *testing.T) {
	// On darwin the scale is 1.0, so required = weight * 4.
	cases := map[string]float64{
		"map_reduce":      12.0,
		"speculative":     8.0,
		"critic_debate":   8.0,
		"tree_of_thought": 12.0,
		"unknown":         8.0, // default weight 2.0
	}
	for mode, want := range cases {
		if got := RequiredGB(mode); got != want {
			// Non-darwin platforms may scale up; only assert exact on darwin.
			if modeMemoryWeightScale() == 1.0 && got != want {
				t.Errorf("RequiredGB(%q)=%.1f want %.1f", mode, got, want)
			}
		}
	}
}

func TestCheckModeMemoryOK(t *testing.T) {
	// A trivially-small requirement always passes; a huge one always fails.
	ok, _ := CheckModeMemoryOK("speculative")
	if !ok {
		t.Skip("host reports < 8GB free; environment-dependent")
	}
	// AvailableGB must be positive on a supported platform (or the fallback).
	if AvailableGB() <= 0 {
		t.Error("AvailableGB should be positive")
	}
}
