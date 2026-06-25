// Package memory is the Go port of cofiswarm_orchestrate.memory_utils: the host
// RAM snapshot and per-mode memory cap used to guard mode startup (MS-25).
package memory

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const fallbackAvailableGB = 16.0

// roundGB matches Python round(bytes/1024^3, 1).
func roundGB(bytesVal float64) float64 {
	return math.Round(bytesVal/(1024*1024*1024)*10) / 10
}

var darwinFreeRE = regexp.MustCompile(`(?s)Pages free:\s+(\d+).*?Pages inactive:\s+(\d+).*?Pages speculative:\s+(\d+)`)
var darwinPageRE = regexp.MustCompile(`page size of (\d+) bytes`)

// AvailableGB returns available host RAM in GB for memory-cap checks, with a
// conservative fallback when detection fails.
func AvailableGB() float64 {
	if free, ok := freeGB(); ok {
		return free
	}
	log.Printf("host memory unavailable on %s — using %.1f GB fallback", runtime.GOOS, fallbackAvailableGB)
	return fallbackAvailableGB
}

func freeGB() (float64, bool) {
	switch runtime.GOOS {
	case "darwin":
		return darwinFreeGB()
	case "linux":
		return linuxFreeGB()
	default:
		return 0, false
	}
}

func darwinFreeGB() (float64, bool) {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		log.Printf("darwin memory snapshot failed: %v", err)
		return 0, false
	}
	totalBytes, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil || totalBytes <= 0 {
		return 0, false
	}
	vm, err := exec.Command("vm_stat").Output()
	if err != nil {
		log.Printf("darwin vm_stat failed: %v", err)
		return 0, false
	}
	pageSize := int64(4096)
	if m := darwinPageRE.FindSubmatch(vm); m != nil {
		if ps, err := strconv.ParseInt(string(m[1]), 10, 64); err == nil {
			pageSize = ps
		}
	}
	m := darwinFreeRE.FindSubmatch(vm)
	if m == nil {
		return 0, false
	}
	var pages int64
	for i := 1; i <= 3; i++ {
		n, _ := strconv.ParseInt(string(m[i]), 10, 64)
		pages += n
	}
	return roundGB(float64(pages * pageSize)), true
}

var meminfoRE = regexp.MustCompile(`^(\w+):\s+(\d+)\s+kB$`)

func linuxFreeGB() (float64, bool) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		log.Printf("linux meminfo read failed: %v", err)
		return 0, false
	}
	defer f.Close()
	var totalKB, availKB, freeKB, buffersKB, cachedKB int64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		m := meminfoRE.FindStringSubmatch(strings.TrimSpace(sc.Text()))
		if m == nil {
			continue
		}
		kb, _ := strconv.ParseInt(m[2], 10, 64)
		switch m[1] {
		case "MemTotal":
			totalKB = kb
		case "MemAvailable":
			availKB = kb
		case "MemFree":
			freeKB = kb
		case "Buffers":
			buffersKB = kb
		case "Cached":
			cachedKB = kb
		}
	}
	if totalKB <= 0 {
		return 0, false
	}
	if availKB <= 0 {
		availKB = freeKB + buffersKB + cachedKB
	}
	return roundGB(float64(availKB * 1024)), true
}

// preferredBackend mirrors memory_utils.preferred_backend (only the scale need).
func preferredBackend() string {
	switch runtime.GOOS {
	case "darwin":
		return "mlx"
	case "linux":
		if which("nvidia-smi") && dockerVLLMReachable() {
			return "vllm"
		}
		if which("llama-server") {
			return "llama.cpp"
		}
		return "vllm"
	default:
		return "llama.cpp"
	}
}

func modeMemoryWeightScale() float64 {
	if runtime.GOOS == "darwin" {
		return 1.0
	}
	if preferredBackend() == "vllm" {
		return 1.15
	}
	return 1.0
}

// Mirror src/utils/modeManifestData.js python-mode memoryWeight values.
var modeMemoryWeight = map[string]float64{
	"map_reduce":      3.0,
	"speculative":     2.0,
	"critic_debate":   2.0,
	"tree_of_thought": 3.0,
}

const gbPerWeight = 4.0

// RequiredGB is the host RAM required before starting a mode.
func RequiredGB(modeID string) float64 {
	weight, ok := modeMemoryWeight[modeID]
	if !ok {
		weight = 2.0
	}
	return weight * gbPerWeight * modeMemoryWeightScale()
}

// CheckModeMemoryOK returns (ok, errMessage) using a live snapshot.
func CheckModeMemoryOK(modeID string) (bool, string) {
	required := RequiredGB(modeID)
	free := AvailableGB()
	if free < required {
		return false, fmt.Sprintf("insufficient host memory for %q: %.1f GB free, ~%.1f GB required",
			modeID, free, required)
	}
	return true, ""
}

func which(bin string) bool {
	_, err := exec.LookPath(bin)
	return err == nil
}

// dockerVLLMReachable is true when at least one vLLM port responds.
func dockerVLLMReachable() bool {
	for _, p := range []int{8080, 8081, 8082, 8083} {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", p), 400*time.Millisecond)
		if err == nil {
			conn.Close()
			return true
		}
	}
	return false
}
