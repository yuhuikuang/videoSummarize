package main

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

func dataRoot() string { return filepath.Join(".", "data") }

func newID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x", b)
}

func runFFmpeg(args []string) error {
	cmd := exec.Command("ffmpeg", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// detectGPUType detects available GPU hardware acceleration
func detectGPUType() string {
	// Check for NVIDIA GPU (NVENC)
	if checkFFmpegEncoder("h264_nvenc") {
		return "nvidia"
	}
	// Check for AMD GPU (AMF)
	if checkFFmpegEncoder("h264_amf") {
		return "amd"
	}
	// Check for Intel GPU (QSV)
	if checkFFmpegEncoder("h264_qsv") {
		return "intel"
	}
	return "cpu"
}

// checkFFmpegEncoder checks if a specific encoder is available in ffmpeg
func checkFFmpegEncoder(encoder string) bool {
	cmd := exec.Command("ffmpeg", "-encoders")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = nil // Suppress stderr
	if err := cmd.Run(); err != nil {
		return false
	}
	return strings.Contains(out.String(), encoder)
}

// getHardwareAccelArgs returns ffmpeg arguments for hardware acceleration
func getHardwareAccelArgs(gpuType string) []string {
	switch gpuType {
	case "nvidia":
		return []string{"-hwaccel", "cuda", "-hwaccel_output_format", "cuda"}
	case "amd":
		return []string{"-hwaccel", "d3d11va"}
	case "intel":
		return []string{"-hwaccel", "qsv"}
	default:
		return []string{} // CPU fallback
	}
}

// getHardwareEncoder returns the appropriate hardware encoder
func getHardwareEncoder(gpuType string) string {
	switch gpuType {
	case "nvidia":
		return "h264_nvenc"
	case "amd":
		return "h264_amf"
	case "intel":
		return "h264_qsv"
	default:
		return "libx264" // CPU fallback
	}
}

func probeDuration(path string) (float64, error) {
	cmd := exec.Command("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil { return 0, err }
	s := strings.TrimSpace(out.String())
	return strconv.ParseFloat(s, 64)
}

func mustJSON(v any) []byte {
	b, _ := json.MarshalIndent(v, "", "  ")
	return b
}

func truncateWords(s string, n int) string {
	toks := tokenize(s)
	if len(toks) <= n { return s }
	return strings.Join(toks[:n], " ") + "..."
}

func absFloat(x float64) float64 { if x < 0 { return -x }; return x }

func min(a, b int) int { if a < b { return a }; return b }

var nonLetter = regexp.MustCompile(`[^a-zA-Z0-9\p{Han}]+`)
var stops = map[string]struct{}{"the":{},"and":{},"a":{},"an":{},"of":{},"to":{},"in":{},"is":{},"are":{},"for":{},"on":{},"with":{},"that":{},"this":{},"it":{},"as":{},"at":{},"be":{},"by":{},"from":{}}

func tokenize(s string) []string {
	s = strings.ToLower(s)
	s = nonLetter.ReplaceAllString(s, " ")
	parts := strings.Fields(s)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if _, ok := stops[p]; ok { continue }
		if p == "" { continue }
		out = append(out, p)
	}
	return out
}

func formatTime(sec float64) string {
	sec = math.Max(sec, 0)
	m := int(sec) / 60
	s := int(sec) % 60
	return fmt.Sprintf("%02d:%02d", m, s)
}