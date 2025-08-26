package main

type Frame struct {
	TimestampSec float64 `json:"timestamp_sec"`
	Path         string  `json:"path"`
}

type PreprocessResponse struct {
	JobID     string  `json:"job_id"`
	AudioPath string  `json:"audio_path"`
	Frames    []Frame `json:"frames"`
}

type Segment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

type TranscribeRequest struct {
	JobID     string `json:"job_id"`
	AudioPath string `json:"audio_path"`
}

type TranscribeResponse struct {
	JobID    string    `json:"job_id"`
	Segments []Segment `json:"segments"`
}

type Item struct {
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Text     string  `json:"text"`
	Summary  string  `json:"summary"`
	FramePath string `json:"frame_path"`
}

type SummarizeRequest struct {
	JobID    string    `json:"job_id"`
	Segments []Segment `json:"segments"`
}

type SummarizeResponse struct {
	JobID string `json:"job_id"`
	Items []Item `json:"items"`
}

type StoreRequest struct {
	JobID string `json:"job_id"`
	Items []Item `json:"items"`
}

type StoreResponse struct {
	JobID string `json:"job_id"`
	Count int    `json:"count"`
}

type Hit struct {
	Score    float64 `json:"score"`
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Text     string  `json:"text"`
	Summary  string  `json:"summary"`
	FramePath string `json:"frame_path"`
}

type QueryRequest struct {
	JobID   string `json:"job_id"`
	Question string `json:"question"`
	TopK    int    `json:"top_k"`
}

type QueryResponse struct {
	JobID    string `json:"job_id"`
	Question string `json:"question"`
	Answer   string `json:"answer"`
	Hits     []Hit  `json:"hits"`
}