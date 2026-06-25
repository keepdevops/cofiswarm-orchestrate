package orchestrate

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/keepdevops/cofiswarm-orchestrate/pkg/manager"
)

// ragBase resolves the RAG ingest service base URL (RAG_INGEST_HOST:PORT).
func ragBase() string {
	host := os.Getenv("RAG_INGEST_HOST")
	if host == "" {
		host = "http://127.0.0.1"
	}
	port := os.Getenv("RAG_INGEST_PORT")
	if port == "" {
		port = "8001"
	}
	return host + ":" + port
}

// fetchRAGChunks calls the RAG ingest /retrieve endpoint. Non-fatal: returns nil
// on any error so a missing RAG service never breaks orchestration.
func fetchRAGChunks(ctx context.Context, query string, k int) []map[string]any {
	url := ragBase() + "/retrieve"
	body, _ := json.Marshal(map[string]any{"query": query, "k": k})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	cl := &http.Client{Timeout: 10 * time.Second}
	resp, err := cl.Do(req)
	if err != nil {
		log.Printf("rag retrieve failed (non-fatal): %v", err)
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("rag retrieve HTTP %d from %s", resp.StatusCode, url)
		return nil
	}
	var out struct {
		Chunks []map[string]any `json:"chunks"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		log.Printf("rag retrieve decode (non-fatal): %v", err)
		return nil
	}
	return out.Chunks
}

// ragContextFor retrieves top-k chunks for a request, filtered by min_score (the
// max cosine distance to accept, 1.0 = no filter). Returns nil when RAG is off.
func ragContextFor(ctx context.Context, body map[string]any, prompt string) []map[string]any {
	if b, _ := body["use_rag"].(bool); !b {
		return nil
	}
	k := 3
	if v, ok := toInt(body["rag_top_k"]); ok && v > 0 {
		k = v
	}
	minScore := 1.0
	if v, ok := toFloat(body["rag_min_score"]); ok {
		minScore = v
	}
	chunks := fetchRAGChunks(ctx, prompt, k)
	filtered := make([]map[string]any, 0, len(chunks))
	for _, c := range chunks {
		dist, _ := toFloat(c["distance"])
		if dist <= minScore {
			filtered = append(filtered, c)
		}
	}
	return filtered
}

// rosterRAGDefault bridges per-agent use_rag into the request. When the request
// is SILENT on use_rag, enable it if any swarm agent opts in, defaulting
// rag_top_k to the max across opted-in agents. An explicit request value wins.
func rosterRAGDefault(swarm map[string]manager.AgentConfig, body map[string]any) map[string]any {
	if _, present := body["use_rag"]; present {
		return body
	}
	var opted []manager.AgentConfig
	for _, a := range swarm {
		if a.UseRag {
			opted = append(opted, a)
		}
	}
	if len(opted) == 0 {
		return body
	}
	out := cloneBody(body)
	out["use_rag"] = true
	if v, ok := toInt(out["rag_top_k"]); !ok || v == 0 {
		topk := 0
		for _, a := range opted {
			if a.RagTopK != nil && *a.RagTopK > topk {
				topk = *a.RagTopK
			}
		}
		if topk > 0 {
			out["rag_top_k"] = topk
		}
	}
	return out
}

func cloneBody(b map[string]any) map[string]any {
	out := make(map[string]any, len(b)+1)
	for k, v := range b {
		out[k] = v
	}
	return out
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return int(i), true
		}
	}
	return 0, false
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case json.Number:
		if f, err := n.Float64(); err == nil {
			return f, true
		}
	}
	return 0, false
}
