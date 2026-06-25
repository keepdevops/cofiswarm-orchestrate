// Package orchestrate is the Go port of mlx_coordinator.service_orchestrate +
// sidecar: it serves POST /api/orchestrate[/stream], dispatching to the Go
// orchestration modes over MLX backends, with the per-agent RAG bridge and the
// host-memory guard.
package orchestrate

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/keepdevops/cofiswarm-backend-mlx/pkg/mlx"
	"github.com/keepdevops/cofiswarm-backend-sdk/pkg/backend"
	"github.com/keepdevops/cofiswarm-orchestrate/pkg/manager"
	"github.com/keepdevops/cofiswarm-orchestrate/pkg/memory"
	"github.com/keepdevops/cofiswarm-tools/pkg/mode"
)

// Service holds the loaded swarm and per-engine backends shared by both handlers.
type Service struct {
	Swarm    map[string]manager.AgentConfig
	Backends map[string]backend.InferenceBackend
	modes    map[string]mode.OrchestrationMode
}

// NewService builds backends from the swarm (one MlxBackend per engine/group —
// llama and mlx speak the same OpenAI API, so MlxBackend works for any engine).
func NewService(swarm map[string]manager.AgentConfig) *Service {
	backends := make(map[string]backend.InferenceBackend, len(swarm))
	for agentID, cfg := range swarm {
		key := firstNonEmpty(deref(cfg.Engine), deref(cfg.ServerGroup), agentID)
		port := 8083
		if cfg.Port != nil && *cfg.Port > 0 {
			port = *cfg.Port
		}
		backends[key] = mlx.New(port, agentID, cfg.SystemPrompt, cfg.MaxTokens, 0)
	}
	return &Service{Swarm: swarm, Backends: backends, modes: mode.Registry()}
}

// Close tears down every backend.
func (s *Service) Close() {
	for _, b := range s.Backends {
		_ = b.Close()
	}
}

// Register wires both orchestrate routes onto a mux.
func (s *Service) Register(mux *http.ServeMux) {
	mux.HandleFunc("/api/orchestrate", s.handleOrchestrate)
	mux.HandleFunc("/api/orchestrate/stream", s.handleOrchestrateStream)
}

// request is the parsed, validated common preamble of both handlers.
type request struct {
	body      map[string]any
	modeID    string
	prompt    string
	mode      mode.OrchestrationMode
	sessionID string
	params    map[string]any
	modeCtx   *mode.ModeContext
}

// parse validates the request, applies the RAG bridge, and assembles the
// ModeContext. It writes the appropriate HTTP error and returns ok=false on
// failure (mirrors the Python guards, fail-loudly).
func (s *Service) parse(w http.ResponseWriter, r *http.Request) (*request, bool) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return nil, false
	}
	var body map[string]any
	dec := json.NewDecoder(r.Body)
	dec.UseNumber()
	if err := dec.Decode(&body); err != nil {
		log.Printf("orchestrate: bad JSON: %v", err)
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return nil, false
	}
	modeID := strings.TrimSpace(getString(body, "mode"))
	if modeID == "" {
		http.Error(w, "'mode' required", http.StatusBadRequest)
		return nil, false
	}
	prompt := strings.TrimSpace(getString(body, "prompt"))
	if prompt == "" {
		http.Error(w, "'prompt' required", http.StatusBadRequest)
		return nil, false
	}
	m, ok := s.modes[modeID]
	if !ok {
		http.Error(w, "unknown Python mode "+modeID, http.StatusBadRequest)
		return nil, false
	}
	if memOK, memErr := memory.CheckModeMemoryOK(modeID); !memOK {
		log.Printf("orchestrate: memory guard blocked mode=%s: %s", modeID, memErr)
		http.Error(w, memErr, http.StatusServiceUnavailable)
		return nil, false
	}

	sessionID := strings.TrimSpace(getString(body, "session_id"))
	if sessionID == "" {
		sessionID = newUUID()
	}
	params := map[string]any{}
	if p, ok := body["params"].(map[string]any); ok {
		params = cloneBody(p)
	}

	// Roster→MLX use_rag bridge (request silent → opt-in from agents), then
	// retrieve top-k chunks into params["rag_context"]; the modes inject it
	// per-agent via rag_xml() — do NOT also prepend it (double-inject regression).
	body = rosterRAGDefault(s.Swarm, body)
	if chunks := ragContextFor(r.Context(), body, prompt); len(chunks) > 0 {
		params["rag_context"] = chunks
	}

	ctx := &mode.ModeContext{
		Swarm:     manager.ModeSwarm(s.Swarm),
		Backends:  s.Backends,
		Agents:    sortedKeys(s.Swarm),
		Params:    params,
		RequestID: sessionID,
	}
	return &request{body, modeID, prompt, m, sessionID, params, ctx}, true
}

// handleOrchestrate runs a mode to completion and returns blocking JSON.
func (s *Service) handleOrchestrate(w http.ResponseWriter, r *http.Request) {
	req, ok := s.parse(w, r)
	if !ok {
		return
	}
	var parts []string
	meta := map[string]any{}
	err := req.mode.Execute(r.Context(), req.modeCtx, req.prompt, func(e mode.Event) error {
		switch e.Kind {
		case "token":
			parts = append(parts, e.Text)
		case "result":
			if e.Meta != nil {
				meta = cloneBody(e.Meta)
			}
		case "error":
			log.Printf("orchestrate: mode=%s agent=%s error: %s", req.modeID, e.AgentID, e.Text)
		}
		return nil
	})
	if err != nil {
		log.Printf("orchestrate: mode=%s session=%s failed: %v", req.modeID, req.sessionID, err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if rc, ok := req.params["rag_context"].([]map[string]any); ok && len(rc) > 0 {
		meta["rag_chunks"] = rc
	}
	writeJSON(w, map[string]any{
		"result": strings.Join(parts, ""), "session_id": req.sessionID,
		"mode": req.modeID, "meta": meta,
	})
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func deref(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func sortedKeys(m map[string]manager.AgentConfig) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	// Stable order so agents[0]/[1] defaults (ToT) are deterministic.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

func getString(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("orchestrate: write JSON: %v", err)
	}
}
