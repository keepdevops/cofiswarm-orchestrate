package orchestrate

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/keepdevops/cofiswarm-tools/pkg/mode"
)

// handleOrchestrateStream is the SSE streaming variant of handleOrchestrate. It
// emits token/agent_start/agent_end/error events and a final "done" with the
// accumulated result and per-agent timings.
func (s *Service) handleOrchestrateStream(w http.ResponseWriter, r *http.Request) {
	req, ok := s.parse(w, r)
	if !ok {
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	h.Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(http.StatusOK)

	send := func(event string, data any) {
		b, _ := json.Marshal(data)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		flusher.Flush()
	}

	var resultParts []string
	finalMeta := map[string]any{}
	tokenCounts := map[string]int{}
	agentElapsed := map[string]float64{}
	agentStart := map[string]time.Time{}

	err := req.mode.Execute(r.Context(), req.modeCtx, req.prompt, func(e mode.Event) error {
		switch e.Kind {
		case "token":
			resultParts = append(resultParts, e.Text)
			if e.AgentID != "" {
				tokenCounts[e.AgentID] += len(strings.Fields(e.Text))
			}
			send("token", map[string]any{"agent_id": e.AgentID, "text": e.Text})
		case "agent_start":
			if e.AgentID != "" {
				agentStart[e.AgentID] = time.Now()
			}
			send("agent_start", map[string]any{"agent_id": e.AgentID, "meta": e.Meta})
		case "agent_end":
			if e.AgentID != "" {
				if t0, ok := agentStart[e.AgentID]; ok {
					agentElapsed[e.AgentID] += float64(time.Since(t0).Milliseconds())
					delete(agentStart, e.AgentID)
				}
			}
			send("agent_end", map[string]any{"agent_id": e.AgentID})
		case "result":
			if e.Meta != nil {
				finalMeta = cloneBody(e.Meta)
			}
		case "error":
			log.Printf("orchestrate/stream: mode=%s agent=%s error: %s", req.modeID, e.AgentID, e.Text)
			send("error", map[string]any{"agent_id": e.AgentID, "error": e.Text})
		}
		return nil
	})
	if err != nil {
		log.Printf("orchestrate/stream: mode=%s session=%s failed: %v", req.modeID, req.sessionID, err)
		send("error", map[string]any{"agent_id": nil, "error": err.Error()})
		return
	}

	if rc, ok := req.params["rag_context"].([]map[string]any); ok && len(rc) > 0 {
		finalMeta["rag_chunks"] = rc
	}
	timings := map[string]any{}
	for id := range mergeKeys(tokenCounts, agentElapsed) {
		timings[id] = map[string]any{
			"completion_tokens": tokenCounts[id],
			"total_ms":          int(agentElapsed[id]),
		}
	}
	if len(timings) > 0 {
		finalMeta["timings"] = timings
	}
	send("done", map[string]any{
		"result": strings.Join(resultParts, ""), "session_id": req.sessionID,
		"mode": req.modeID, "meta": finalMeta,
	})
}

func mergeKeys(a map[string]int, b map[string]float64) map[string]struct{} {
	out := make(map[string]struct{}, len(a)+len(b))
	for k := range a {
		out[k] = struct{}{}
	}
	for k := range b {
		out[k] = struct{}{}
	}
	return out
}

// newUUID returns a random RFC-4122 v4 UUID string (session id fallback).
func newUUID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		log.Printf("orchestrate: uuid rand failed: %v", err)
		return "session-fallback"
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
