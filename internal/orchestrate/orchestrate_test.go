package orchestrate

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/keepdevops/cofiswarm-orchestrate/pkg/manager"
)

func ptrI(i int) *int       { return &i }
func ptrS(s string) *string { return &s }

func TestRosterRAGDefault(t *testing.T) {
	swarm := map[string]manager.AgentConfig{
		"a": {AgentID: "a", UseRag: true, RagTopK: ptrI(5)},
		"b": {AgentID: "b", UseRag: false},
	}
	// Silent request → opt-in from roster, top_k = max across opted agents.
	out := rosterRAGDefault(swarm, map[string]any{})
	if out["use_rag"] != true {
		t.Errorf("silent request should enable use_rag: %+v", out)
	}
	if v, _ := toInt(out["rag_top_k"]); v != 5 {
		t.Errorf("rag_top_k = %v, want 5", out["rag_top_k"])
	}
	// Explicit false wins — never overridden.
	out = rosterRAGDefault(swarm, map[string]any{"use_rag": false})
	if out["use_rag"] != false {
		t.Errorf("explicit use_rag:false must win: %+v", out)
	}
	// No opted agents → unchanged.
	none := map[string]manager.AgentConfig{"a": {UseRag: false}}
	out = rosterRAGDefault(none, map[string]any{})
	if _, present := out["use_rag"]; present {
		t.Errorf("no opted agents should leave body silent: %+v", out)
	}
}

func TestRAGContextFor(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/retrieve" {
			w.WriteHeader(404)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"chunks": []map[string]any{
			{"source_path": "near.go", "content": "A", "distance": 0.2},
			{"source_path": "far.go", "content": "B", "distance": 0.9},
		}})
	}))
	defer srv.Close()
	host, port, _ := strings.Cut(strings.TrimPrefix(srv.URL, "http://"), ":")
	t.Setenv("RAG_INGEST_HOST", "http://"+host)
	t.Setenv("RAG_INGEST_PORT", port)

	// use_rag off → nil.
	if c := ragContextFor(context.Background(), map[string]any{}, "q"); c != nil {
		t.Errorf("rag off should yield nil, got %v", c)
	}
	// min_score filters out the far chunk.
	got := ragContextFor(context.Background(),
		map[string]any{"use_rag": true, "rag_min_score": 0.5}, "q")
	if len(got) != 1 || got[0]["source_path"] != "near.go" {
		t.Errorf("min_score filter = %v", got)
	}
}

func newTestService() *Service {
	swarm := map[string]manager.AgentConfig{
		"a1": {AgentID: "a1", Name: "a1", SystemPrompt: "S", MaxTokens: 32, Engine: ptrS("e1"), Port: ptrI(9001)},
	}
	return NewService(swarm)
}

func TestHandlerValidation(t *testing.T) {
	svc := newTestService()
	defer svc.Close()
	mux := http.NewServeMux()
	svc.Register(mux)

	cases := []struct {
		body   string
		status int
	}{
		{`not json`, http.StatusBadRequest},
		{`{"prompt":"hi"}`, http.StatusBadRequest},               // missing mode
		{`{"mode":"map_reduce"}`, http.StatusBadRequest},         // missing prompt
		{`{"mode":"nope","prompt":"hi"}`, http.StatusBadRequest}, // unknown mode
	}
	for _, c := range cases {
		req := httptest.NewRequest(http.MethodPost, "/api/orchestrate", strings.NewReader(c.body))
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != c.status {
			t.Errorf("body=%q → status %d, want %d", c.body, rec.Code, c.status)
		}
	}

	// GET is rejected.
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/orchestrate", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET → %d, want 405", rec.Code)
	}
}

func TestBackendKeying(t *testing.T) {
	svc := newTestService()
	defer svc.Close()
	if _, ok := svc.Backends["e1"]; !ok {
		t.Errorf("backend should be keyed by engine 'e1', have %v", keysOf(svc.Backends))
	}
}

func keysOf[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
