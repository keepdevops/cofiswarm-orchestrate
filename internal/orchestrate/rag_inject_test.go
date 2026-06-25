package orchestrate

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/keepdevops/cofiswarm-backend-sdk/pkg/backend"
)

// recordingBackend captures the prompt each GenerateStream receives.
type recordingBackend struct{ prompts []string }

func (b *recordingBackend) GenerateStream(_ context.Context, req backend.GenerateRequest, emit func(backend.TokenChunk) error) error {
	b.prompts = append(b.prompts, req.Prompt)
	return emit(backend.TokenChunk{Done: true})
}
func (b *recordingBackend) Embed(context.Context, []string) ([][]float32, error) { return nil, nil }
func (b *recordingBackend) Health(context.Context) backend.HealthStatus {
	return backend.HealthStatus{OK: true}
}
func (b *recordingBackend) Close() error { return nil }

// Regression guard (ported from the Python rag_inject_once_check): the RAG block
// is injected exactly once on the orchestrate path — the modes inject
// params["rag_context"] via rag_xml(); the service must NOT also prepend its own
// copy. Asserts <retrieved> appears once in every prompt the backend receives.
func TestRAGInjectedExactlyOnce(t *testing.T) {
	rag := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"chunks":[{"source_path":"x.go","content":"CTX","distance":0.1}]}`))
	}))
	defer rag.Close()
	host, port, _ := strings.Cut(strings.TrimPrefix(rag.URL, "http://"), ":")
	t.Setenv("RAG_INGEST_HOST", "http://"+host)
	t.Setenv("RAG_INGEST_PORT", port)

	rec := &recordingBackend{}
	svc := newTestService() // single agent a1 on engine e1
	defer svc.Close()
	svc.Backends["e1"] = rec // capture both map + reduce calls (a1 is worker and synth)

	mux := http.NewServeMux()
	svc.Register(mux)
	body := `{"mode":"map_reduce","prompt":"q","use_rag":true,"params":{"chunks":["c0"],"synthesizer":"a1"}}`
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/orchestrate", strings.NewReader(body)))
	if rr.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rr.Code, rr.Body.String())
	}

	if len(rec.prompts) == 0 {
		t.Fatal("backend received no prompt — RAG path not exercised")
	}
	for i, p := range rec.prompts {
		if !strings.Contains(p, "CTX") {
			t.Errorf("prompt %d missing retrieved content: %s", i, p)
		}
		if n := strings.Count(p, "<retrieved>"); n != 1 {
			t.Errorf("RAG block injected %d times (want exactly 1) in prompt %d:\n%s", n, i, p)
		}
	}
}
