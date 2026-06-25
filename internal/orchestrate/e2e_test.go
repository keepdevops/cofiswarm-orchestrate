package orchestrate

import (
	"bufio"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/keepdevops/cofiswarm-orchestrate/pkg/manager"
)

// fakeMLX serves an mlx_lm.server-compatible SSE stream emitting `reply`.
func fakeMLX(reply string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			w.WriteHeader(200)
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			for _, tok := range strings.Fields(reply) {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", tok+" ")
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
}

func portOf(s *httptest.Server) int {
	_, p, _ := strings.Cut(strings.TrimPrefix(s.URL, "http://"), ":")
	n, _ := strconv.Atoi(p)
	return n
}

// End-to-end: POST /api/orchestrate drives critic_debate over two fake MLX
// backends; the critic says SHIP so it returns in round 1.
func TestEndToEndCriticDebate(t *testing.T) {
	gen := fakeMLX("a draft answer")
	defer gen.Close()
	critic := fakeMLX("SHIP")
	defer critic.Close()

	swarm := map[string]manager.AgentConfig{
		"gen":    {AgentID: "gen", Name: "gen", SystemPrompt: "G", MaxTokens: 32, Engine: ptrS("eg"), Port: ptrI(portOf(gen))},
		"critic": {AgentID: "critic", Name: "critic", SystemPrompt: "C", MaxTokens: 32, Engine: ptrS("ec"), Port: ptrI(portOf(critic))},
	}
	svc := NewService(swarm)
	defer svc.Close()
	mux := http.NewServeMux()
	svc.Register(mux)

	body := `{"mode":"critic_debate","prompt":"do it","params":{"generator":"gen","critic":"critic","max_rounds":2}}`
	req := httptest.NewRequest(http.MethodPost, "/api/orchestrate", strings.NewReader(body))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "draft") {
		t.Errorf("expected generator text in result: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"verdict":"SHIP"`) {
		t.Errorf("expected SHIP verdict: %s", rec.Body.String())
	}
}

// End-to-end streaming: /api/orchestrate/stream yields SSE token + done events.
func TestEndToEndStream(t *testing.T) {
	worker := fakeMLX("found something")
	defer worker.Close()
	synth := fakeMLX("final summary")
	defer synth.Close()

	swarm := map[string]manager.AgentConfig{
		"w": {AgentID: "w", Name: "w", SystemPrompt: "W", MaxTokens: 32, Engine: ptrS("ew"), Port: ptrI(portOf(worker))},
		"s": {AgentID: "s", Name: "s", SystemPrompt: "S", MaxTokens: 32, Engine: ptrS("es"), Port: ptrI(portOf(synth))},
	}
	svc := NewService(swarm)
	defer svc.Close()
	mux := http.NewServeMux()
	svc.Register(mux)

	body := `{"mode":"map_reduce","prompt":"q","params":{"chunks":["c0","c1"],"synthesizer":"s"}}`
	req := httptest.NewRequest(http.MethodPost, "/api/orchestrate/stream", strings.NewReader(body))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	var sawToken, sawDone bool
	sc := bufio.NewScanner(strings.NewReader(rec.Body.String()))
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "event: token") {
			sawToken = true
		}
		if strings.HasPrefix(line, "event: done") {
			sawDone = true
		}
	}
	if !sawToken || !sawDone {
		t.Errorf("SSE stream missing events (token=%v done=%v):\n%s", sawToken, sawDone, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "summary") {
		t.Errorf("expected synthesizer output in done event: %s", rec.Body.String())
	}
}
