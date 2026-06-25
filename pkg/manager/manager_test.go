package manager

import (
	"os"
	"path/filepath"
	"testing"
)

func writeAgent(t *testing.T, dir, file, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, file), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadSwarmValid(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("MODELS", "/m")
	writeAgent(t, dir, "scout.json", `{"agent_id":"scout","name":"Scout","model":"${MODELS}/x.gguf",
		"system_prompt":"be terse","context":4096,"max_tokens":256,"engine":"mlx","port":8083,
		"use_rag":true,"rag_top_k":5}`)
	writeAgent(t, dir, "synth.json", `{"agent_id":"synth","name":"Synth","model":"/abs/y.gguf",
		"system_prompt":"merge","context":4096,"max_tokens":512}`)

	swarm, err := New(dir).LoadSwarm()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(swarm) != 2 {
		t.Fatalf("want 2 agents, got %d", len(swarm))
	}
	if swarm["scout"].Model != "/m/x.gguf" {
		t.Errorf("model expand = %q", swarm["scout"].Model)
	}
	if !swarm["scout"].UseRag || swarm["scout"].RagTopK == nil || *swarm["scout"].RagTopK != 5 {
		t.Errorf("rag fields not parsed: %+v", swarm["scout"])
	}
	if m := swarm["scout"].ToMode(); m.Engine != "mlx" || m.MaxTokens != 256 {
		t.Errorf("ToMode = %+v", m)
	}
}

func TestLoadSwarmFailsLoudly(t *testing.T) {
	cases := map[string]string{
		"missing-prompt": `{"agent_id":"a","name":"A","model":"/x","context":1,"max_tokens":1}`,
		"bad-context":    `{"agent_id":"a","name":"A","model":"/x","system_prompt":"p","context":0,"max_tokens":1}`,
		"whitespace-id":  `{"agent_id":"a b","name":"A","model":"/x","system_prompt":"p","context":1,"max_tokens":1}`,
		"unresolved-var": `{"agent_id":"a","name":"A","model":"${NOPE_UNSET}/x","system_prompt":"p","context":1,"max_tokens":1}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			writeAgent(t, dir, "a.json", body)
			if _, err := New(dir).LoadSwarm(); err == nil {
				t.Errorf("%s: expected validation error", name)
			}
		})
	}
}

func TestLoadSwarmEmptyDir(t *testing.T) {
	if _, err := New(t.TempDir()).LoadSwarm(); err == nil {
		t.Error("expected error for empty agents dir")
	}
	if _, err := New("/no/such/dir").LoadSwarm(); err == nil {
		t.Error("expected error for missing dir")
	}
}

func TestDuplicateAgentID(t *testing.T) {
	dir := t.TempDir()
	body := `{"agent_id":"dup","name":"A","model":"/x","system_prompt":"p","context":1,"max_tokens":1}`
	writeAgent(t, dir, "a.json", body)
	writeAgent(t, dir, "b.json", body)
	if _, err := New(dir).LoadSwarm(); err == nil {
		t.Error("expected duplicate agent_id error")
	}
}
