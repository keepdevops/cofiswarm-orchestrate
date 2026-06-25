// Package manager is the Go port of cofiswarm_orchestrate.manager: it discovers
// and validates per-agent JSON configs under config/agents/ at startup, failing
// loudly on missing required fields (CLAUDE.md §2).
package manager

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/keepdevops/cofiswarm-tools/pkg/mode"
)

// defaultModelDir mirrors scripts/matrix-env.sh — the MATRIX_MODEL_DIR fallback.
func defaultModelDir() string {
	if runtime.GOOS == "darwin" {
		return "/Users/Shared/llama/models"
	}
	return ""
}

func init() {
	if os.Getenv("MATRIX_MODEL_DIR") == "" {
		_ = os.Setenv("MATRIX_MODEL_DIR", defaultModelDir())
	}
}

// AgentConfig is the schema for one agent file under config/agents/<slug>.json,
// matching the Pydantic AgentConfig. Pointer fields are optional.
type AgentConfig struct {
	AgentID      string `json:"agent_id"`
	Name         string `json:"name"`
	Model        string `json:"model"`
	SystemPrompt string `json:"system_prompt"`
	Context      int    `json:"context"`
	MaxTokens    int    `json:"max_tokens"`

	Engine          *string `json:"engine,omitempty"`
	ServerGroup     *string `json:"server_group,omitempty"`
	Port            *int    `json:"port,omitempty"`
	GPULayers       *int    `json:"gpu_layers,omitempty"`
	NBatch          *int    `json:"n_batch,omitempty"`
	ReadTimeoutSecs *int    `json:"read_timeout_secs,omitempty"`
	MaxConcurrency  *int    `json:"max_concurrency,omitempty"`
	Coordinator     *string `json:"coordinator,omitempty"`

	UseRag  bool           `json:"use_rag"`
	RagTopK *int           `json:"rag_top_k,omitempty"`
	Rag     map[string]any `json:"rag,omitempty"`
}

// validate enforces the required-field constraints and expands the model path.
func (c *AgentConfig) validate() error {
	if strings.TrimSpace(c.AgentID) == "" {
		return fmt.Errorf("agent_id required")
	}
	for _, r := range c.AgentID {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			return fmt.Errorf("agent_id must be slug-form (no whitespace)")
		}
	}
	if c.Name == "" {
		return fmt.Errorf("name required")
	}
	if c.Model == "" {
		return fmt.Errorf("model required")
	}
	if c.SystemPrompt == "" {
		return fmt.Errorf("system_prompt required")
	}
	if c.Context <= 0 {
		return fmt.Errorf("context must be > 0")
	}
	if c.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be > 0")
	}
	expanded, err := expandModelPath(c.Model)
	if err != nil {
		return err
	}
	c.Model = expanded
	return nil
}

// expandModelPath mirrors Python os.path.expandvars(expanduser(...)) plus the
// fail-loud check: an unresolved ${VAR} would silently break llama-server.
func expandModelPath(v string) (string, error) {
	s := v
	if strings.HasPrefix(s, "~") {
		if home, err := os.UserHomeDir(); err == nil {
			s = home + s[1:]
		}
	}
	// os.Expand, but leave unset ${VAR} literally in place (Python expandvars
	// behaviour) so the "$" check below catches it.
	s = os.Expand(s, func(k string) string {
		if val, ok := os.LookupEnv(k); ok {
			return val
		}
		return "${" + k + "}"
	})
	if strings.Contains(s, "$") {
		return "", fmt.Errorf("unresolved env var in model path: %s", v)
	}
	return s, nil
}

// ToMode projects the subset of fields the orchestration modes read.
func (c AgentConfig) ToMode() mode.AgentConfig {
	return mode.AgentConfig{
		Name:         c.Name,
		SystemPrompt: c.SystemPrompt,
		MaxTokens:    c.MaxTokens,
		Engine:       deref(c.Engine),
		ServerGroup:  deref(c.ServerGroup),
	}
}

// SwarmFactory discovers and loads agent configs into an in-memory registry.
type SwarmFactory struct {
	AgentsDir string
}

// DefaultAgentsDir resolves config/agents under COFISWARM_CONFIG_ROOT's parent.
func DefaultAgentsDir() string {
	root := os.Getenv("COFISWARM_CONFIG_ROOT")
	if root == "" {
		root = "/etc/cofiswarm/config"
	}
	return filepath.Join(filepath.Dir(root), "config", "agents")
}

// New constructs a factory; an empty dir uses DefaultAgentsDir.
func New(agentsDir string) *SwarmFactory {
	if agentsDir == "" {
		agentsDir = DefaultAgentsDir()
	}
	return &SwarmFactory{AgentsDir: agentsDir}
}

// LoadSwarm reads every *.json under AgentsDir, validating each and rejecting
// duplicate agent_ids. Fails loudly (mirrors load_swarm).
func (f *SwarmFactory) LoadSwarm() (map[string]AgentConfig, error) {
	info, err := os.Stat(f.AgentsDir)
	if err != nil || !info.IsDir() {
		return nil, fmt.Errorf("agents dir missing: %s", f.AgentsDir)
	}
	paths, err := filepath.Glob(filepath.Join(f.AgentsDir, "*.json"))
	if err != nil {
		return nil, fmt.Errorf("glob %s: %w", f.AgentsDir, err)
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		return nil, fmt.Errorf("no agents found in %s", f.AgentsDir)
	}

	swarm := make(map[string]AgentConfig, len(paths))
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("failed to read agent config %s: %w", path, err)
		}
		var cfg AgentConfig
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		if err := dec.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("invalid agent config %s: %w", path, err)
		}
		if err := cfg.validate(); err != nil {
			return nil, fmt.Errorf("invalid agent config %s: %w", path, err)
		}
		if _, dup := swarm[cfg.AgentID]; dup {
			return nil, fmt.Errorf("duplicate agent_id: %s (%s)", cfg.AgentID, path)
		}
		swarm[cfg.AgentID] = cfg
	}
	log.Printf("[BOOT] loaded %d isolated agents from %s", len(swarm), f.AgentsDir)
	return swarm, nil
}

// ModeSwarm projects a loaded swarm to the mode-facing AgentConfig map.
func ModeSwarm(swarm map[string]AgentConfig) map[string]mode.AgentConfig {
	out := make(map[string]mode.AgentConfig, len(swarm))
	for id, cfg := range swarm {
		out[id] = cfg.ToMode()
	}
	return out
}

func deref(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}
