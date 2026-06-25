// Package bus wires orchestrate onto the NATS observer bus via the shared
// cofiswarm-observer-sdk service component: it announces presence and serves
// .orchestrate.{info,health}. Go port of cofiswarm_orchestrate.observer.
package bus

import (
	"github.com/keepdevops/cofiswarm-observer-sdk/pkg/servicecomponent"
)

var modes = []string{"map_reduce", "speculative", "critic_debate", "tree_of_thought"}

// Routes wires orchestrate's capability subjects (.orchestrate.{info,health}).
func Routes() map[string]servicecomponent.Handler {
	return map[string]servicecomponent.Handler{
		servicecomponent.Prefix + ".orchestrate.info":   infoHandler(),
		servicecomponent.Prefix + ".orchestrate.health": healthHandler(),
	}
}

// InferMLXRoutes wires the same HTTP service under the mlx-engine identity
// (.infer.mlx.{info,health}) — the Go replacement for cofiswarm-infer-mlx's
// duplicate sidecar.
func InferMLXRoutes() map[string]servicecomponent.Handler {
	return map[string]servicecomponent.Handler{
		servicecomponent.Prefix + ".infer.mlx.info": func([]byte) (any, error) {
			return engineReply{SchemaVersion: servicecomponent.SchemaVersion, OK: true, Engine: "mlx", Stub: false}, nil
		},
		servicecomponent.Prefix + ".infer.mlx.health": healthHandler(),
	}
}

func infoHandler() servicecomponent.Handler {
	return func([]byte) (any, error) {
		return infoReply{SchemaVersion: servicecomponent.SchemaVersion, OK: true,
			Component: "orchestrate", Modes: modes}, nil
	}
}

func healthHandler() servicecomponent.Handler {
	return func([]byte) (any, error) {
		return healthReply{SchemaVersion: servicecomponent.SchemaVersion, OK: true, Status: "ok"}, nil
	}
}

type infoReply struct {
	SchemaVersion string   `json:"schema_version"`
	OK            bool     `json:"ok"`
	Error         string   `json:"error,omitempty"`
	Component     string   `json:"component"`
	Modes         []string `json:"modes"`
}

type healthReply struct {
	SchemaVersion string `json:"schema_version"`
	OK            bool   `json:"ok"`
	Error         string `json:"error,omitempty"`
	Status        string `json:"status"`
}

type engineReply struct {
	SchemaVersion string `json:"schema_version"`
	OK            bool   `json:"ok"`
	Error         string `json:"error,omitempty"`
	Engine        string `json:"engine"`
	Stub          bool   `json:"stub"`
}
