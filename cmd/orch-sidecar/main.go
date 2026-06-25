// Command orch-sidecar is the orchestrate-only HTTP sidecar (Go port of
// cofiswarm_orchestrate.mlx_coordinator.sidecar). It serves /api/orchestrate*
// on ORCH_SIDECAR_PORT (default 3003), handling the map_reduce, speculative,
// critic_debate, and tree_of_thought modes. All agents speak the OpenAI
// /v1/chat/completions API, so the MLX backend works for any engine. It also
// replaces the duplicate cofiswarm-infer-mlx sidecar entrypoint.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/keepdevops/cofiswarm-observer-sdk/pkg/servicecomponent"
	"github.com/keepdevops/cofiswarm-orchestrate/internal/bus"
	"github.com/keepdevops/cofiswarm-orchestrate/internal/orchestrate"
	"github.com/keepdevops/cofiswarm-orchestrate/pkg/manager"
)

func main() {
	host := flag.String("host", envOr("ORCH_SIDECAR_HOST", "127.0.0.1"), "listen host")
	port := flag.Int("port", envIntOr("ORCH_SIDECAR_PORT", 3003), "listen port")
	component := flag.String("component", envOr("ORCH_SIDECAR_COMPONENT", "orchestrate"),
		"bus identity: 'orchestrate' (.orchestrate.*) or 'infer-mlx' (.infer.mlx.*)")
	flag.Parse()

	busName, busRoutes := "orchestrate", bus.Routes()
	if *component == "infer-mlx" {
		busName, busRoutes = "infer-mlx", bus.InferMLXRoutes()
	}

	swarm, err := manager.New("").LoadSwarm()
	if err != nil {
		log.Fatalf("orch-sidecar: %v", err)
	}
	svc := orchestrate.NewService(swarm)
	defer svc.Close()
	log.Printf("orch-sidecar: loaded %d agents for orchestrate modes", len(swarm))

	mux := http.NewServeMux()
	svc.Register(mux)

	// Optional observer-bus presence (default-off via COFISWARM_NATS_URL).
	var comp *servicecomponent.Component
	if url := os.Getenv("COFISWARM_NATS_URL"); url != "" {
		nc, cErr := servicecomponent.Connect(url, "cofiswarm-"+busName)
		if cErr != nil {
			log.Printf("observer: NATS connect %s failed: %v (running without presence)", url, cErr)
		} else {
			defer nc.Close()
			comp = servicecomponent.New(nc, busName, busName, busRoutes)
			if sErr := comp.Start(); sErr != nil {
				log.Printf("observer: bus start: %v (running without presence)", sErr)
				comp = nil
			} else {
				log.Printf("observer: %s announced on %s", busName, url)
			}
		}
	}

	addr := *host + ":" + strconv.Itoa(*port)
	httpSrv := &http.Server{Addr: addr, Handler: cors(mux)}
	go func() {
		log.Printf("orch-sidecar: starting on %s", addr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("orch-sidecar: server error: %v", err)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	log.Printf("orch-sidecar: shutting down")
	if comp != nil {
		comp.Shutdown()
	}
	shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutCtx); err != nil {
		log.Printf("orch-sidecar: graceful shutdown: %v", err)
	}
}

// cors applies the same permissive policy as the Python sidecar's middleware.
func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", "*")
		h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		h.Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOr(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}
