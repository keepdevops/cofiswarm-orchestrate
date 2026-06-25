ROLE := orchestrate
.PHONY: test test-standalone-layout test-go build
test: test-standalone-layout test-go
test-standalone-layout:
	./test/scripts/assert-layout.sh $(ROLE)
# Go-only post-migration: pkg/manager + pkg/memory + internal/orchestrate
# (handlers, RAG bridge, e2e vs fake mlx_lm.server, the rag-inject-once guard).
test-go:
	go build ./... && go test ./...
build:
	CGO_ENABLED=0 go build -o bin/orch-sidecar ./cmd/orch-sidecar
