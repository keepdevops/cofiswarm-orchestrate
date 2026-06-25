ROLE := orchestrate
.PHONY: test test-standalone-layout test-import test-orchestrate-gate test-rag-inject-gate
test: test-standalone-layout test-import test-orchestrate-gate test-rag-inject-gate
test-standalone-layout:
	./test/scripts/assert-layout.sh $(ROLE)
test-import:
	PYTHONPATH=src python3 -c "from cofiswarm_orchestrate import manager; print('ok:', manager.__name__)"
test-orchestrate-gate:
	./test/scripts/test-orchestrate-gate.sh
test-rag-inject-gate:
	./test/scripts/test-rag-inject-gate.sh
