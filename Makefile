ROLE := orchestrate
.PHONY: test test-standalone-layout test-import
test: test-standalone-layout test-import
test-standalone-layout:
	./test/scripts/assert-layout.sh $(ROLE)
test-import:
	PYTHONPATH=src python3 -c "from cofiswarm_orchestrate import manager; print('ok:', manager.__name__)"
