ROLE := orchestrate
.PHONY: test test-standalone-layout
test: test-standalone-layout
test-standalone-layout:
	./test/scripts/assert-layout.sh $(ROLE)
