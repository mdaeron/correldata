all: doc
	@echo "All done!"

doc:
	@echo "Building documentation..."
	@cd src; uv run python ../build_doc.py
