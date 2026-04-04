default: version doc

version:
	uv run build-metadata.py

doc:
	cd src; uv run ../build-doc.py

build:
	rm -rf dist
	uv build

testpublish: build
	uv publish --publish-url https://test.pypi.org/legacy
