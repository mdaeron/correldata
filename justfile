default: version

version:
	uv run update-metadata.py

build:
	rm -rf dist
	uv build

testpublish: build
	uv publish --publish-url https://test.pypi.org/legacy
