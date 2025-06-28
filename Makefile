.PHONY: all clean build

build:
	@hpack
	@cabal build

run: build
	@cabal run Main