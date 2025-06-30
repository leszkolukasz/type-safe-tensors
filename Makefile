.PHONY: all build run test

build:
	@hpack
	@cabal build

run: build
	@cabal run mnist

test: build
	@cabal test