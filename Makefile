.PHONY: all build run test

build:
	@hpack
	@cabal build

run: build
	@cabal run example

test: build
	@cabal test