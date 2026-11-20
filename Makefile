CXX ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter

.PHONY: all clean test

all: build/nanollm

build:
	mkdir -p build

build/nanollm: src/nanollm.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf build

test: build/nanollm
	bash tests/golden.sh
