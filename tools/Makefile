
CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

SHARED=../shared

all: converter

converter: converter.cpp
	$(CC) converter.cpp -o converter $(CFLAGS)
			
clean:
	rm -f converter
