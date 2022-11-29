
CC=g++
NC=nvcc
CFLAGS=-std=c++14 -O3

#NFLAGS=-arch=sm_80

SHARED=../shared

all: converter_stdin

converter_stdin: converter_stdin.cpp
	$(CC) converter_stdin.cpp -o converter_stdin $(CFLAGS)
			
clean:
	rm -f converter_stdin
