

CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

SHARED=shared
SUBWAY=subway
TOOLS=tools

DEP=$(SHARED)/timer.o $(SHARED)/argument_parsing.o $(SHARED)/graph.o $(SHARED)/subgraph.o $(SHARED)/partitioner.o $(SHARED)/subgraph_generator.o $(SHARED)/gpu_kernels.o $(SHARED)/subway_utilities.o $(SHARED)/test.o  

all: make1 make2 make3 bfs-sync cc-sync sssp-sync sswp-sync pr-sync bfs-async cc-async sssp-async sswp-async pr-async

make1:
	make -C $(SHARED)

make2:
	make -C $(SUBWAY)

make3:
	make -C $(TOOLS)


bfs-sync: $(SUBWAY)/bfs-sync.o $(DEP)
	$(NC) $(SUBWAY)/bfs-sync.o $(DEP) -o bfs-sync $(CFLAGS) $(NFLAGS)
	
cc-sync: $(SUBWAY)/cc-sync.o $(DEP)
	$(NC) $(SUBWAY)/cc-sync.o $(DEP) -o cc-sync $(CFLAGS) $(NFLAGS)

sssp-sync: $(SUBWAY)/sssp-sync.o $(DEP)
	$(NC) $(SUBWAY)/sssp-sync.o $(DEP) -o sssp-sync $(CFLAGS) $(NFLAGS)

sswp-sync: $(SUBWAY)/sswp-sync.o $(DEP)
	$(NC) $(SUBWAY)/sswp-sync.o $(DEP) -o sswp-sync $(CFLAGS) $(NFLAGS)
	
pr-sync: $(SUBWAY)/pr-sync.o $(DEP)
	$(NC) $(SUBWAY)/pr-sync.o $(DEP) -o pr-sync $(CFLAGS) $(NFLAGS)	
	
bfs-async: $(SUBWAY)/bfs-async.o $(DEP)
	$(NC) $(SUBWAY)/bfs-async.o $(DEP) -o bfs-async $(CFLAGS) $(NFLAGS)	
	
cc-async: $(SUBWAY)/cc-async.o $(DEP)
	$(NC) $(SUBWAY)/cc-async.o $(DEP) -o cc-async $(CFLAGS) $(NFLAGS)		
	
sssp-async: $(SUBWAY)/sssp-async.o $(DEP)
	$(NC) $(SUBWAY)/sssp-async.o $(DEP) -o sssp-async $(CFLAGS) $(NFLAGS)	

sswp-async: $(SUBWAY)/sswp-async.o $(DEP)
	$(NC) $(SUBWAY)/sswp-async.o $(DEP) -o sswp-async $(CFLAGS) $(NFLAGS)	
	
pr-async: $(SUBWAY)/pr-async.o $(DEP)
	$(NC) $(SUBWAY)/pr-async.o $(DEP) -o pr-async $(CFLAGS) $(NFLAGS)
	
clean:
	make -C $(SHARED) clean
	make -C $(SUBWAY) clean
	make -C $(TOOLS) clean
	rm -f bfs-sync cc-sync sssp-sync sswp-sync pr-sync bfs-async cc-async sssp-async sswp-async pr-async
