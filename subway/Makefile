
CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

SHARED=../shared

all: bfs-sync.o cc-sync.o sssp-sync.o sswp-sync.o pr-sync.o bfs-async.o cc-async.o sssp-async.o sswp-async.o pr-async.o


bfs-sync.o: bfs-sync.cu
	$(NC) -c bfs-sync.cu $(CFLAGS) $(NFLAGS)
	
cc-sync.o: cc-sync.cu
	$(NC) -c cc-sync.cu $(CFLAGS) $(NFLAGS)	

sssp-sync.o: sssp-sync.cu
	$(NC) -c sssp-sync.cu $(CFLAGS) $(NFLAGS)	

sswp-sync.o: sswp-sync.cu
	$(NC) -c sswp-sync.cu $(CFLAGS) $(NFLAGS)
	
pr-sync.o: pr-sync.cu
	$(NC) -c pr-sync.cu $(CFLAGS) $(NFLAGS)			
	
bfs-async.o: bfs-async.cu
	$(NC) -c bfs-async.cu $(CFLAGS) $(NFLAGS)		

cc-async.o: cc-async.cu
	$(NC) -c cc-async.cu $(CFLAGS) $(NFLAGS)	
		
sssp-async.o: sssp-async.cu
	$(NC) -c sssp-async.cu $(CFLAGS) $(NFLAGS)		

sswp-async.o: sswp-async.cu
	$(NC) -c sswp-async.cu $(CFLAGS) $(NFLAGS)	
	
pr-async.o: pr-async.cu
	$(NC) -c pr-async.cu $(CFLAGS) $(NFLAGS)		
			
clean:
	rm  *.o 
