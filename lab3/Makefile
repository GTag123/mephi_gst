RESULT := main
OBJECTS := $(wildcard *.o)
BIN := $(wildcard *.bin)
TXT := $(wildcard *.txt)
compile:
	mpicc -o $(RESULT) compute.c

clean:
	rm -f $(OBJECTS) $(RESULT) $(BIN) $(TXT)