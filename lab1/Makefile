SOURCES := $(wildcard *.cpp)
RESULT := main

OBJECTS := $(SOURCES:.cpp=.o)
CFLAGS := -std=c++20 -fopenmp

all: run

run: compile
	./$(RESULT)

compile: $(SOURCES) $(RESULT)

%.o: %.cpp $(wildcard *.h)
	g++ -c $(CFLAGS) $< -o $@

$(RESULT): $(OBJECTS)
	g++ $(CFLAGS) $(OBJECTS) -o $(RESULT)

clean:
	rm -f $(OBJECTS) $(RESULT)
