# Physics-Aware ILU Preconditioner Makefile
# Author: Aditya Kumar

# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -std=c++11 -fopenmp
TARGET := main

# Default target
all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

# Run with a simple test (if you have a data file)
run: $(TARGET)
	./$(TARGET) data/sample_matrix.mtx

# Clean up
clean:
	rm -f $(TARGET)

.PHONY: all run clean
