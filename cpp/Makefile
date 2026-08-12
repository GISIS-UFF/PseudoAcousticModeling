NVCC = /usr/local/cuda-13.3/bin/nvcc
TARGET = run_modeling

SOURCES = src/Modeling.cu src/Survey.cpp src/modeling_main.cpp
HEADERS = $(wildcard include/*.cuh include/*.hpp)
FLAGS = -std=c++17 -O3 -Iinclude -Xcompiler -fopenmp

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(FLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)