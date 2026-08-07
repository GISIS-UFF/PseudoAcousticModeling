#!/bin/bash

set -e

nvcc -std=c++17 -O3 src/Modeling.cu src/Survey.cpp src/modeling_main.cpp -Iinclude -Xcompiler -fopenmp -o run_modeling

./run_modeling