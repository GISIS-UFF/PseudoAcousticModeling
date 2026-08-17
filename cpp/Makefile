NVCC=/usr/local/cuda-13.3/bin/nvcc

MODELING=run_modeling
MIGRATION=run_migration

COMMON=src/Modeling.cu src/Survey.cpp

MODELING_SRC=$(COMMON) src/modeling_main.cpp
MIGRATION_SRC=$(COMMON) src/Migration.cu src/migration_main.cpp

HEADERS=$(wildcard include/*.cuh include/*.hpp)

FLAGS=-std=c++17 -O3 -Iinclude -Xcompiler "-fopenmp -pthread"

.PHONY: all modeling migration run-modeling run-migration clean

all: $(MODELING) $(MIGRATION)

$(MODELING): $(MODELING_SRC) $(HEADERS)
	$(NVCC) $(FLAGS) $(MODELING_SRC) -o $(MODELING)

$(MIGRATION): $(MIGRATION_SRC) $(HEADERS)
	$(NVCC) $(FLAGS) $(MIGRATION_SRC) -o $(MIGRATION)

modeling: $(MODELING)

migration: $(MIGRATION)

run-modeling: $(MODELING)
	./$(MODELING)

run-migration: $(MIGRATION)
	./$(MIGRATION)

clean:
	rm -f $(MODELING) $(MIGRATION)