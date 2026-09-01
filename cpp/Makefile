NVCC=/usr/local/cuda-13.3/bin/nvcc

MODELING=run_modeling
MIGRATION=run_migration
INVERSION=run_inversion

COMMON=Modeling.cu Survey.cpp

MODELING_SRC=$(COMMON) modeling_main.cpp
MIGRATION_SRC=$(COMMON) Migration.cu migration_main.cpp
INVERSION_SRC=$(COMMON) Migration.cu Inversion.cu inversion_main.cpp

HEADERS=$(wildcard include/*.cuh include/*.hpp)

FLAGS=-std=c++17 -O3 -Iinclude -I$(CONDA_PREFIX)/include -Xcompiler "-fopenmp -pthread"

.PHONY: all modeling migration inversion run-modeling run-migration run-inversion clean

all: $(MODELING) $(MIGRATION) $(INVERSION)

$(MODELING): $(MODELING_SRC) $(HEADERS)
	$(NVCC) $(FLAGS) $(MODELING_SRC) -o $(MODELING)

$(MIGRATION): $(MIGRATION_SRC) $(HEADERS)
	$(NVCC) $(FLAGS) $(MIGRATION_SRC) -o $(MIGRATION)

$(INVERSION): $(INVERSION_SRC) $(HEADERS)
	$(NVCC) $(FLAGS) $(INVERSION_SRC) -o $(INVERSION)

modeling: $(MODELING)

migration: $(MIGRATION)

inversion: $(INVERSION)

run-modeling: $(MODELING)
	./$(MODELING)

run-migration: $(MIGRATION)
	./$(MIGRATION)

run-inversion: $(INVERSION)
	./$(INVERSION)

clean:
	rm -f $(MODELING) $(MIGRATION) $(INVERSION)