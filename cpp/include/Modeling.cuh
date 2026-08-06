#pragma once
#include "Survey.hpp"
# include <cuda_runtime.h>
# include <curand_kernel.h>
#include <fstream>
#include <iostream>

# define nThreads 256

class Modeling 
{
public:
    Survey* pmt;

    Modeling(Survey* parameters);
    float* source;
    float* A;

    // Modelos
    float* vp = nullptr;
    float* epsilon = nullptr;
    float* delta = nullptr;
    float* theta = nullptr;

    // Modelos expandidos
    float* vp_exp = nullptr;
    float* epsilon_exp = nullptr;
    float* delta_exp = nullptr;
    float* theta_exp = nullptr;

    // Campos 
    float* current = nullptr;
    float* future = nullptr;

    // Sismograma
    float* seismogram = nullptr;
    float* seismograms = nullptr;

    void Initializefields();
    void create_wavelet();
    void importBin(std::string path, float* array, int n);
    void exportBin(std::string path, float* array, int n);
    void expandModel(float* model, float* output);
    void reduceModel(const float* model_exp, float* output);

    

}