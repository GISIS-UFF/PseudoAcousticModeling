#pragma once
#include "Survey.hpp"
#include "Modeling.cuh"
#include "Migration.cuh"
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>

class Inversion{
    public:
    Survey* pmt;
    Modeling* mdl;
    Migration* mgt;

    Inversion(Survey* parameters, Modeling* modeling, Migration* migration);

    float* eps_grad = nullptr;
    float* delta_grad = nullptr;
    float* theta_grad = nullptr;

    float* X;
    float* slowness2 = nullptr;
    float* residual = nullptr;
    

};