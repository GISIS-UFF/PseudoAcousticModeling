#pragma once
#include "Survey.hpp"
#include "Modeling.cuh"
#include <cuda_runtime.h>

class Migration{
    public:
    Migration(Survey* parameters, Modeling* modeling)

    float* image = nullptr;
    float* ilum  = nullptr;
    float* currentbck = nullptr;
    float* futurebck = nullptr;
    float* savefield = nullptr;

    float* eps_grad = nullptr;
    float* delta_grad = nullptr;
    float* theta_grad = nullptr;   

    float* AUc = nullptr;
    float* BUc = nullptr;
    float* HUc = nullptr;
    float* QCxUc = nullptr;
    float* QCzUc = nullptr;

}