#pragma once

#include <string>
#include <vector>
#include "Survey.hpp"

class Modeling 
{
public:
    Survey* pmt;

    Modeling(Survey* parameters);
    ~Modeling();

    float* source;

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

    // CPML
    float* PsixFR = nullptr;
    float* PsixFL = nullptr;
    float* PsizFU = nullptr;
    float* PsizFD = nullptr;
    float* ZetaxFR = nullptr;
    float* ZetaxFL = nullptr;
    float* ZetazFU = nullptr;
    float* ZetazFD = nullptr;

    void Initializefields();
    void create_wavelet();
    

}