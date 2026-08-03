#include <vector>
#include <cmath>
#include "Modeling.hpp"

const float pi = 3.14159265358979323846f;

Modeling::Modeling(Survey* parameters)
{
    pmt = parameters;
}

Modeling::~Modeling()
{
    delete[] source;

    delete[] vp;
    delete[] epsilon;
    delete[] delta;
    delete[] theta;

    delete[] vp_exp;
    delete[] epsilon_exp;
    delete[] delta_exp;
    delete[] theta_exp;

    delete[] current;
    delete[] future;

    delete[] seismogram;
}

void Modeling::Initializefields(){
    int n_model = pmt->nx * pmt->nz;
    int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    int n_seismogram = pmt->Nrec * pmt->nt;
    
    source = new float[pmt->nt]();

    vp = new float[n_model]();
    vp_exp = new float[n_model_exp]();

    current = new float[n_model_exp]();
    future = new float[n_model_exp]();

    seismogram = new float[n_seismogram]();

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI") {
        
        epsilon = new float[n_model]();
        delta = new float[n_model]();
        theta = new float[n_model]();

        epsilon_exp = new float[n_model_exp]();
        delta_exp = new float[n_model_exp]();
        theta_exp = new float[n_model_exp]();

    }

    if (pmt->approximation == "TTI") {

        theta = new float[n_model]();
        theta_exp = new float[n_model_exp]();
    }

    if (pmt->ABC == "CPML") {
        int nz_model_exp = pmt->nz_abc * (pmt->N_abc + 4);
        int nx_model_exp = pmt->nx_abc * (pmt->N_abc + 4);

        PsixFR = new float[nz_model_exp]();
        PsixFL = new float[nz_model_exp]();
        PsizFU = new float[nx_model_exp]();
        PsizFD = new float[nx_model_exp]();
        ZetaxFR = new float[nz_model_exp]();
        ZetaxFL = new float[nz_model_exp]();
        ZetazFU = new float[nx_model_exp]();
        ZetazFD = new float[nx_model_exp]();
    }
}

void Modeling::create_wavelet(){
    float tlag = pmt->tlag;
    float dt = pmt->dt;
    float fcut = pmt->fcut;

    float scale = 1.0f / (pmt->dx * pmt->dz);
    float fc = fcut / (3.0f * sqrtf(pi));
    for (int n = 0; n < pmt->nt; n++){
        float td = n*dt - tlag;

        float arg = pi*pi*pi*fc*fc*td*td;

        source[n] = (1.0f - 2.0f*arg)*expf(-arg)*scale;
    }

}