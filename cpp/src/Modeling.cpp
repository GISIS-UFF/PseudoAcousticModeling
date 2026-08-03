#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
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

void Modeling::initializeFields(){
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

void Modeling::createWavelet(){
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

void Modeling::importBin(std::string path, float* array, int n){
    std::ifstream file(path, std::ios::in);
    if (!file.is_open()){
        throw std::invalid_argument("Could not open file. Please verify the file path.");
    }

    file.read((char *) array, n * sizeof(float));
    file.close();
}

void Modeling::exportBin(std::string path, float* array, int n){
    std::ofstream file(path, std::ios::out);
    if (!file.is_open()){
        throw std::invalid_argument("Could not open file. Please verify the file path.");
    }
    file.write((char*) array, n * sizeof(float));
    std::cout<<"File saved to the path" + path <<std::endl;
    file.close();
}

void Modeling::expandModel(float* model, float* output){

    int N_abc = pmt->N_abc;
    int nx = pmt->nx;
    int nz = pmt->nz;
    int nx_abc = pmt->nx_abc;
    int nz_abc = pmt->nz_abc;
    int index;

    // // Centro
    // for (int i = N_abc; i < nx + N_abc; i++){
    //     for (int j = N_abc; j < nz + N_abc, j++){
    //         index = (j - N_abc)*nx + (i - N_abc);
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }
    // // Esquerda
    // for (int i = 0; i < N_abc; i++){
    //     for (int j = N_abc; j < nz + N_abc; j++){
    //         index = (j - N_abc)*nx;
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }
    // // Direita
    // for (int i = nx + N_abc; i < nx_abc; i++){
    //     for (int j = N_abc; j < nz + N_abc; j++){
    //         index = (j - N_abc)*nx + (i - N_abc);
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }

    // for (int i = N_abc; i < nx + N_abc; i++){
    //     for (int j = 0; j < N_abc; j++){
    //         index = (j + N_abc)*nx + i;
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }

    // for (int i = N_abc; i < nx + N_abc; i++){
    //     for (int j = nz + N_abc; j < nz_abc; j++){
    //         index = (j - N_abc)*nx + i;
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }

    // for (int i = 0; i < N_abc; i++){
    //     for (int j = 0; j < N_abc; j++){
    //         index = (j + N_abc)*nx + (i + N_abc);
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }

    // for (int i = nx + N_abc; i < nx_abc; i++){
    //     for (int j = 0; j < N_abc; j++){
    //         index = (j + N_abc)*nx + (i - N_abc);
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }

    // for (int i = 0; i < N_abc; i++){
    //     for (int j = nz + N_abc; j < nz_abc; j++){
    //         index = (j - N_abc)*nx + (i + N_abc);
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }

    // for (int i = nx + N_abc; i < nx_abc; i++){
    //     for (int j = nz + N_abc; j < nz_abc; j++){
    //         index = (j - N_abc)*nx + (i - N_abc);
    //         output[j*nx_abc + i] = model[index];
    //     }
    // }

}
