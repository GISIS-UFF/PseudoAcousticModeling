#pragma once
#include "Survey.hpp"
#include "Modeling.cuh"
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>
#include <vector>

const float pi = 3.14159265358979323846f;

class Migration{
public:
    Survey* pmt;
    Modeling* mdl;

    Migration(Survey* parameters, Modeling* modeling);
    float *image = nullptr;
    float *ilum = nullptr;
    float *currentbck = nullptr;
    float *futurebck = nullptr;
    double *savefield = nullptr;
    double *d_current = nullptr;
    double *d_future = nullptr;
    double *h_current = nullptr;
    double *h_future = nullptr;
    double *h_current_next = nullptr;
    double *h_future_next = nullptr;
    float *AUc = nullptr;
    float *BUc = nullptr;
    float *HUc = nullptr;
    float *QCxUc = nullptr;
    float *QCzUc = nullptr;
    int expBlocks;
    int seisBlocks;
    int nBlocks;

    void initializeMigrationFields();
    void freeMemory();
    void resetFields();
    void Mute(double* seismogram, const int shot);
    void loadSeismogram(const int shot);
    bool* createMask(const float* vp);
    float gaussianKernel(const int x);
    std::vector<float> gaussianFilter1D();
    void smoothModel(float* f, const bool* mask, const bool parameter = false);
    void backward_step(const double* P);
    void setModel();
    void saveCheckpoint(const int k);
    void importCheckpoint(const int k, double* current, double* future);
    void saveImage();
    void solveReverseTimeMigrationOntheFly();
    void solveReverseTimeMigrationCheckpoint();
};

__global__ void normalizeImage(float* __restrict__ image, const float* __restrict__ ilum, int nx, int nz);
__global__ void removeSource(double* __restrict__ future, const float* __restrict__ source, int k, const int nt, const int nx_abc, const int sx, const int sz, float dx, float dz, float dt);
__global__ void injectAdjointSource(float* __restrict__ futurebck, const double* __restrict__ seismogram, const int* rx, const int* rz, int t, int Nrec, int nx_abc, float dt, const float* __restrict__ vp);
__global__ void updateAdjointWaveEquation(float* __restrict__ Uf, float* __restrict__ Uc, const double* __restrict__ P, float* __restrict__ image, float* __restrict__ ilum, const float* __restrict__ vp,const int nz,const int nx,const float dz,const float dx,const float dt, float* __restrict__ A, int N_abc);
__global__ void calculateAdjointVTIProducts(const float* __restrict__ Uc, const double* __restrict__ P, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc, const int nx, const int nz, const float dx, const float dz, const float* __restrict__ epsilon, const float* __restrict__ delta);
__global__ void updateAdjointWaveEquationVTI(float* __restrict__ Uf, float* __restrict__ Uc, const double* __restrict__ P, float* __restrict__ image, float* __restrict__ ilum, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp, float* __restrict__ A, int N_abc);
__global__ void calculateAdjointTTIProducts(const float* __restrict__ Uc, const double* __restrict__ P, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ HUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc, const int nx, const int nz, const float dx, const float dz, const float* __restrict__ epsilon, const float* __restrict__ delta, const float* __restrict__ theta);
__global__ void updateAdjointWaveEquationTTI(float* __restrict__ Uf, float* __restrict__ Uc, const double* __restrict__ P, float* __restrict__ image, float* __restrict__ ilum, const float* __restrict__ AUc, const float* __restrict__ BUc, const float* __restrict__ HUc, const float* __restrict__ QCxUc, const float* __restrict__ QCzUc, const int nx, const int nz, const float dt, const float dx, const float dz, const float* __restrict__ vp, float* __restrict__ A, int N_abc);

