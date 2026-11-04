#pragma once
#include "Survey.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#define nThreads 256

class Modeling {
public:
    Survey* pmt;
    Modeling(Survey* parameters);

    int *rx = nullptr;
    int *rz = nullptr;
    float *source = nullptr;
    float *A = nullptr;
    float *vp = nullptr;
    float *epsilon = nullptr;
    float *delta = nullptr;
    float *theta = nullptr;
    double *current = nullptr;
    double *future = nullptr;
    double  *seismogram = nullptr;
    float *snapshot = nullptr;
    float *d_snapshot = nullptr;
    int expBlocks;
    int seisBlocks;
    int sx; 
    int sz;
    
    cudaStream_t copy_stream; 
    cudaStream_t compute_stream;
    
    void freeMemory();
    void initializeFields();
    void createWavelet();
    void importBin(std::string path, float* array, int n);
    void createCerjanVector();
    void resetFields();
    void expandModel(const float* __restrict__ model, float* __restrict__ output);
    void checkDispersionAndStability(const float* __restrict__ vp_h, const float* __restrict__ epsilon_h, const float* __restrict__ delta_h, const float* __restrict__ theta_h);
    void setModel();
    void saveSnapshot(const int shot,const int k);
    void saveSeismogram(const int shot);
    void forward_step(bool save_snapshot = false);
    void solveWaveEquation();
};

__global__ void injectSource(double* future, const float* source, int k, const int nt, const int nx_abc, const int sx, const int sz, const float dx, const float dz, const float dt);
__global__ void storeSeismogram(const double* current, double* seismogram, const int* rx, const int* rz, int k, int itlag, int Nrec, int nx_abc);
__global__ void updateWaveEquation(double* __restrict__ Uf, const double* __restrict__ Uc,const float* __restrict__ vp,const int nz,const int nx,const float dz,const float dx,const float dt, float* __restrict__ A, int N_abc, float* snapshot = nullptr, bool save_snapshot = false);
__global__ void updateWaveEquationVTI(double* __restrict__ Uf, const double* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta, float* __restrict__ A, int N_abc, float* snapshot = nullptr, bool save_snapshot = false);
__global__ void updateWaveEquationTTI(double* __restrict__ Uf, const double* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta,const float* __restrict__ theta, float* __restrict__ A, int N_abc, float* snapshot = nullptr, bool save_snapshot = false);
