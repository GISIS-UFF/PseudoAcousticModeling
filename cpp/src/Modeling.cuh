#pragma once
#include "Survey.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

# define nThreads 256

class Modeling 
{
public:
    Survey* pmt;

    Modeling(Survey* parameters);

    float* source = nullptr;
    float* A = nullptr;

    float* vp = nullptr;
    float* epsilon = nullptr;
    float* delta = nullptr;
    float* theta = nullptr;

    float* current = nullptr;
    float* future = nullptr;

    float* seismogram = nullptr;

    float* snapshot = nullptr;
    float* d_snapshot = nullptr;

    int* rx = nullptr;
    int* rz = nullptr;

    int sx = 0;
    int sz = 0;

    int expBlocks = 0;
    int BlocksSeis = 0;

    cudaStream_t copy_stream;
    cudaStream_t compute_stream;

    void freeMemory();
    void initializeFields();
    void createWavelet();
    void createCerjanVector();
    void importBin(std::string path, float* array, int n);
    void expandModel(float* model, float* output);
    void resetFields();
    void checkDispersionAndStability(const float* vp_h, const float* epsilon_h, const float* delta_h, const float* theta_h);
    void setModel();
    void saveSnapshot(const int shot,const int k);
    void saveSeismogram(const int shot);
    void forward_step(const int k);
    void solveWaveEquation(); 
};

__global__ void injectSource(float* __restrict__ current,const float* __restrict__ source,int k,const int nt,const int nx_abc,const int sx,const int sz);
__global__ void storeSeismogram(const float* current, float* seismogram, const int* rx, const int* rz, int k, int itlag, int Nrec, int nx_abc);
__global__ void updateWaveEquation(float* __restrict__ Uf, float* __restrict__ Uc,const float* __restrict__ vp,const int nz,const int nx,const float dz,const float dx,const float dt, float* __restrict__ A, int N_abc);
__global__ void updateWaveEquationVTI(float* __restrict__ Uf, float* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta, float* __restrict__ A, int N_abc );
__global__ void updateWaveEquationTTI(float* __restrict__ Uf, float* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta,const float* __restrict__ theta,  float* __restrict__ A, int N_abc);
