#pragma once

#include "Survey.hpp"
#include "Modeling.cuh"
#include "Migration.cuh"

#include <cuda_runtime.h>
#include <string>
#include <vector>

class Inversion {
public:
    Survey *pmt = nullptr;
    Modeling *mdl = nullptr;
    Migration *mgt = nullptr;

    int XBlocks = 0;

    // Campos da funcao objetivo e do modelo em GPU
    float *X = nullptr;
    float *residual = nullptr;
    float *residual_buffer = nullptr;
    float *s2 = nullptr;

    // Campos auxiliares no host
    float *X_h = nullptr;
    float *obs_h = nullptr;
    float *obs_buffer = nullptr;
    float *s2_h = nullptr; 
    float *s2new_h = nullptr;
    float *grad_s2_h = nullptr;
    float *grad_s2new_h = nullptr;
    float *p_s2 = nullptr;
    float *eps_h = nullptr;
    float *epsnew_h = nullptr;
    float *delta_h = nullptr;
    float *deltanew_h = nullptr;
    float *theta_h = nullptr;
    float *thetanew_h = nullptr;
    bool *water_mask = nullptr;

    // Gradientes e pseudo-Hessiana na GPU
    float *eps_grad = nullptr;
    float *delta_grad = nullptr;
    float *theta_grad = nullptr;
    float *B_s2 = nullptr;
    float *B_eps = nullptr;
    float *B_delta = nullptr;

    // Campos de Born e perturbacoes fisicas na GPU
    float *current_born = nullptr;
    float *future_born = nullptr;
    float *ds2 = nullptr; // Perturbacao fisica de s2 = 1/vp^2
    float *deps = nullptr;
    float *ddelta = nullptr;

    // L-BFGS monoparametrico
    std::vector<std::vector<float>> s_s2_store;
    std::vector<std::vector<float>> y_s2_store;

    // Gauss-Newton truncado / CG
    std::vector<float> grads;
    std::vector<float> models;
    std::vector<float> Q;
    std::vector<float> p;
    std::vector<float> x;
    std::vector<float> Hx;
    std::vector<float> dm;
    double beta1 = 0.0;
    double beta2 = 0.0;
    double beta2_new = 0.0;

    Inversion(Survey *parameters, Modeling *modeling, Migration *migration);
    void InitializeInversionFields();
    void freeMemory();
    float ObjectiveFunction();
    void readObsSeismogram(const int shot, float *obs_h);
    void resetGradients();
    void backward_step(const float *P, const bool TGN);
    float calculateGradientOntheFly(bool TGN = false);
    float calculateGradientCheckpoint(bool TGN = false);
    float dot(const float *a, const float *b);
    double dot_vector(const std::vector<float> &a, const std::vector<float> &b);
    double norm_vector(const std::vector<float> &a);
    void twoLoopRecursion(const float *gradient, float *p, const std::vector<std::vector<float>> &s_store, const std::vector<std::vector<float>> &y_store);
    void applyModelStep(const std::string &parameter, const float *model, const float *p, float *model_new, const float alpha);
    void ExpandModelDevice(const std::string &parameter, const float *model);
    void uploadPhysicalModels(const std::vector<float> &physical);
    float calculateGradient(const std::string &parameter, float *gradient);
    float calculateMultiparameterGradient(const bool update_eps, const bool update_delta, std::vector<float> &grads, std::vector<float> &models, const bool TGN);
    float linesearch(const std::string &parameter, const float *model0, const float *grad0, const float *p, const float X0, float &X_new, float *grad_new, const bool empty, const float scale);
    float zoom(const std::string &parameter, const float *model0, const float *grad0, const float *p, const float X0, const float gp0, float alpha_lo, float alpha_hi, const float X_lo_initial, float &X_new, float *grad_new, const float scale);
    float linesearchMultiparameter(const std::vector<float> &models, const std::vector<float> &grads, const std::vector<float> &dm, const bool update_eps, const bool update_delta, const float X0, float &X_new);
    void applyMultiparameterStep(const bool update_eps, const bool update_delta, std::vector<float> &models, const std::vector<float> &dm, const float alpha);
    void adjustfmax(const float fmax);
    void saveModel(const std::string &file_name, const float *model);
    void setModel();
    float getGradientScale(const float *gradient);
    void scaleGradient(float *gradient, const float scale);
    void updateLBFGSHistory(const float *model, const float *model_new, const float *gradient,const float *gradient_new, std::vector<std::vector<float>> &s_store, std::vector<std::vector<float>> &y_store);
    void solveFullWaveformInversionMonoparameter();
    void saveSeismogramBorn(const int shot);
    void computeHessianVectorProduct();
    void setTGN(const bool update_eps, const bool update_delta);
    void setBornPerturbation(const std::vector<float> &x);
    float computeEisenstatWalker(const std::vector<float> &grads, const std::vector<float> &grad_past, const std::vector<float> &Bdm_past, const float alpha_past, const float nk_past);
    void solveGaussNewton(std::vector<float> &g, const float nk, const float grad_norm, const bool update_eps, const bool update_delta);
    void convertNormalizedAndPhysical(const bool to_physical, std::vector<float> &models);
    void solveFullWaveformInversionMultiparameterHierarchical();
    
    //Validação
    void validategradient(const float h);
    void validateborn(const std::vector<float>& direction, const float h);
    void validateHessian(const std::vector<float>& direction, const float h);
};

__global__ void computeObjectiveFunction(float *X, float *__restrict__ residual, const float *__restrict__ calculated, const int nt, const int Nrec);
__global__ void s2ToVp(const float *__restrict__ s2, float *__restrict__ vp, int nx, int nz);
__global__ void updateAdjointWaveEquationandGradient(float *__restrict__ Uf, float *__restrict__ Uc, const float *__restrict__ P, float *__restrict__ s2_grad, const float *__restrict__ vp, const int nz, const int nx,
const float dz, const float dx, const float dt, const float *__restrict__ A, const int N_abc);
__global__ void calculateAdjointVTIProductsAndGradients(const float *__restrict__ Uc, const float *__restrict__ P, float *__restrict__ AUc, float *__restrict__ BUc, float *__restrict__ QCxUc, float *__restrict__ QCzUc,
float *__restrict__ s2_grad, float *__restrict__ eps_grad, float *__restrict__ delta_grad, const float *__restrict__ vp, const float *__restrict__ epsilon, const float *__restrict__ delta, const float dt, const float dx, const float dz, const int nx,
const int nz, const int N_abc, const bool multiparameter, float *__restrict__ B_s2, float *__restrict__ B_eps, float *__restrict__ B_delta, const bool TGN);
__global__ void updateAdjointWaveEquationVTI(float *__restrict__ Uf, float *__restrict__ Uc, float *__restrict__ AUc, float *__restrict__ BUc, float *__restrict__ QCxUc, float *__restrict__ QCzUc,
const int nx, const int nz, const float dt, const float dx, const float dz, const float *__restrict__ vp, float *__restrict__ A, int N_abc);
__global__ void calculateAdjointTTIProductsAndGradients(const float *__restrict__ Uc, const float *__restrict__ P, float *__restrict__ AUc, float *__restrict__ BUc, float *__restrict__ HUc, float *__restrict__ QCxUc,
float *__restrict__ QCzUc, float *__restrict__ s2_grad, float *__restrict__ eps_grad, float *__restrict__ delta_grad, float *__restrict__ theta_grad, const float *__restrict__ vp, const float *__restrict__ epsilon, const float *__restrict__ delta,
const float *__restrict__ theta, const float dx, const float dz, const int nx, const int nz, const int N_abc, const bool multiparameter);
__global__ void updateAdjointWaveEquationTTI(float *__restrict__ Uf, float *__restrict__ Uc, const float *__restrict__ AUc, const float *__restrict__ BUc, const float *__restrict__ HUc, const float *__restrict__ QCxUc,
const float *__restrict__ QCzUc, const int nx, const int nz, const float dt, const float dx, const float dz, const float *__restrict__ vp, float *__restrict__ A, int N_abc);
__global__ void updateWaveEquationVTIBornTGN(float *__restrict__ dUf, float *__restrict__ dUc, float *__restrict__ U0f, float *__restrict__ U0c, const float *__restrict__ vp, const float *__restrict__ epsilon,
const float *__restrict__ delta, const float *__restrict__ ds2, const float *__restrict__ deps, const float *__restrict__ ddelta, int nz, int nx, float dz, float dx, float dt, float *__restrict__ A, int N_abc);

