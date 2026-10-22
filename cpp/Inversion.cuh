#pragma once

#include "Survey.hpp"
#include "Modeling.cuh"
#include "Migration.cuh"
#include <cuda_runtime.h>
#include <string>
#include <vector>

class Inversion {
public:
    Survey* pmt = nullptr;
    Modeling* mdl = nullptr;
    Migration* mgt = nullptr;

    int XBlocks = 0;

    float* X = nullptr;
    float* residual = nullptr;
    float* residual_buffer = nullptr;
    float* slowness2 = nullptr;

    float* X_h = nullptr;
    float* obs_h = nullptr;
    float* obs_buffer = nullptr;
    float* vp_h = nullptr;
    float* vpnew_h = nullptr;
    float* grad_vp_h = nullptr;
    float* grad_vpnew_h = nullptr;
    float* p_vp = nullptr;
    float* eps_h = nullptr;
    float* epsnew_h = nullptr;
    float* delta_h = nullptr;
    float* deltanew_h = nullptr;
    float* theta_h = nullptr;
    float* thetanew_h = nullptr;
    bool* water_mask = nullptr;

    float* eps_grad = nullptr;
    float* delta_grad = nullptr;
    float* theta_grad = nullptr;
    float* B_vp = nullptr;
    float* B_eps = nullptr;
    float* B_delta = nullptr;

    float* current_born = nullptr;
    float* future_born = nullptr;
    float* dvp = nullptr;   
    float* deps = nullptr;
    float* ddelta = nullptr;

    std::vector<std::vector<float>> s_vp_store;
    std::vector<std::vector<float>> y_vp_store;

    std::vector<float> grad;
    std::vector<float> Q;
    std::vector<float> p;
    std::vector<float> x;
    std::vector<float> Hx;
    std::vector<float> dm;
    double beta1 = 0.0;
    double beta2 = 0.0;
    double beta2_new = 0.0;

    Inversion(Survey* parameters, Modeling* modeling, Migration* migration);

    void InitializeInversionFields();
    void freeMemory();
    float ObjectiveFunction();
    void readObsSeismogram(const int shot, float* obs_h);
    void resetGradients();
    void backward_step(const float* P, const bool TGN);
    float calculateGradientOntheFly(bool TGN);
    float calculateGradientCheckpoint(bool TGN);
    float dot(const float* a, const float* b);
    double dot_vector(const std::vector<float>& a, const std::vector<float>& b);
    double norm_vector(const std::vector<float>& a);
    void twoLoopRecursion(const float* gradient, float* p, const std::vector<std::vector<float>>& s_store, const std::vector<std::vector<float>>& y_store);
    void applyModelStep(const std::string& parameter, const float* model, const float* p, float* model_new, const float alpha);
    void ExpandModelDevice(const std::string& parameter, const float* model);
    float calculateGradient(const std::string& parameter, float* gradient);
    float calculateMultiparameterGradient(const bool update_eps, const bool update_delta, std::vector<float>& grad, const bool TGN);
    float linesearch(const std::string& parameter, const float* model0, const float* grad0, const float* p, const float X0, float& X_new, float* grad_new, const bool empty, const float scale);
    float zoom(const std::string& parameter, const float* model0, const float* grad0, const float* p, const float X0, const float gp0, float alpha_lo, float alpha_hi, const float X_lo_initial, float& X_new, float* grad_new, const float scale);
    float linesearchMultiparameter(const std::vector<float>& grad, const std::vector<float>& dm, const bool update_eps, const bool update_delta, const float X0, float& X_new);
    void applyMultiparameterStep(const bool update_eps, const bool update_delta, const float* vp, const float* epsilon, const float* delta, float* vp_new, float* epsilon_new, float* delta_new, const std::vector<float>& dm, const float alpha);
    void adjustfmax(const float fmax);
    void saveModel(const std::string& file_name, const float* model);
    void setModel();
    float getGradientScale(const float* gradient);
    void scaleGradient(float* gradient, const float scale);
    void updateLBFGSHistory(const float* model, const float* model_new, const float* gradient, const float* gradient_new, std::vector<std::vector<float>>& s_store, std::vector<std::vector<float>>& y_store);
    void solveFullWaveformInversionMonoparameter();
    void saveSeismogramBorn(const int shot);
    void computeHessianVectorProduct();
    void setTGN(const bool update_eps, const bool update_delta);
    void setBornPerturbation(const std::vector<float>& x);
    float computeEisenstatWalker(const std::vector<float>& grad, const std::vector<float>& grad_past, const std::vector<float>& Bdm_past, const float alpha_past, const float nk_past);
    void solveGaussNewton(std::vector<float>& g, const float nk, const float grad_norm, const bool update_eps, const bool update_delta);
    void solveFullWaveformInversionMultiparameterHierarchical();
};

__global__ void computeObjectiveFunction(float* X, float* residual, const float* calculated, int nt, int Nrec);
__global__ void slowness2ToVp(const float* slowness2, float* vp, int nx, int nz);
__global__ void updateAdjointWaveEquationandGradient(float* Uf, float* Uc, const float* P, float* vp_grad, const float* vp, int nz, int nx, float dz, float dx, float dt, const float* A, int N_abc);
__global__ void calculateAdjointVTIProductsAndGradients(const float* Uc, const float* P, float* AUc, float* BUc, float* QCxUc, float* QCzUc, float* vp_grad, float* eps_grad, float* delta_grad, const float* vp, const float* epsilon, const float* delta, float dt, float dx, float dz, int nx, int nz, int N_abc, bool multiparameter, float* B_vp, float* B_eps, float* B_delta, bool TGN);
__global__ void updateAdjointWaveEquationVTI(float* Uf, float* Uc, float* AUc, float* BUc, float* QCxUc, float* QCzUc, int nx, int nz, float dt, float dx, float dz, const float* vp, float* A, int N_abc);
__global__ void calculateAdjointTTIProductsAndGradients(const float* Uc, const float* P, float* AUc, float* BUc, float* HUc, float* QCxUc, float* QCzUc, float* vp_grad, float* eps_grad, float* delta_grad, float* theta_grad, const float* vp, const float* epsilon, const float* delta, const float* theta, float dx, float dz, int nx, int nz, int N_abc, bool multiparameter);
__global__ void updateAdjointWaveEquationTTI(float* Uf, float* Uc, const float* AUc, const float* BUc, const float* HUc, const float* QCxUc, const float* QCzUc, int nx, int nz, float dt, float dx, float dz, const float* vp, float* A, int N_abc);
__global__ void updateWaveEquationVTIBornTGN(float* dUf, float* dUc, float* U0f, float* U0c, const float* vp, const float* epsilon, const float* delta, const float* dvp, const float* deps, const float* ddelta, int nz, int nx, float dz, float dx, float dt, float* A, int N_abc);

