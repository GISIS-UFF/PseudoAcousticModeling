#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "Survey.hpp"
#include "Modeling.cuh"
#include "Migration.cuh"

class Inversion {
public:
    Inversion(Survey* parameters, Modeling* modeling, Migration* migration);

    Survey* pmt = nullptr;
    Modeling* mdl = nullptr;
    Migration* mgt = nullptr;

    int XBlocks = 0;

    float* X = nullptr;
    float* residual = nullptr;
    float* residual_buffer = nullptr;
    float* slowness2 = nullptr;
    float* past_field = nullptr;

    float* eps_grad = nullptr;
    float* delta_grad = nullptr;
    float* theta_grad = nullptr;

    float* X_h = nullptr;
    float* obs_h = nullptr;
    float* obs_buffer = nullptr;

    float* vp_h = nullptr;
    float* vpnew_h = nullptr;
    float* eps_h = nullptr;
    float* epsnew_h = nullptr;
    float* delta_h = nullptr;
    float* deltanew_h = nullptr;
    float* theta_h = nullptr;
    float* thetanew_h = nullptr;

    float* grad_vp_h = nullptr;
    float* grad_vpnew_h = nullptr;
    float* p_vp = nullptr;

    std::vector<double> grad;
    std::vector<double> grad_new;
    std::vector<double> p;

    float* B_vp = nullptr;
    float* B_eps = nullptr;
    float* B_delta = nullptr;
    std::vector<double> Q;

    float* dvp = nullptr;
    float* deps = nullptr;
    float* ddelta = nullptr;

    float* current_born = nullptr;
    float* future_born = nullptr;

    std::vector<double> dm;
    std::vector<double> x;
    std::vector<double> Hx;

    double beta1 = 0.0;
    double beta2 = 0.0;
    double beta2_new = 0.0;

    bool* water_mask = nullptr;
    bool* TGN = nullptr;

    std::vector<std::vector<float>> s_vp_store;
    std::vector<std::vector<float>> y_vp_store;

    void InitializeInversionFields();
    void freeMemory();
    float ObjectiveFunction();
    void solveFullWaveformInversionMonoparameter();
    void solveFullWaveformInversionMultiparameterHierarchical();
    void readObsSeismogram(int shot, float* obs);
    void resetGradients();
    void backward_step(int k, float* Pc, float* Pp, float* Pf, bool TGN);
    float calculateGradientOntheFly(bool TGN);
    float calculateGradientCheckpoint(bool TGN);
    float calculateGradient(const std::string& parameter, float* gradient);
    float calculateMultiparameterGradient(bool update_eps, bool update_delta, std::vector<double>& grad, bool TGN);
    void copyDeviceFloatToDouble(const float* device, double* host, int n);
    float dot(const float* a, const float* b);
    double dot_vector(const std::vector<double>& a, const std::vector<double>& b);
    double norm_vector(const std::vector<double>& a);
    void twoLoopRecursion(const float* gradient, float* p, const std::vector<std::vector<float>>& s_store, const std::vector<std::vector<float>>& y_store);
    void applyModelStep(const std::string& parameter, const float* model, const float* p, float* model_new, float alpha);
    void ExpandModelDevice(const std::string& parameter, const float* model);
    float Backtracking(const std::string& parameter, const float* model0, const float* grad0, const float* p, float X0, bool empty, float scale);
    float linesearch(const std::string& parameter, const float* model0, const float* grad0, const float* p, float X0, float& X_new, float* grad_new, bool empty, float scale);
    float zoom(const std::string& parameter, const float* model0, const float* grad0, const float* p, float X0, float gp0, float alpha_lo, float alpha_hi, float X_lo_initial, float& X_new, float* grad_new, float scale);
    float linesearchMultiparameter(const std::vector<double>& grad, const std::vector<double>& dm, bool update_eps, bool update_delta, float X0, float& X_new);
    void applyMultiparameterStep(bool update_eps, bool update_delta, const float* vp, const float* epsilon, const float* delta, float* vp_new, float* epsilon_new, float* delta_new, const std::vector<double>& dm, float alpha);
    void adjustfmax(float fmax);
    void saveModel(const std::string& file_name, const float* model);
    void saveModel(const std::string& file_name, const double* model); 
    void setModel();
    void scaleGradient(float* gradient, float scale);
    float getGradientScale(const float* gradient);
    void updateLBFGSHistory(const float* model, const float* model_new, const float* gradient, const float* gradient_new, std::vector<std::vector<float>>& s_store, std::vector<std::vector<float>>& y_store);
    void bornforward_step(int k);
    void computeHessianVectorProduct();
    void setTGN(float scale_q, float scale_eps, float scale_delta, bool update_eps, bool update_delta);
    void setBornPerturbation(const std::vector<float>& x);
    float computeEisenstatWalker(const std::vector<double>& grad, const std::vector<double>& grad_past, const std::vector<double>& Bdm_past, double alpha_past, double nk_past);
    void solveGaussNewton(std::vector<double>& g, double nk, double grad_norm, const double scales[3], bool update_eps, bool update_delta);
};

__global__ void computeObjectiveFunction(float* X, float* residual, const float* calculated, int nt, int Nrec);
__global__ void slowness2ToVp(const float* slowness2, float* vp, int nx, int nz);
__global__ void updateAdjointWaveEquationandGradient(float* Uf, float* Uc, const float* Pp, const float* Pc, const float* Pf, float* vp_grad, const float* vp, int nz, int nx, float dz, float dx, float dt, const float* A, int N_abc);
__global__ void calculateAdjointVTIProductsAndGradients(const float* Uc, const float* Pp, const float* Pc, const float* Pf, float* AUc, float* BUc, float* QCxUc, float* QCzUc, float* vp_grad, float* eps_grad, float* delta_grad, const float* vp, const float* epsilon, const float* delta, float dt, float dx, float dz, int nx, int nz, int N_abc, bool multiparameter, float* B_m, float* B_eps, float* B_delta, bool TGN);
__global__ void updateAdjointWaveEquationVTI(float* Uf, float* Uc, float* AUc, float* BUc, float* QCxUc, float* QCzUc, int nx, int nz, float dt, float dx, float dz, const float* vp, float* A, int N_abc);
__global__ void calculateAdjointTTIProductsAndGradients(const float* Uc, const float* Pp, const float* Pc, const float* Pf, float* AUc, float* BUc, float* HUc, float* QCxUc, float* QCzUc, float* vp_grad, float* eps_grad, float* delta_grad, float* theta_grad, const float* epsilon, const float* delta, const float* theta, float dt, float dx, float dz, int nx, int nz, int N_abc, bool multiparameter);
__global__ void updateAdjointWaveEquationTTI(float* Uf, float* Uc, const float* AUc, const float* BUc, const float* HUc, const float* QCxUc, const float* QCzUc, int nx, int nz, float dt, float dx, float dz, const float* vp, float* A, int N_abc);
__global__ void updateWaveEquationBornTGN(float* dUf, float* dUc, float* U0f, float* U0c, const float* vp, const float* dvp, int nz, int nx, float dz, float dx, float dt, float* A, int N_abc);
__global__ void updateWaveEquationVTIBornTGN(float* dUf, float* dUc, float* U0f, float* U0c, const float* vp, const float* epsilon, const float* delta, const float* dvp, const float* deps, const float* ddelta, int nz, int nx, float dz, float dx, float dt, float* A, int N_abc);
