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

    double* X = nullptr;
    float* residual = nullptr;
    float* residual_buffer = nullptr;
    float* slowness2 = nullptr;
    float* past_field = nullptr;

    float* eps_grad = nullptr;
    float* delta_grad = nullptr;
    float* theta_grad = nullptr;

    double* X_h = nullptr;
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
    float* grad_eps_h = nullptr;
    float* grad_epsnew_h = nullptr;
    float* p_eps = nullptr;
    float* grad_delta_h = nullptr;
    float* grad_deltanew_h = nullptr;
    float* p_delta = nullptr;
    float* grad_theta_h = nullptr;
    float* grad_thetanew_h = nullptr;
    float* p_theta = nullptr;

    bool* water_mask = nullptr;

    std::vector<std::vector<float>> s_vp_store;
    std::vector<std::vector<float>> y_vp_store;
    std::vector<std::vector<float>> s_eps_store;
    std::vector<std::vector<float>> y_eps_store;
    std::vector<std::vector<float>> s_delta_store;
    std::vector<std::vector<float>> y_delta_store;
    std::vector<std::vector<float>> s_theta_store;
    std::vector<std::vector<float>> y_theta_store;

    void InitializeInversionFields();
    void freeMemory();
    double ObjectiveFunction();
    void solveFullWaveformInversionMonoparameter();
    void solveFullWaveformInversionMultiparameterHierarchical();
    void readObsSeismogram(int shot, float* obs);
    void resetGradients();
    void backward_step(int k, float* Pc, float* Pp, float* Pf);
    double calculateGradientOntheFly();
    double calculateGradientCheckpoint();
    double calculateGradient(const std::string& parameter, float* gradient);
    double calculateMultiparameterGradient(bool update_eps, bool update_delta, bool update_theta, float* gradient_vp, float* gradient_eps, float* gradient_delta, float* gradient_theta);
    double dot(const float* a, const float* b);
    void twoLoopRecursion(const float* gradient, float* p, const std::vector<std::vector<float>>& s_store, const std::vector<std::vector<float>>& y_store);
    void applyModelStep(const std::string& parameter, const float* model, const float* p, float* model_new, float alpha);
    void ExpandModelDevice(const std::string& parameter, const float* model);
    float armijolinesearch(const std::string& parameter, const float* model0, const float* grad0, const float* p, double X0, bool empty, float scale);
    float linesearch(const std::string& parameter, const float* model0, const float* grad0, const float* p, double X0, double& X_new, float* grad_new, bool empty, float scale);
    float zoom(const std::string& parameter, const float* model0, const float* grad0, const float* p, double X0, double gp0, float alpha_lo, float alpha_hi, double X_lo_initial, double& X_new, float* grad_new, float scale);
    float linesearchMultiparameter(const float* vp, const float* epsilon, const float* delta, const float* theta, const float* p_vp, const float* p_eps, const float* p_delta, const float* p_theta, const float* g_vp, const float* g_eps, const float* g_delta, const float* g_theta, float beta_vp, float beta_eps, float beta_delta, float beta_theta, float scale_vp, float scale_eps, float scale_delta, float scale_theta, bool update_eps, bool update_delta, bool update_theta, double X0);
    void applyMultiparameterStep(bool update_eps, bool update_delta, bool update_theta, const float* vp, const float* epsilon, const float* delta, const float* theta, const float* p_vp, const float* p_eps, const float* p_delta, const float* p_theta, float* vp_new, float* epsilon_new, float* delta_new, float* theta_new, float alpha, float beta_vp, float beta_eps, float beta_delta, float beta_theta);
    void adjustfmax(float fmax);
    void saveModel(const std::string& file_name, const float* model);
    void setModel();
    void scaleGradient(float* gradient, float scale);
    float getGradientScale(const float* gradient);
    void updateLBFGSHistory(const float* model, const float* model_new, const float* gradient, const float* gradient_new, std::vector<std::vector<float>>& s_store, std::vector<std::vector<float>>& y_store);
};

__global__ void computeObjectiveFunction(double* X, float* residual, const float* calculated, int nt, int Nrec);
__global__ void slowness2ToVp(const float* slowness2, float* vp, int nx, int nz);
__global__ void updateAdjointWaveEquationandGradient(float* Uf, float* Uc, const float* Pp, const float* Pc, const float* Pf, float* ilum, float* vp_grad, const float* vp, int nz, int nx, float dz, float dx, float dt, const float* A, int N_abc);
__global__ void calculateAdjointVTIProductsAndGradients(const float* Uc, const float* Pp, const float* Pc, const float* Pf, float* ilum, float* AUc, float* BUc, float* QCxUc, float* QCzUc, float* vp_grad, float* eps_grad, float* delta_grad, const float* epsilon, const float* delta, float dt, float dx, float dz, int nx, int nz, int N_abc, bool multiparameter);
__global__ void updateAdjointWaveEquationVTI(float* Uf, float* Uc, float* AUc, float* BUc, float* QCxUc, float* QCzUc, int nx, int nz, float dt, float dx, float dz, const float* vp, float* A, int N_abc);
__global__ void calculateAdjointTTIProductsAndGradients(const float* Uc, const float* Pp, const float* Pc, const float* Pf, float* ilum, float* AUc, float* BUc, float* HUc, float* QCxUc, float* QCzUc, float* vp_grad, float* eps_grad, float* delta_grad, float* theta_grad, const float* epsilon, const float* delta, const float* theta, float dt, float dx, float dz, int nx, int nz, int N_abc, bool multiparameter);
__global__ void updateAdjointWaveEquationTTI(float* Uf, float* Uc, const float* AUc, const float* BUc, const float* HUc, const float* QCxUc, const float* QCzUc, int nx, int nz, float dt, float dx, float dz, const float* vp, float* A, int N_abc);
