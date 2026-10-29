#include "Inversion.cuh"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>
#include <numeric>

Inversion::Inversion(Survey *parameters, Modeling *modeling, Migration *migration) {
    pmt = parameters;
    mdl = modeling;
    mgt = migration;
}

void Inversion::InitializeInversionFields() {
    const int n_model = pmt->nx * pmt->nz;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec * pmt->nt_data;

    XBlocks = (n_seis + nThreads - 1) / nThreads;

    cudaMalloc((void **)&X, sizeof(float));
    cudaMallocHost((void **)&X_h, sizeof(float));
    cudaMallocHost((void **)&obs_h, n_seis * sizeof(float));
    cudaMallocHost((void **)&obs_buffer, n_seis * sizeof(float));
    cudaMallocHost((void **)&s2_h, n_model * sizeof(float));
    cudaMallocHost((void **)&s2new_h, n_model * sizeof(float));
    cudaMalloc((void **)&residual, n_seis * sizeof(float));
    cudaMalloc((void **)&residual_buffer, n_seis * sizeof(float));
    cudaMalloc((void **)&s2, n_model_exp * sizeof(float));

    if (!pmt->multiparameter) {
        cudaMallocHost((void **)&grad_s2_h, n_model * sizeof(float));
        cudaMallocHost((void **)&grad_s2new_h, n_model * sizeof(float));
        cudaMallocHost((void **)&p_s2, n_model * sizeof(float));
    }

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI") {
        cudaMallocHost((void **)&eps_h, n_model * sizeof(float));
        cudaMallocHost((void **)&delta_h, n_model * sizeof(float));
        if (pmt->multiparameter) {
            cudaMallocHost((void **)&epsnew_h, n_model * sizeof(float));
            cudaMallocHost((void **)&deltanew_h, n_model * sizeof(float));
            cudaMalloc((void **)&eps_grad, n_model * sizeof(float));
            cudaMalloc((void **)&delta_grad, n_model * sizeof(float));
            cudaMalloc((void **)&B_s2, n_model * sizeof(float));
            cudaMalloc((void **)&B_eps, n_model * sizeof(float));
            cudaMalloc((void **)&B_delta, n_model * sizeof(float));
            cudaMalloc((void **)&current_born, n_model_exp * sizeof(float));
            cudaMalloc((void **)&future_born, n_model_exp * sizeof(float));
            cudaMalloc((void **)&ds2, n_model_exp * sizeof(float));
            cudaMalloc((void **)&deps, n_model_exp * sizeof(float));
            cudaMalloc((void **)&ddelta, n_model_exp * sizeof(float));
        }
    }
    if (pmt->approximation == "TTI") {
        cudaMallocHost((void **)&theta_h, n_model * sizeof(float));
        if (pmt->multiparameter) {
            cudaMallocHost((void **)&thetanew_h, n_model * sizeof(float));
            cudaMalloc((void **)&theta_grad, n_model * sizeof(float));
        }
    }
}

void Inversion::freeMemory() {
    cudaFree(X);
    cudaFreeHost(X_h);
    cudaFreeHost(obs_h);
    cudaFreeHost(obs_buffer);
    cudaFreeHost(s2_h);
    cudaFreeHost(s2new_h);
    if (!pmt->multiparameter) {
        cudaFreeHost(grad_s2_h);
        cudaFreeHost(grad_s2new_h);
        cudaFreeHost(p_s2);
    }
    cudaFree(residual);
    cudaFree(residual_buffer);
    cudaFree(s2);

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI") {
        cudaFreeHost(eps_h);
        cudaFreeHost(delta_h);
        if (pmt->multiparameter) {
            cudaFreeHost(epsnew_h);
            cudaFreeHost(deltanew_h);
            cudaFree(eps_grad);
            cudaFree(delta_grad);
            cudaFree(current_born);
            cudaFree(future_born);
            cudaFree(B_s2);
            cudaFree(B_eps);
            cudaFree(B_delta);
            cudaFree(ds2);
            cudaFree(deps);
            cudaFree(ddelta);
        }
    }
    if (pmt->approximation == "TTI") {
        cudaFreeHost(theta_h);
        if (pmt->multiparameter) {
            cudaFreeHost(thetanew_h);
            cudaFree(theta_grad);
        }
    }

    delete[] water_mask;
}

float Inversion::ObjectiveFunction() {
    const int n_seis = pmt->Nrec * pmt->nt_data;
    cudaMemset(X, 0, sizeof(float));
    s2ToVp<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(s2, mdl->vp, pmt->nx_abc, pmt->nz_abc);
    readObsSeismogram(0, obs_h);
    cudaMemcpyAsync(residual, obs_h, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
    cudaStreamSynchronize(mdl->copy_stream);
    for (int shot = 0; shot < pmt->Nshot; shot++) {
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        if (shot > 0) {
            cudaStreamSynchronize(mdl->compute_stream);
            cudaStreamSynchronize(mdl->copy_stream);
            cudaMemcpyAsync(residual, residual_buffer, n_seis * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
            cudaStreamSynchronize(mdl->copy_stream);
        }

        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];
        mdl->resetFields();
        for (int k = 0; k < pmt->nt; k++) {
            mdl->forward_step();
            injectSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz, pmt->dx, pmt->dz, pmt->dt);
            if (k >= pmt->itlag) {
                storeSeismogram<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>( mdl->current, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }
            std::swap(mdl->current, mdl->future);
        }
        computeObjectiveFunction<<<XBlocks, nThreads, 0, mdl->compute_stream>>>(X, residual, mdl->seismogram, pmt->nt_data, pmt->Nrec);
        if (shot + 1 < pmt->Nshot) {
            readObsSeismogram(shot + 1, obs_buffer);
            cudaMemcpyAsync(residual_buffer, obs_buffer, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
        }
        cudaStreamSynchronize(mdl->compute_stream);
    }

    cudaMemcpyAsync(X_h, X, sizeof(float), cudaMemcpyDeviceToHost, mdl->compute_stream);
    cudaStreamSynchronize(mdl->compute_stream);
    std::cout << "info: Wave equation solved" << std::endl;

    return *X_h;
}

void Inversion::readObsSeismogram(const int shot, float *obs_h) {
    int n_seis = pmt->nt_data * pmt->Nrec;

    std::ostringstream fcut_stream;
    fcut_stream << std::fixed << std::setprecision(1) << pmt->fcut;
    std::string seismogramFile = pmt->seismogramFolder + "seismogram_shot_" + std::to_string(shot + 1) + "_Nt" + std::to_string(pmt->nt_data) + "_Nrec" + std::to_string(pmt->Nrec) + "_fcut" + fcut_stream.str() + ".bin";
    mdl->importBin(seismogramFile, obs_h, n_seis);
}

void Inversion::resetGradients() {
    const int n_model = pmt->nx * pmt->nz;

    cudaMemset(mgt->image, 0, n_model * sizeof(float));
    if (pmt->multiparameter) {
        if (pmt->approximation == "VTI" || pmt->approximation == "TTI") {
            cudaMemset(eps_grad, 0, n_model * sizeof(float));
            cudaMemset(delta_grad, 0, n_model * sizeof(float));
        }
        if (pmt->approximation == "TTI") {
            cudaMemset(theta_grad, 0, n_model * sizeof(float));
        }
    }
}

void Inversion::backward_step(const float *P, const bool TGN) {
    if (pmt->approximation == "acoustic") {
        updateAdjointWaveEquationandGradient<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>( mgt->futurebck, mgt->currentbck, P, mgt->image, mdl->vp, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt, mdl->A, pmt->N_abc);
    } else if (pmt->approximation == "VTI") {
        calculateAdjointVTIProductsAndGradients<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(mgt->currentbck, P, mgt->AUc, mgt->BUc, mgt->QCxUc, mgt->QCzUc, mgt->image, eps_grad, delta_grad, mdl->vp, mdl->epsilon, mdl->delta, pmt->dt, pmt->dx, pmt->dz, pmt->nx_abc, pmt->nz_abc, pmt->N_abc, pmt->multiparameter, B_s2, B_eps, B_delta, TGN);
        updateAdjointWaveEquationVTI<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(mgt->futurebck, mgt->currentbck, mgt->AUc, mgt->BUc, mgt->QCxUc, mgt->QCzUc, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, mdl->vp, mdl->A, pmt->N_abc);
    } else if (pmt->approximation == "TTI") {
        calculateAdjointTTIProductsAndGradients<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(mgt->currentbck, P, mgt->AUc, mgt->BUc, mgt->HUc, mgt->QCxUc, mgt->QCzUc, mgt->image,eps_grad, delta_grad, theta_grad, mdl->vp, mdl->epsilon, mdl->delta, mdl->theta,pmt->dx, pmt->dz, pmt->nx_abc, pmt->nz_abc, pmt->N_abc, pmt->multiparameter);
        updateAdjointWaveEquationTTI<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(mgt->futurebck, mgt->currentbck, mgt->AUc, mgt->BUc, mgt->HUc, mgt->QCxUc, mgt->QCzUc, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, mdl->vp, mdl->A, pmt->N_abc);
    }
}

float Inversion::calculateGradientOntheFly(bool TGN) {
    std::cout << "info: Solving " + pmt->approximation + " Reverse Time Migration by " + pmt->migration + " method." << std::endl;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec * pmt->nt_data;
    cudaMemsetAsync(X, 0, sizeof(float), mdl->compute_stream);
    s2ToVp<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(s2, mdl->vp, pmt->nx_abc, pmt->nz_abc);
    readObsSeismogram(0, obs_h);
    cudaMemcpyAsync(residual, obs_h, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
    cudaStreamSynchronize(mdl->copy_stream);
    resetGradients();
    for (int shot = 0; shot < pmt->Nshot; shot++) {
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        if (shot > 0) {
            cudaStreamSynchronize(mdl->compute_stream);
            cudaStreamSynchronize(mdl->copy_stream);
            cudaMemcpyAsync(residual, residual_buffer, n_seis * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
            cudaStreamSynchronize(mdl->copy_stream);
        }

        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];
        mgt->resetFields();
        mdl->resetFields();
        for (int k = 0; k < pmt->nt; k++) {
            mdl->forward_step();
            injectSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz, pmt->dx, pmt->dz, pmt->dt);
            if (k >= pmt->itlag) {
                storeSeismogram<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>( mdl->current, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }
            cudaMemcpyAsync(mgt->savefield + k * n_model_exp, mdl->current, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->compute_stream);
            std::swap(mdl->current, mdl->future);
        }
        computeObjectiveFunction<<<XBlocks, nThreads, 0, mdl->compute_stream>>>( X, residual, mdl->seismogram, pmt->nt_data, pmt->Nrec);
        if (shot + 1 < pmt->Nshot) {
            readObsSeismogram(shot + 1, obs_buffer);
            cudaMemcpyAsync(residual_buffer, obs_buffer, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
        }
        cudaStreamSynchronize(mdl->compute_stream);
        for (int t = pmt->nt - 1; t >= 0; t--) {
            const float *P = mgt->savefield + t * n_model_exp;
            backward_step(P, TGN);

            if (t >= pmt->itlag) {
                int it = t - pmt->itlag;
                injectAdjointSource<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>( mgt->futurebck, residual, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dt, mdl->vp);
            }

            std::swap(mgt->currentbck, mgt->futurebck);
        }
        cudaStreamSynchronize(mdl->compute_stream);
    }

    cudaMemcpy(X_h, X, sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(mdl->compute_stream);
    std::cout << "info: Reverse Time Migration" << std::endl;
    return *X_h;
}

float Inversion::calculateGradientCheckpoint(bool TGN) {
    std::cout << "info: Solving " + pmt->approximation + " Reverse Time Migration by " + pmt->migration + " method." << std::endl;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec * pmt->nt_data;
    const int last_t = pmt->nt - 1;
    const int last_checkpoint = last_t - pmt->step;

    cudaMemsetAsync(X, 0, sizeof(float), mdl->compute_stream);
    s2ToVp<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(s2, mdl->vp, pmt->nx_abc, pmt->nz_abc);
    readObsSeismogram(0, obs_h);
    cudaMemcpyAsync(residual, obs_h, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
    cudaStreamSynchronize(mdl->copy_stream);
    resetGradients();
    for (int shot = 0; shot < pmt->Nshot; shot++) {
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        if (shot > 0) {
            cudaStreamSynchronize(mdl->compute_stream);
            cudaStreamSynchronize(mdl->copy_stream);
            cudaMemcpyAsync(residual, residual_buffer, n_seis * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
            cudaStreamSynchronize(mdl->copy_stream);
        }

        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];

        mgt->resetFields();
        mdl->resetFields();
        for (int k = 0; k < pmt->nt; k++) {
            mdl->forward_step();
            injectSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz, pmt->dx, pmt->dz, pmt->dt);
            if (k >= pmt->itlag) {
                storeSeismogram<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>( mdl->current, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }

            if ((k < last_t) && ((last_t - k) % pmt->step == 0)) {
                if (k >= pmt->step) {
                    cudaStreamSynchronize(mdl->copy_stream);
                    mgt->saveCheckpoint(k - pmt->step);
                }

                cudaStreamSynchronize(mdl->compute_stream);
                cudaMemcpyAsync(mgt->d_current, mdl->current, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mgt->d_future, mdl->future, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                if (k != last_checkpoint) {
                    cudaMemcpyAsync(mgt->h_current, mgt->d_current, n_model_exp * sizeof(float), cudaMemcpyDeviceToHost, mdl->copy_stream);
                    cudaMemcpyAsync(mgt->h_future, mgt->d_future, n_model_exp * sizeof(float), cudaMemcpyDeviceToHost, mdl->copy_stream);
                }
            }
            std::swap(mdl->current, mdl->future);
        }

        computeObjectiveFunction<<<XBlocks, nThreads, 0, mdl->compute_stream>>>( X, residual, mdl->seismogram, pmt->nt_data, pmt->Nrec);
        if (shot + 1 < pmt->Nshot) {
            readObsSeismogram(shot + 1, obs_buffer);
            cudaMemcpyAsync(residual_buffer, obs_buffer, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
        }
        std::swap(mdl->current, mdl->future);
        for (int window_start = last_t; window_start >= 0; window_start -= pmt->step) {
            const int window_end = std::max(0, window_start - pmt->step + 1);

            if (window_start != last_t) {
                cudaStreamSynchronize(mdl->compute_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                cudaMemcpyAsync(mdl->current, mgt->d_current, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mdl->future, mgt->d_future, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
            }

            for (int t = window_start; t >= window_end; t--) {
                removeSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, t, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz, pmt->dx, pmt->dz, pmt->dt);
                mdl->forward_step();
                backward_step(mdl->current, TGN);
                if (t >= pmt->itlag) {
                    const int it = t - pmt->itlag;
                    injectAdjointSource<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>( mgt->futurebck, residual, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dt, mdl->vp);
                }

                std::swap(mdl->current, mdl->future);
                std::swap(mgt->currentbck, mgt->futurebck);
            }

            const int next_checkpoint = window_start - pmt->step;
            if ((next_checkpoint >= 0) && (window_start != last_t)) {
                mgt->importCheckpoint(next_checkpoint, mgt->h_current_next, mgt->h_future_next);
                cudaMemcpyAsync(mgt->d_current, mgt->h_current_next, n_model_exp * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mgt->d_future, mgt->h_future_next, n_model_exp * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
            }
        }
        cudaStreamSynchronize(mdl->compute_stream);
        cudaStreamSynchronize(mdl->copy_stream);
    }

    cudaMemcpyAsync(X_h, X, sizeof(float), cudaMemcpyDeviceToHost, mdl->compute_stream);
    cudaStreamSynchronize(mdl->compute_stream);
    std::cout << "info: Reverse Time Migration" << std::endl;

    return *X_h;
}

float Inversion::dot(const float *a, const float *b) {
    float result = 0.0;
    const int n_model = pmt->nx * pmt->nz;

    #pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < n_model; i++) {
        result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }

    return result;
}

double Inversion::dot_vector(const std::vector<float> &a, const std::vector<float> &b) {
    double result = 0.0;

    #pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < a.size(); ++i) {
        result += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return result;
}

double Inversion::norm_vector(const std::vector<float> &a) {
    double result = 0.0;

    #pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < a.size(); ++i) {
        const double ai = static_cast<double>(a[i]);
        result += ai * ai;
    }

    return std::sqrt(result);
}

void Inversion::twoLoopRecursion(const float *gradient, float *p, const std::vector<std::vector<float>> &s_store, const std::vector<std::vector<float>> &y_store) {
    if (s_store.size() != y_store.size()) {
        throw std::runtime_error("Info: Different L-BFGS history sizes.");
    }

    const int n_model = pmt->nx * pmt->nz;
    std::memcpy(p, gradient, n_model * sizeof(float));

    std::vector<float> alpha(s_store.size(), 0.0);
    std::vector<float> rho(s_store.size(), 0.0);

    for (int i = s_store.size() - 1; i >= 0; --i) {
        const float sy = dot(s_store[i].data(), y_store[i].data());

        if (sy <= 0.0 || !std::isfinite(sy)) {
            throw std::runtime_error("Info: Invalid L-BFGS pair: s^T y <= 0.");
        }

        rho[i] = 1.0 / sy;
        alpha[i] = rho[i] * dot(s_store[i].data(), p);

    #pragma omp parallel for
        for (int j = 0; j < n_model; ++j) {
            p[j] -= alpha[i] * y_store[i][j];
        }
    }

    float gamma = 1.0;

    if (s_store.size() > 0) {
        const float sy = dot(s_store.back().data(), y_store.back().data());
        const float yy = dot(y_store.back().data(), y_store.back().data());

        if (yy <= 0.0 || !std::isfinite(yy)) {
            throw std::runtime_error("Info: Invalid L-BFGS pair: y^T y <= 0.");
        }

        gamma = sy / yy;
    }

    #pragma omp parallel for
    for (int j = 0; j < n_model; ++j) {
        p[j] *= gamma;
    }

    for (int i = 0; i < s_store.size(); ++i) {
        float beta = rho[i] * dot(y_store[i].data(), p);

        float coefficient = alpha[i] - beta;

    #pragma omp parallel for
        for (int j = 0; j < n_model; ++j) {
            p[j] += coefficient * s_store[i][j];
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < n_model; j++) {
        p[j] = -p[j];
    }
}

void Inversion::applyModelStep(const std::string &parameter, const float *model, const float *p, float *model_new, const float alpha) {
    const int n_model = pmt->nx * pmt->nz;
    float model_min = 0.0f;
    float model_max = 0.0f;

    if (parameter == "s2") {
        model_min = 1.0f / (pmt->vmax * pmt->vmax);
        model_max = 1.0f / (pmt->vmin * pmt->vmin);
    } else if (parameter == "epsilon") {
        model_min = pmt->epsmin;
        model_max = pmt->epsmax;
    } else if (parameter == "delta") {
        model_min = pmt->deltamin;
        model_max = pmt->deltamax;
    } else if (parameter == "theta") {
        model_min = pmt->thetamin;
        model_max = pmt->thetamax;
    } else {
        throw std::runtime_error("Info: Invalid inversion parameter: " + parameter);
    }

#pragma omp parallel for
    for (int i = 0; i < n_model; i++) {
        model_new[i] = std::clamp(model[i] + alpha * p[i], model_min, model_max);
    }
}

void Inversion::ExpandModelDevice(const std::string &parameter, const float *model) {
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    float *model_exp = new float[n_model_exp];

    mdl->expandModel(model, model_exp);

    if (parameter == "s2") {
        cudaMemcpy(s2, model_exp, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    } else if (parameter == "epsilon") {
        cudaMemcpy(mdl->epsilon, model_exp, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    } else if (parameter == "delta") {
        cudaMemcpy(mdl->delta, model_exp, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    } else if (parameter == "theta") {
        cudaMemcpy(mdl->theta, model_exp, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        delete[] model_exp;
        throw std::runtime_error("Info: Invalid inversion parameter: " + parameter);
    }

    delete[] model_exp;
}

void Inversion::uploadPhysicalModels(const std::vector<float> &physical) {
    const int n_model = pmt->nx * pmt->nz;
    ExpandModelDevice("s2", physical.data());
    ExpandModelDevice("epsilon", physical.data() + n_model);
    ExpandModelDevice("delta", physical.data() + 2 * n_model);
}

float Inversion::calculateGradient(const std::string &parameter, float *gradient) {
    const bool multiparameter = pmt->multiparameter;
    const float *gradient_device = nullptr;
    if (parameter == "s2") {
        pmt->multiparameter = false;
        gradient_device = mgt->image;
    } else if (parameter == "epsilon" && (pmt->approximation == "VTI" || pmt->approximation == "TTI")) {
        pmt->multiparameter = true;
        gradient_device = eps_grad;
    } else if (parameter == "delta" && (pmt->approximation == "VTI" || pmt->approximation == "TTI")) {
        pmt->multiparameter = true;
        gradient_device = delta_grad;
    } else if (parameter == "theta" && pmt->approximation == "TTI") {
        pmt->multiparameter = true;
        gradient_device = theta_grad;
    } else {
        throw std::runtime_error("Info: Invalid gradient parameter: " + parameter);
    }

    float X_current = 0.0f;

    if (pmt->migration == "onthefly") {
        X_current = calculateGradientOntheFly();
    } else if (pmt->migration == "checkpoint") {
        X_current = calculateGradientCheckpoint();
    } else {
        pmt->multiparameter = multiparameter;
        throw std::runtime_error("Info: FWI gradient is not implemented for migration method: " + pmt->migration);
    }

    const int n_model = pmt->nx * pmt->nz;
    cudaMemcpy(gradient, gradient_device, n_model * sizeof(float), cudaMemcpyDeviceToHost);

    // #pragma omp parallel for
    // for (int i = 0; i < n_model; i++) {
    //     if (water_mask[i]) {
    //         gradient[i] = 0.0f;
    //     }
    // }

    pmt->multiparameter = multiparameter;
    return X_current;
}

float Inversion::calculateMultiparameterGradient(const bool update_eps, const bool update_delta, std::vector<float> &grads, std::vector<float> &models, const bool TGN) {
    const bool multiparameter = pmt->multiparameter;
    const int n_model = pmt->nx * pmt->nz;
    pmt->multiparameter = update_eps || update_delta;

    std::vector<float> physical = models;
    convertNormalizedAndPhysical(true, physical);

    uploadPhysicalModels(physical);

    float X_current = 0.0;

    if (pmt->migration == "onthefly") {
        X_current = calculateGradientOntheFly(TGN);
    } else if (pmt->migration == "checkpoint") {
        X_current = calculateGradientCheckpoint(TGN);
    } else {
        pmt->multiparameter = multiparameter;
        throw std::runtime_error("Info: Invalid migration method for FWI gradient.");
    }

    grads.assign(3 * n_model, 0.0);
    cudaMemcpy(grads.data(), mgt->image, n_model * sizeof(float), cudaMemcpyDeviceToHost);
    if (update_eps) {
        cudaMemcpy(grads.data() + n_model, eps_grad, n_model * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (update_delta) {
        cudaMemcpy(grads.data() + 2 * n_model, delta_grad, n_model * sizeof(float), cudaMemcpyDeviceToHost);
    }

    const float s2_min = 1.0f / (pmt->vmax * pmt->vmax);
    const float s2_max = 1.0f / (pmt->vmin * pmt->vmin);

    for (int i = 0; i < n_model; ++i) {
        // if (water_mask[i]) {
        //     grads[i] = 0.0;
        //     grads[n_model + i] = 0.0;
        //     grads[2 * n_model + i] = 0.0;
        // } else {
        grads[i] *= s2_max - s2_min;
        grads[n_model + i] *= pmt->epsmax - pmt->epsmin;
        grads[2 * n_model + i] *= pmt->deltamax - pmt->deltamin;
        // }
    }

    pmt->multiparameter = multiparameter;

    return X_current;
}

float Inversion::linesearch(const std::string &parameter, const float *model0, const float *grad0, const float *p, const float X0, float &X_new, float *grad_new, const bool empty, const float scale) {
    std::cout << "Info: Starting Strong Wolfe with zoom line search for parameter " << parameter << std::endl;

    const int n_model = pmt->nx * pmt->nz;
    float model_min = 0.0f;
    float model_max = 0.0f;

    if (parameter == "s2") {
        model_min = 1.0f / (pmt->vmax * pmt->vmax);
        model_max = 1.0f / (pmt->vmin * pmt->vmin);
    }
    if (parameter == "epsilon") {
        model_min = pmt->epsmin;
        model_max = pmt->epsmax;
    }
    if (parameter == "delta") {
        model_min = pmt->deltamin;
        model_max = pmt->deltamax;
    }
    if (parameter == "theta") {
        model_min = pmt->thetamin;
        model_max = pmt->thetamax;
    }

    const float c1 = 1.0e-4;
    const float c2 = 0.9;
    const int max_iters = 10;
    const float gp0 = static_cast<float>(scale) * dot(grad0, p);

    if (gp0 >= 0.0 || !std::isfinite(gp0)) {
        throw std::runtime_error("Info: Strong Wolfe needs a descent direction: g^T p < 0.");
    }

    float alpha_i;
    float alpha_max;

    if (empty) {
        alpha_i = 0.01f * (model_max - model_min);
        alpha_max = 5.0f * (model_max - model_min);
    } else {
        alpha_i = 1.0f;
        alpha_max = 5.0f;
    }

    float alpha_past = 0.0f;
    float X_past = X0;

    float *model_i = new float[n_model];
    float *grad_i = new float[n_model];

    for (int i = 0; i < max_iters; i++) {
        applyModelStep(parameter, model0, p, model_i, alpha_i);
        ExpandModelDevice(parameter, model_i);
        float X_i = ObjectiveFunction();

        std::cout << "parameter = " << parameter << std::endl;
        std::cout << "alpha = " << alpha_i << std::endl;
        std::cout << "X = " << X0 << std::endl;
        std::cout << "X_new = " << X_i << std::endl;
        std::cout << "gTp0 = " << gp0 << std::endl;
        std::cout << "Armijo limit = " << (X0 + c1 * alpha_i * gp0) << std::endl;

        if ((X_i > X0 + c1 * alpha_i * gp0) || (i > 0 && X_i >= X_past)) {
            const float alpha = zoom(parameter, model0, grad0, p, X0, gp0, alpha_past, alpha_i, X_past, X_new, grad_new, scale);
            delete[] model_i;
            delete[] grad_i;
            return alpha;
        }

        X_i = calculateGradient(parameter, grad_i);
        scaleGradient(grad_i, scale);
        const float gpi = static_cast<float>(scale) * dot(grad_i, p);
        std::cout << "gTp = " << gpi << std::endl;
        std::cout << "Curvature limit = " << -c2 * gp0 << std::endl;

        if (std::fabs(gpi) <= -c2 * gp0) {
            X_new = X_i;
            std::memcpy(grad_new, grad_i, n_model * sizeof(float));

            delete[] model_i;
            delete[] grad_i;
            return alpha_i;
        }

        if (gpi >= 0.0) {
            const float alpha = zoom(parameter, model0, grad0, p, X0, gp0, alpha_i, alpha_past, X_i, X_new, grad_new, scale);
            delete[] model_i;
            delete[] grad_i;
            return alpha;
        }

        alpha_past = alpha_i;
        X_past = X_i;
        alpha_i = alpha_i + 0.5f * (alpha_max - alpha_i);
    }

    ExpandModelDevice(parameter, model0);
    X_new = X0;
    std::memcpy(grad_new, grad0, n_model * sizeof(float));

    delete[] model_i;
    delete[] grad_i;

    std::cout << "warning: Strong Wolfe did not reduce the objective." << std::endl;
    return 0.0f;
}

float Inversion::zoom(const std::string &parameter, const float *model0, const float *grad0, const float *p, const float X0, const float gp0, float alpha_lo, float alpha_hi, const float X_lo_initial, float &X_new, float *grad_new, const float scale) {
    std::cout << "Info: Starting zoom for parameter " << parameter << std::endl;

    const int n_model = pmt->nx * pmt->nz;
    const float c1 = 1.0e-4;
    const float c2 = 0.9;
    const int max_iters = 10;

    float *model_i = new float[n_model];
    float *grad_i = new float[n_model];

    float X_lo = X_lo_initial;
    float alpha_i;

    for (int i = 0; i < max_iters; i++) {

        alpha_i = 0.5f * (alpha_lo + alpha_hi);

        applyModelStep(parameter, model0, p, model_i, alpha_i);
        ExpandModelDevice(parameter, model_i);
        float X_i = ObjectiveFunction();

        std::cout << "alpha_lo = " << alpha_lo << std::endl;
        std::cout << "alpha_hi = " << alpha_hi << std::endl;
        std::cout << "alpha = " << alpha_i << std::endl;
        std::cout << "X = " << X0 << std::endl;
        std::cout << "X_new = " << X_i << std::endl;

        if (X_i > X0 + c1 * alpha_i * gp0 || X_i >= X_lo) {
            alpha_hi = alpha_i;
        } else {

            X_i = calculateGradient(parameter, grad_i);
            scaleGradient(grad_i, scale);

            const float gpi = static_cast<float>(scale) * dot(grad_i, p);

            std::cout << "gTp = " << gpi << std::endl;
            std::cout << "Curvature limit = " << -c2 * gp0 << std::endl;

            if (std::fabs(gpi) <= -c2 * gp0) {

                X_new = X_i;
                std::memcpy(grad_new, grad_i, n_model * sizeof(float));

                delete[] model_i;
                delete[] grad_i;

                return alpha_i;
            }

            if (gpi * (alpha_hi - alpha_lo) >= 0.0) {
                alpha_hi = alpha_lo;
            }

            alpha_lo = alpha_i;
            X_lo = X_i;
        }
    }

    ExpandModelDevice(parameter, model0);
    X_new = X0;
    std::memcpy(grad_new, grad0, n_model * sizeof(float));

    delete[] model_i;
    delete[] grad_i;

    std::cout << "warning: Zoom reached the iteration limit." << std::endl;
    return 0.0f;
}

float Inversion::linesearchMultiparameter(const std::vector<float> &models, const std::vector<float> &grads, const std::vector<float> &dm, const bool update_eps, const bool update_delta, const float X0, float &X_new) {
    const int n_model = pmt->nx * pmt->nz;
    const float c1 = 1e-4f;
    const int max_iters = 10;

    X_new = X0;

    std::vector<float> models_i = models;

    const bool active[3] = {true, update_eps, update_delta};

    float gTp = 0.0;

    for (int j = 0; j < 3; ++j) {

        if (!active[j])
            continue;

        for (int i = 0; i < n_model; ++i) {
            const int k = j * n_model + i;

            gTp += grads[k] * dm[k];
        }
    }

    float alpha = 1.0f;

    if (gTp < 0.0) {

        for (int iter = 0; iter < max_iters; ++iter) {
            models_i = models;

            applyMultiparameterStep(update_eps, update_delta, models_i, dm, alpha);

            std::vector<float> physical = models_i;
            convertNormalizedAndPhysical(true, physical);

            uploadPhysicalModels(physical);

            const float Xi = ObjectiveFunction();

            const float armijo = float(X0) + c1 * alpha * gTp;

            std::cout << "alpha = " << alpha << " X = " << Xi << " armijo = " << armijo
                      << std::endl;

            if (Xi < X0 && float(Xi) <= armijo) {

                X_new = Xi;
                break;
            }

            alpha *= 0.5f;
        }
    }

    if (X_new == X0) {
        alpha = 0.0f;
    }

    return alpha;
}

void Inversion::applyMultiparameterStep(const bool update_eps, const bool update_delta, std::vector<float> &models, const std::vector<float> &dm, const float alpha) {
    const int n_model = pmt->nx * pmt->nz;

    #pragma omp parallel for
    for (int i = 0; i < n_model; i++) {
        models[i] = models[i] + alpha * dm[i];
        if (update_eps) {
            models[n_model + i] = models[n_model + i] + alpha * dm[i + n_model];
        }
        if (update_delta) {
            models[2 * n_model + i] = models[2 * n_model + i] + alpha * dm[i + 2 * n_model];
        }
    }
}

void Inversion::adjustfmax(const float fmax) {
    const float pi = 3.14159265358979323846f;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;

    pmt->fcut = fmax;
    pmt->tlag = 2.0f * std::sqrt(pi) / pmt->fcut;
    pmt->itlag = std::round(pmt->tlag / pmt->dt);
    pmt->nt = pmt->itlag + pmt->nt_data;
    cudaFree(mdl->source);
    cudaMalloc(&mdl->source, pmt->nt * sizeof(float));
    mdl->createWavelet();
    if (pmt->migration == "onthefly") {
        cudaFree(mgt->savefield);
        cudaMalloc(&mgt->savefield, pmt->nt * n_model_exp * sizeof(float));
    }
}

void Inversion::saveModel(const std::string &file_name, const float *model) {
    const int n = pmt->nx * pmt->nz;
    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        throw std::invalid_argument("Info: Could not open file. Please verify the file path.");
    }

    file.write((char *)model, n * sizeof(float));
    file.close();

    std::cout << "Info: File saved to " << file_name << std::endl;
}

void Inversion::setModel() {
    const int n_model = pmt->nx * pmt->nz;

    float *v_h = new float[n_model]();

    mdl->importBin(pmt->vpFile, v_h, n_model);
    // water_mask = mgt->createMask(v_h);
    // mgt->smoothModel(v_h, water_mask, false);
    const std::string vp_smooth_file = pmt->modelFolder + "fwi_vp_smooth_" + pmt->approximation +
                                       "_Nx" + std::to_string(pmt->nx) + "_Nz" +
                                       std::to_string(pmt->nz) + ".bin";
    saveModel(vp_smooth_file, v_h);

    #pragma omp parallel for
    for (int i = 0; i < n_model; i++) {
        s2_h[i] = 1.0f / (v_h[i] * v_h[i]);
    }

    ExpandModelDevice("s2", s2_h);

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI") {
        mdl->importBin(pmt->epsilonFile, eps_h, n_model);
        // mgt->smoothModel(eps_h, water_mask, true);

        ExpandModelDevice("epsilon", eps_h);

        mdl->importBin(pmt->deltaFile, delta_h, n_model);
        // mgt->smoothModel(delta_h, water_mask, true);

        ExpandModelDevice("delta", delta_h);

        saveModel(pmt->modelFolder + "fwi_epsilon_smooth_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + ".bin", eps_h);
        saveModel(pmt->modelFolder + "fwi_delta_smooth_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + ".bin", delta_h);
    }

    if (pmt->approximation == "TTI") {
        mdl->importBin(pmt->thetaFile, theta_h, n_model);
        // mgt->smoothModel(theta_h, water_mask, true);

    #pragma omp parallel for
        for (int i = 0; i < n_model; i++) {
            const float pi = 3.14159265358979323846f;
            theta_h[i] = theta_h[i] * pi / 180.0f;
        }

        ExpandModelDevice("theta", theta_h);
        saveModel(pmt->modelFolder + "fwi_theta_smooth_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + ".bin", theta_h);
    }

    delete[] v_h;
}

float Inversion::getGradientScale(const float *gradient) {
    const int n_model = pmt->nx * pmt->nz;
    float scale = 0.0f;
    for (int i = 0; i < n_model; i++) {
        scale = std::max(scale, std::fabs(gradient[i]));
    }

    return scale;
}

void Inversion::scaleGradient(float *gradient, const float scale) {
    const int n_model = pmt->nx * pmt->nz;
    const float inv_scale = 1.0f / scale;

    #pragma omp parallel for
    for (int i = 0; i < n_model; i++) {
        gradient[i] *= inv_scale;
    }
}

void Inversion::updateLBFGSHistory(const float *model, const float *model_new, const float *gradient, const float *gradient_new, std::vector<std::vector<float>> &s_store, std::vector<std::vector<float>> &y_store) {
    const int n_model = pmt->nx * pmt->nz;

    s_store.emplace_back(n_model);
    y_store.emplace_back(n_model);

    #pragma omp parallel for
    for (int i = 0; i < n_model; i++) {
        s_store.back()[i] = model_new[i] - model[i];
        y_store.back()[i] = gradient_new[i] - gradient[i];
    }

    const float sy = dot(s_store.back().data(), y_store.back().data());
    if (sy <= 0.0 || !std::isfinite(sy)) {
        std::cout << "warning: Invalid L-BFGS pair, sTy = " << sy << std::endl;
        s_store.pop_back();
        y_store.pop_back();

        return;
    }

    if (s_store.size() > 5) {
        s_store.erase(s_store.begin());
        y_store.erase(y_store.begin());
    }
}

void Inversion::solveFullWaveformInversionMonoparameter(){
    std::cout << "info: Solving Full Waveform Inversion" << std::endl;

    const int n_model = pmt->nx * pmt->nz;

    setModel();

    float *final_vp = new float[n_model];

    const std::string history_file = "../outputs/history.txt";
    std::ofstream history_stream(history_file);

    mdl->createCerjanVector();
    for (const float fmax : pmt->freqs) {
        std::cout << "info: FWI frequency " << fmax << std::endl;
        adjustfmax(fmax);
        s_s2_store.clear();
        y_s2_store.clear();

        float X_current = calculateGradient("s2", grad_s2_h);

        const float gmax0 = getGradientScale(grad_s2_h);
        scaleGradient(grad_s2_h, gmax0);

        const float X_freq0 = X_current;
        history_stream << X_current / X_freq0 << " " << fmax << std::endl;

        for (int itr = 0; itr < pmt->niter; itr++) {
            std::cout << "info: FWI iteration " << itr + 1 << "/" << pmt->niter << " for frequency " << fmax << std::endl;
            std::ostringstream fcut_stream;
            fcut_stream << std::fixed << std::setprecision(1) << fmax;
            const std::string gradient_file = pmt->gradientsFolder + "s2_gradient_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";
            saveModel(gradient_file, grad_s2_h);

            twoLoopRecursion(grad_s2_h, p_s2, s_s2_store, y_s2_store);

            float X_new;
            const bool empty = s_s2_store.empty();
            const float alpha_s2 = linesearch("s2", s2_h, grad_s2_h, p_s2, X_current, X_new, grad_s2new_h, empty, gmax0);

            applyModelStep("s2", s2_h, p_s2, s2new_h, alpha_s2);

            history_stream << X_new / X_freq0 << " " << fmax << std::endl;

            updateLBFGSHistory(s2_h, s2new_h, grad_s2_h, grad_s2new_h, s_s2_store, y_s2_store);

            std::swap(s2_h, s2new_h);
            std::swap(grad_s2_h, grad_s2new_h);
            X_current = X_new;

            #pragma omp parallel for
            for (int i = 0; i < n_model; i++) {
                final_vp[i] = 1.0f / std::sqrt(s2_h[i]);
            }

            const std::string model_file =
                pmt->estimatedmodelsFolder + "fwi_vp_" + pmt->approximation + "_Nx" +
                std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_itr" +
                std::to_string(itr + 1) + "_freq" + fcut_stream.str() + ".bin";
            saveModel(model_file, final_vp);
        }
    }

    delete[] final_vp;

    history_stream.close();
    std::cout << "info: FWI history saved to " << history_file << std::endl;
}


void Inversion::saveSeismogramBorn(const int shot) {
    std::ostringstream fcut_stream;
    fcut_stream << std::fixed << std::setprecision(1) << pmt->fcut;
    std::string seismogramFile = pmt->seismogramFolder + "seismogram_born_shot_" + std::to_string(shot + 1) + "_Nt" + std::to_string(pmt->nt_data) + "_Nrec" + std::to_string(pmt->Nrec) + "_fcut" + fcut_stream.str() + ".bin";

    const int n_seis = pmt->Nrec * pmt->nt_data;
    float *seismogram_h = new float[n_seis];

    cudaMemcpy(seismogram_h, mdl->seismogram, n_seis * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(seismogramFile, std::ios::binary);
    if (!file.is_open()) {
        delete[] seismogram_h;
        throw std::invalid_argument("Info: Could not open file. Please verify the file path.");
    }

    file.write((char *)seismogram_h, n_seis * sizeof(float));
    file.close();

    delete[] seismogram_h;

    std::cout << "Info: File saved to " + seismogramFile << std::endl;
}

void Inversion::computeHessianVectorProduct() {
    std::cout << "info: Solving Hessian Vector Product." << std::endl;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int last_t = pmt->nt - 1;
    const int last_checkpoint = last_t - pmt->step;

    s2ToVp<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(s2, mdl->vp, pmt->nx_abc, pmt->nz_abc);
    resetGradients();
    for (int shot = 0; shot < pmt->Nshot; shot++) {
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];

        mgt->resetFields();
        mdl->resetFields();
        cudaMemset(current_born, 0, n_model_exp * sizeof(float));
        cudaMemset(future_born, 0, n_model_exp * sizeof(float));
        for (int k = 0; k < pmt->nt; k++) {
            updateWaveEquationVTIBornTGN<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>( future_born, current_born, mdl->future, mdl->current, mdl->vp, mdl->epsilon, mdl->delta, ds2, deps, ddelta, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt, mdl->A, pmt->N_abc);
            injectSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz, pmt->dx, pmt->dz, pmt->dt);
            if (k >= pmt->itlag) {
                storeSeismogram<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>(current_born, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }

            if ((k < last_t) && ((last_t - k) % pmt->step == 0)) {
                if (k >= pmt->step) {
                    cudaStreamSynchronize(mdl->copy_stream);
                    mgt->saveCheckpoint(k - pmt->step);
                }

                cudaStreamSynchronize(mdl->compute_stream);
                cudaMemcpyAsync(mgt->d_current, mdl->current, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mgt->d_future, mdl->future, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                if (k != last_checkpoint) {
                    cudaMemcpyAsync(mgt->h_current, mgt->d_current, n_model_exp * sizeof(float), cudaMemcpyDeviceToHost, mdl->copy_stream);
                    cudaMemcpyAsync(mgt->h_future, mgt->d_future, n_model_exp * sizeof(float), cudaMemcpyDeviceToHost, mdl->copy_stream);
                }
            }
            std::swap(current_born, future_born);
            std::swap(mdl->current, mdl->future);
        }
        std::swap(mdl->current, mdl->future);
        saveSeismogramBorn(shot);
        for (int window_start = last_t; window_start >= 0; window_start -= pmt->step) {
            const int window_end = std::max(0, window_start - pmt->step + 1);

            if (window_start != last_t) {
                cudaStreamSynchronize(mdl->compute_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                cudaMemcpyAsync(mdl->current, mgt->d_current, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mdl->future, mgt->d_future, n_model_exp * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
            }

            for (int t = window_start; t >= window_end; t--) {
                removeSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, t, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz, pmt->dx, pmt->dz, pmt->dt);
                mdl->forward_step();
                backward_step(mdl->current, false);
                if (t >= pmt->itlag) {
                    const int it = t - pmt->itlag;
                    injectAdjointSource<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>( mgt->futurebck, mdl->seismogram, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dt, mdl->vp);
                }

                std::swap(mdl->current, mdl->future);
                std::swap(mgt->currentbck, mgt->futurebck);
            }

            const int next_checkpoint = window_start - pmt->step;
            if ((next_checkpoint >= 0) && (window_start != last_t)) {
                mgt->importCheckpoint(next_checkpoint, mgt->h_current_next, mgt->h_future_next);
                cudaMemcpyAsync(mgt->d_current, mgt->h_current_next, n_model_exp * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mgt->d_future, mgt->h_future_next, n_model_exp * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
            }
        }
        cudaStreamSynchronize(mdl->compute_stream);
        cudaStreamSynchronize(mdl->copy_stream);
    }
    cudaStreamSynchronize(mdl->compute_stream);
    std::cout << "info: Reverse Time Migration" << std::endl;
}

void Inversion::setTGN(const bool update_eps, const bool update_delta) {
    const int n_model = pmt->nx * pmt->nz;
    const int n = 3 * n_model;
    const float s2_min = 1.0f / (pmt->vmax * pmt->vmax);
    const float s2_max = 1.0f / (pmt->vmin * pmt->vmin);
    const bool active[3] = {true, update_eps, update_delta};

    Q.resize(n);
    p.assign(n, 0.0f);
    x.assign(n, 0.0f);
    Hx.assign(n, 0.0f);
    dm.assign(n, 0.0f);

    cudaMemcpy(Q.data(), B_s2, n_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q.data() + n_model, B_eps, n_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q.data() + 2 * n_model, B_delta, n_model * sizeof(float), cudaMemcpyDeviceToHost);

    const float fator_norm[3] = {(s2_max - s2_min), (pmt->epsmax - pmt->epsmin),
                                 (pmt->deltamax - pmt->deltamin)};
    const float s[3] = {1.0f, 2.0f, 8.0f};

    float Bmax[3] = {0.0, 0.0, 0.0};
    float lambda[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < n; ++i) {
        const int j = i / n_model;

        if (!active[j])
            continue;

        const float fator = s[j] * fator_norm[j];
        const float B = fator * fator * Q[i];

        Bmax[j] = std::max(Bmax[j], B);
    }

    for (int j = 0; j < 3; ++j) {
        lambda[j] = 0.01f * Bmax[j];
    }

    for (int i = 0; i < n; ++i) {
        const int j = i / n_model;

        if (!active[j]) {
            Q[i] = 0.0f;
            continue;
        }

        const float fator = s[j] * fator_norm[j];
        const float B = fator * fator * Q[i];

        Q[i] = float(1.0 / (B + lambda[j]));

        p[i] = Q[i] * grads[i];
        x[i] = -p[i];
    }
}

void Inversion::setBornPerturbation(const std::vector<float> &x) {
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_model = pmt->nx * pmt->nz;
    const float s2_min = 1.0f / (pmt->vmax * pmt->vmax);
    const float s2_max = 1.0f / (pmt->vmin * pmt->vmin);

    std::vector<float> model(3 * n_model);
    std::vector<float> temp(n_model_exp);

    for (int i = 0; i < n_model; ++i) {
        model[i] = (s2_max - s2_min) * x[i];
        model[n_model + i] = (pmt->epsmax - pmt->epsmin) * x[n_model + i];
        model[2 * n_model + i] = (pmt->deltamax - pmt->deltamin) * x[2 * n_model + i];
    }

    mdl->expandModel(model.data(), temp.data());
    cudaMemcpy(ds2, temp.data(), n_model_exp * sizeof(float), cudaMemcpyHostToDevice);

    mdl->expandModel(model.data() + n_model, temp.data());
    cudaMemcpy(deps, temp.data(), n_model_exp * sizeof(float), cudaMemcpyHostToDevice);

    mdl->expandModel(model.data() + 2 * n_model, temp.data());
    cudaMemcpy(ddelta, temp.data(), n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
}

float Inversion::computeEisenstatWalker(const std::vector<float> &grads, const std::vector<float> &grad_past, const std::vector<float> &Bdm_past, const float alpha_past, const float nk_past) {
    const int n = grads.size();

    float num2 = 0.0;
    float den2 = 0.0;

    #pragma omp parallel for reduction(+ : num2, den2)
    for (int i = 0; i < n; ++i) {
        const float gi = grads[i];
        const float gi_past = grad_past[i];
        const float Hx = Bdm_past[i];

        const float dif = gi - gi_past - alpha_past * Hx;
        num2 += dif * dif;
        den2 += gi_past * gi_past;
    }

    if (den2 <= 0.0)
        return 0.9f;
    float nk = std::sqrt(num2 / den2);
    const float phi = 0.5 * (1.0 + std::sqrt(5.0));
    const float nk_safe = std::pow(nk_past, phi);

    if (nk_safe > 0.1)
        nk = std::max(nk, nk_safe);
    if (nk > 1.0)
        nk = 0.9;

    return nk;
}

void Inversion::solveGaussNewton(std::vector<float> &g, const float nk, const float grad_norm, const bool update_eps, const bool update_delta) {
    const int n_model = pmt->nx * pmt->nz;
    const int n = 3 * n_model;
    const bool active[3] = {true, update_eps, update_delta};
    int inner = 0;
    const float s2_min = 1.0f / (pmt->vmax * pmt->vmax);
    const float s2_max = 1.0f / (pmt->vmin * pmt->vmin);

    while (norm_vector(g) > nk * grad_norm && inner < 5) {
        setBornPerturbation(x);
        computeHessianVectorProduct();

        cudaMemcpy(Hx.data(), mgt->image, n_model * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Hx.data() + n_model, eps_grad, n_model * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Hx.data() + 2 * n_model, delta_grad, n_model * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n; ++i) {
            const int j = i / n_model;
            const int k = i % n_model;

            if (!active[j] || water_mask[k]) {
                Hx[i] = 0.0f;
                continue;
            }
            Hx[i] = -Hx[i];

            if (j == 0) {
                Hx[i] *= (s2_max - s2_min);
            } else if (j == 1) {
                Hx[i] *= (pmt->epsmax - pmt->epsmin);
            } else {
                Hx[i] *= (pmt->deltamax - pmt->deltamin);
            }
        }

        std::cout << "inner = " << inner << std::endl;
        std::cout << "||Hx|| = " << norm_vector(Hx) << std::endl;

        beta1 = dot_vector(x, Hx);
        beta2 = dot_vector(g, p);

        std::cout << "beta1 = xTBx = " << beta1 << std::endl;
        std::cout << "beta2 = gTQg = " << beta2 << std::endl;

        if (beta1 <= 0.0 || beta2 <= 0.0) {
            dm.assign(n, 0.0);
            break;
        }

        const float alpha_TGN = beta2 / beta1;
        for (int i = 0; i < n; ++i) {
            dm[i] += alpha_TGN * x[i];
            g[i] += alpha_TGN * Hx[i];
            p[i] = Q[i] * g[i];
        }

        beta2_new = dot_vector(g, p);
        if (beta2_new < 0.0) {
            dm.assign(n, 0.0);
            break;
        }
        for (int i = 0; i < n; ++i) {
            x[i] = -p[i] + (beta2_new / beta2) * x[i];
        }
        inner++;
    }
}

void Inversion::convertNormalizedAndPhysical(const bool to_physical, std::vector<float> &models) {
    const int n_model = pmt->nx * pmt->nz;
    const float s2_min = 1.0f / (pmt->vmax * pmt->vmax);
    const float s2_max = 1.0f / (pmt->vmin * pmt->vmin);
    if (to_physical) {
        for (int i = 0; i < n_model; ++i) {
            models[i] = s2_min + (s2_max - s2_min) * models[i];
            models[n_model + i] = (models[n_model + i] * (pmt->epsmax - pmt->epsmin)) + pmt->epsmin;
            models[2 * n_model + i] = (models[2 * n_model + i] * (pmt->deltamax - pmt->deltamin)) + pmt->deltamin;
        }
    } else {
        for (int i = 0; i < n_model; ++i) {
            models[i] = (models[i] - s2_min) / (s2_max - s2_min);
            models[n_model + i] = (models[n_model + i] - pmt->epsmin) / (pmt->epsmax - pmt->epsmin);
            models[2 * n_model + i] = (models[2 * n_model + i] - pmt->deltamin) / (pmt->deltamax - pmt->deltamin);
        }
    }
}

void Inversion::solveFullWaveformInversionMultiparameterHierarchical() {
    std::cout << "info: Solving hierarchical multiparameter FWI" << std::endl;

    if (pmt->approximation != "VTI") {
        throw std::runtime_error("Hierarchical multiparameter FWI requires VTI.");
    }

    if (!pmt->multiparameter) {
        throw std::runtime_error("Hierarchical multiparameter FWI requires multiparameter=true.");
    }

    const int n_model = pmt->nx * pmt->nz;
    const float eps_start = 0.0f;
    const float delta_start = 0.0f;

    const int eps_first_itr = 1 + static_cast<int>(std::ceil(eps_start * (pmt->niter - 1)));
    const int delta_first_itr = 1 + static_cast<int>(std::ceil(delta_start * (pmt->niter - 1)));

    if (pmt->vmin <= 0.0 || pmt->epsmax <= 0.0 || pmt->deltamax <= 0.0) {
        throw std::runtime_error("FWI: referencias de normalizacao invalidas");
    }

    setModel();

    float *final_vp= new float[n_model];

    const std::string history_file = "../outputs/history.txt";
    std::ofstream history_stream(history_file);

    mdl->createCerjanVector();
    models.resize(3 * n_model);

    for (int i = 0; i < n_model; ++i) {
        models[i] = s2_h[i];
        models[n_model + i] = eps_h[i];
        models[2 * n_model + i] = delta_h[i];
    }

    convertNormalizedAndPhysical(false, models);

    for (const float fmax : pmt->freqs) {
        std::cout << "info: FWI frequency " << fmax << std::endl;

        adjustfmax(fmax);

        float X_current = ObjectiveFunction();

        const float X_freq0 = X_current;

        history_stream << X_current / X_freq0 << " " << fmax << std::endl;

        for (int itr = 0; itr < pmt->niter; itr++) {
            const int iteration = itr + 1;

            std::cout << "info: FWI iteration " << iteration << "/" << pmt->niter << " for frequency " << fmax << std::endl;

            std::ostringstream fcut_stream;
            fcut_stream << std::fixed << std::setprecision(1) << fmax;

            const bool update_eps = iteration >= eps_first_itr;
            const bool update_delta = iteration >= delta_first_itr;

            std::cout << "info: Hierarchical stage" << std::endl;
            std::cout << "s2 update: true" << std::endl;
            std::cout << "Epsilon update: " << update_eps << std::endl;
            std::cout << "Delta update: " << update_delta << std::endl;

            std::vector<float> g(3 * n_model, 0.0);
            std::vector<float> B_s2_h(n_model);
            std::vector<float> B_eps_h(n_model);
            std::vector<float> B_delta_h(n_model);

            cudaMemset(B_s2, 0, n_model * sizeof(float));
            cudaMemset(B_eps, 0, n_model * sizeof(float));
            cudaMemset(B_delta, 0, n_model * sizeof(float));

            X_current = calculateMultiparameterGradient(true, true, grads, models, true);

            if (!update_eps)
                std::fill(grads.begin() + n_model, grads.begin() + 2 * n_model, 0.0);
            if (!update_delta)
                std::fill(grads.begin() + 2 * n_model, grads.end(), 0.0);

            const std::string grad_s2_file = pmt->gradientsFolder + "s2_gradient_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";
            const std::string grad_eps_file = pmt->gradientsFolder + "epsilon_gradient_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";
            const std::string grad_delta_file = pmt->gradientsFolder + "delta_gradient_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";

            saveModel(grad_s2_file, grads.data());
            saveModel(grad_eps_file, grads.data() + n_model);
            saveModel(grad_delta_file, grads.data() + 2 * n_model);

            cudaMemcpy(B_s2_h.data(), B_s2, n_model * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(B_eps_h.data(), B_eps, n_model * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(B_delta_h.data(), B_delta, n_model * sizeof(float), cudaMemcpyDeviceToHost);

            const std::string B_s2_file = pmt->gradientsFolder + "s2_pseudohessian_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";
            const std::string B_eps_file = pmt->gradientsFolder + "epsilon_pseudohessian_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";
            const std::string B_delta_file = pmt->gradientsFolder + "delta_pseudohessian_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";

            saveModel(B_s2_file, B_s2_h.data());
            saveModel(B_eps_file, B_eps_h.data());
            saveModel(B_delta_file, B_delta_h.data());

            setTGN(update_eps, update_delta);
            g = grads;
            const float grad_norm = norm_vector(grads);
            // Temporariamente fixo ate validar J e J^T e o historico EW.
            const float nk = 0.05f;

            std::cout << "||grads|| = " << grad_norm << std::endl;
            std::cout << "||g|| = " << norm_vector(g) << std::endl;
            std::cout << "||p|| = " << norm_vector(p) << std::endl;
            std::cout << "||x|| = " << norm_vector(x) << std::endl;
            std::cout << "eta = " << nk << std::endl;
            std::cout << "CG limit = " << nk * grad_norm << std::endl;

            if (pmt->migration != "checkpoint") {
                throw std::runtime_error("TGN HVP requires checkpoint migration");
            }

            solveGaussNewton(g, nk, grad_norm, update_eps, update_delta);

            if (norm_vector(dm) <= 0.0) {
                std::cout << "warning: TGN produced no valid direction; stopping frequency"
                          << std::endl;
                break;
            }

            float X_new = X_current;
            float alpha = linesearchMultiparameter(models, grads, dm, update_eps, update_delta, X_current, X_new);

            applyMultiparameterStep(update_eps, update_delta, models, dm, alpha);

            const std::string dm_s2_file = pmt->gradientsFolder + "s2_dm_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";
            const std::string dm_eps_file = pmt->gradientsFolder + "epsilon_dm_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";
            const std::string dm_delta_file = pmt->gradientsFolder + "delta_dm_fwi_iter_" + std::to_string(itr + 1) + "_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_freq" + fcut_stream.str() + ".bin";

            saveModel(dm_s2_file, dm.data());
            saveModel(dm_eps_file, dm.data() + n_model);
            saveModel(dm_delta_file, dm.data() + 2 * n_model);

            std::vector<float> physical = models;
            convertNormalizedAndPhysical(true, physical);

            std::copy_n(physical.data(), n_model, s2_h);
            std::copy_n(physical.data() + n_model, n_model, eps_h);
            std::copy_n(physical.data() + 2 * n_model, n_model, delta_h);

            ExpandModelDevice("s2", s2_h);
            ExpandModelDevice("epsilon", eps_h);
            ExpandModelDevice("delta", delta_h);

            X_current = X_new;

            history_stream << X_current / X_freq0 << " " << fmax << std::endl;

            #pragma omp parallel for
            for (int i = 0; i < n_model; i++) {
                final_vp[i] = 1.0f / std::sqrt(s2_h[i]);
            }

            const std::string vp_model_file = pmt->estimatedmodelsFolder + "fwi_vp_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_itr" + std::to_string(iteration) + "_freq" + fcut_stream.str() + ".bin";
            saveModel(vp_model_file, final_vp);

            if (update_eps) {
                const std::string eps_model_file = pmt->estimatedmodelsFolder + "fwi_epsilon_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_itr" + std::to_string(iteration) + "_freq" + fcut_stream.str() + ".bin";
                saveModel(eps_model_file, eps_h);
            }

            if (update_delta) {
                const std::string delta_model_file = pmt->estimatedmodelsFolder + "fwi_delta_" + pmt->approximation + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_itr" + std::to_string(iteration) + "_freq" + fcut_stream.str() + ".bin";
                saveModel(delta_model_file, delta_h);
            }
        }
    }

    delete[] final_vp;
    history_stream.close();

    std::cout << "info: FWI history saved to " << history_file << std::endl;
}

void Inversion::validategradient(const float h){
    const int n_model = pmt->nx * pmt->nz;

    const std::vector<float> model0 = models;

    std::vector<float> implemented;
    std::vector<float> fd(3 * n_model, 0.0f);
    std::vector<float> diff(3 * n_model, 0.0f);

    calculateMultiparameterGradient(true, true, implemented, models, false);

    for(int j = 0; j < 3; j++){
        for(int i = 0; i < n_model; i++){

            const int k = j * n_model + i;

            std::vector<float> physical = model0;
            physical[k] += h;

            convertNormalizedAndPhysical(true, physical);
            uploadPhysicalModels(physical);

            const float X_plus = ObjectiveFunction();

            physical = model0;
            physical[k] -= h;

            convertNormalizedAndPhysical(true, physical);
            uploadPhysicalModels(physical);

            const float X_minus = ObjectiveFunction();

            fd[k] = (X_plus - X_minus) / (2.0 * h);
        }
    }

    for(int k = 0; k < 3 * n_model; k++){
        diff[k] = implemented[k] - fd[k];
    }

    const std::string names[3] = {"s2", "epsilon", "delta"};

    for(int j = 0; j < 3; j++){
        const std::string prefix = pmt->gradientsFolder + "gradient_" + names[j];
        saveModel(prefix + "_implemented.bin", implemented.data() + j * n_model);
        saveModel(prefix + "_fd.bin", fd.data() + j * n_model);
        saveModel(prefix + "_diff.bin", diff.data() + j * n_model);
    }

    std::vector<float> physical = model0;
    convertNormalizedAndPhysical(true, physical);
    uploadPhysicalModels(physical);
}

void Inversion::validateborn(const std::vector<float>& direction, const float h){
    const int n_model = pmt->nx * pmt->nz;
    const int n_seis = pmt->Nrec * pmt->nt_data;

    const std::vector<float> model0 = models;

    std::vector<float> implemented(n_seis);
    std::vector<float> data_plus(n_seis);
    std::vector<float> data_minus(n_seis);
    std::vector<float> fd(n_seis);
    std::vector<float> diff(n_seis);

    std::vector<float> physical = model0;
    convertNormalizedAndPhysical(true, physical);
    uploadPhysicalModels(physical);

    setBornPerturbation(direction);
    computeHessianVectorProduct();

    cudaMemcpy(implemented.data(), mdl->seismogram, n_seis * sizeof(float), cudaMemcpyDeviceToHost);

    physical = model0;

    for(int k = 0; k < 3 * n_model; k++){
        physical[k] += h * direction[k];
    }

    convertNormalizedAndPhysical(true, physical);
    uploadPhysicalModels(physical);
    ObjectiveFunction();

    cudaMemcpy(data_plus.data(), mdl->seismogram, n_seis * sizeof(float), cudaMemcpyDeviceToHost);

    physical = model0;

    for(int k = 0; k < 3 * n_model; k++){
        physical[k] -= h * direction[k];
    }

    convertNormalizedAndPhysical(true, physical);
    uploadPhysicalModels(physical);
    ObjectiveFunction();

    cudaMemcpy(data_minus.data(), mdl->seismogram, n_seis * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n_seis; i++){
        fd[i] = (data_plus[i] - data_minus[i]) / (2.0 * h);

        diff[i] = implemented[i] - fd[i];
    }

    const std::string names[3] = {"born_implemented.bin", "born_fd.bin", "born_diff.bin"};
    const float* data[3] = {implemented.data(), fd.data(), diff.data()};

    for(int j = 0; j < 3; j++){
        std::ofstream file(pmt->gradientsFolder + names[j], std::ios::binary);
        file.write((char*)data[j],n_seis * sizeof(float));
    }

    physical = model0;
    convertNormalizedAndPhysical(true, physical);
    uploadPhysicalModels(physical);
}

void Inversion::validateHessian(const std::vector<float>& direction, const float h){
    const int n_model = pmt->nx * pmt->nz;
    const int n = 3 * n_model;
    const float s2_min = 1.0f / (pmt->vmax * pmt->vmax);
    const float s2_max = 1.0f / (pmt->vmin * pmt->vmin);

    const std::vector<float> model0 = models;

    std::vector<float> implemented(n, 0.0f);
    std::vector<float> grad_plus;
    std::vector<float> grad_minus;
    std::vector<float> fd(n, 0.0f);
    std::vector<float> diff(n, 0.0f);

    
    std::vector<float> physical = model0;
    convertNormalizedAndPhysical(true, physical);
    uploadPhysicalModels(physical);

    setBornPerturbation(direction);
    computeHessianVectorProduct();
    
    cudaMemcpy(implemented.data(), mgt->image, n_model * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(implemented.data() + n_model, eps_grad, n_model * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(implemented.data() + 2 * n_model, delta_grad, n_model * sizeof(float),cudaMemcpyDeviceToHost);

    const float range[3] = {s2_max - s2_min, pmt->epsmax - pmt->epsmin, pmt->deltamax - pmt->deltamin};
    for(int i = 0; i < n; i++){
        const int j = i / n_model;

        implemented[i] *= -range[j];
    }

    for(int k = 0; k < n; k++){
        models[k] = model0[k] + h * direction[k];
    }

    calculateMultiparameterGradient(true, true, grad_plus, models, false);

    for(int k = 0; k < n; k++){
        models[k] = model0[k] - h * direction[k];
    }

    calculateMultiparameterGradient( true, true, grad_minus, models, false);

    for(int k = 0; k < n; k++){
        fd[k] = (grad_plus[k] - grad_minus[k]) / (2.0 * h);

        diff[k] = implemented[k] - fd[k];
    }

    const std::string names[3] = {"s2", "epsilon", "delta"};

    for(int j = 0; j < 3; j++){
        const std::string prefix = pmt->gradientsFolder + "hessian_" + names[j];
        saveModel(prefix + "_implemented.bin", implemented.data() + j * n_model);
        saveModel(prefix + "_fd.bin", fd.data() + j * n_model);
        saveModel(prefix + "_diff.bin", diff.data() + j * n_model);
    }

    models = model0;

    physical = model0;
    convertNormalizedAndPhysical(true, physical);
    uploadPhysicalModels(physical);
}

__global__ void computeObjectiveFunction(float *X, float *__restrict__ residual, const float *__restrict__ calculated, const int nt, const int Nrec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_seis = nt * Nrec;

    if (i < n_seis) {
        const float r = residual[i] - calculated[i];
        residual[i] = r;
        atomicAdd(X, 0.5f * static_cast<float>(r) * static_cast<float>(r));
    }
}

__global__ void s2ToVp(const float *__restrict__ s2, float *__restrict__ vp, int nx, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_model = nx * nz;
    if (i < n_model) {
        vp[i] = rsqrtf(s2[i]);
    }
}

__global__ void updateAdjointWaveEquationandGradient( float *__restrict__ Uf, float *__restrict__ Uc, const float *__restrict__ P, float *__restrict__ s2_grad, const float *__restrict__ vp, const int nz, const int nx, const float dz, const float dx, const float dt, const float *__restrict__ A, const int N_abc) {
    const float c0 = -2.847222222222f;
    const float c1 = 1.6f;
    const float c2 = -0.2f;
    const float c3 = 0.02539682539f;
    const float c4 = -0.00178571428f;

    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);
    const float dt2 = dt * dt;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_size = nx * nz;

    if (i >= total_size)
        return;

    const int iz = i / nx;
    const int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) {
        const float vp2 = vp[i] * vp[i];

        float pxx = (c0 * Uc[i] 
            + c1 * (Uc[i + 1] + Uc[i - 1]) 
            + c2 * (Uc[i + 2] + Uc[i - 2]) 
            + c3 * (Uc[i + 3] + Uc[i - 3]) 
            + c4 * (Uc[i + 4] + Uc[i - 4])) * inv_dx2;

        float pzz = (c0 * Uc[i] 
            + c1 * (Uc[i + nx] + Uc[i - nx]) 
            + c2 * (Uc[i + 2 * nx] + Uc[i - 2 * nx]) 
            + c3 * (Uc[i + 3 * nx] + Uc[i - 3 * nx]) 
            + c4 * (Uc[i + 4 * nx] + Uc[i - 4 * nx])) * inv_dz2;

        Uf[i] = vp2 * dt2 * (pxx + pzz) + 2.0f * Uc[i] - Uf[i];

        if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc) {
            const int xf = ix - N_abc;
            const int zf = iz - N_abc;

            const int nxf = nx - 2 * N_abc;
            const int idx = zf * nxf + xf;

            const float pxx_P = (c0 * P[i] 
                + c1 * (P[i + 1] + P[i - 1]) 
                + c2 * (P[i + 2] + P[i - 2]) 
                + c3 * (P[i + 3] + P[i - 3]) 
                + c4 * (P[i + 4] + P[i - 4])) * inv_dx2;

            const float pzz_P = (c0 * P[i] 
                + c1 * (P[i + nx] + P[i - nx]) 
                + c2 * (P[i + 2 * nx] + P[i - 2 * nx]) 
                + c3 * (P[i + 3 * nx] + P[i - 3 * nx]) 
                + c4 * (P[i + 4 * nx] + P[i - 4 * nx])) * inv_dz2;

            const float s2_product = vp2 * (pxx_P + pzz_P);
            s2_grad[idx] += Uc[i] * s2_product;
        }

        if (ix < N_abc) {
            Uf[i] *= A[ix];
            Uc[i] *= A[ix];
        }
        if (ix >= nx - N_abc) {
            Uf[i] *= A[nx - 1 - ix];
            Uc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc) {
            Uf[i] *= A[iz];
            Uc[i] *= A[iz];
        }
        if (iz >= nz - N_abc) {
            Uf[i] *= A[nz - 1 - iz];
            Uc[i] *= A[nz - 1 - iz];
        }
    }
}

__global__ void calculateAdjointVTIProductsAndGradients( const float *__restrict__ Uc, const float *__restrict__ P, float *__restrict__ AUc, float *__restrict__ BUc, float *__restrict__ QCxUc, float *__restrict__ QCzUc, float *__restrict__ s2_grad, float *__restrict__ eps_grad, float *__restrict__ delta_grad, const float *__restrict__ vp, const float *__restrict__ epsilon, const float *__restrict__ delta, const float dt, const float dx, const float dz, const int nx, const int nz, const int N_abc, const bool multiparameter, float *__restrict__ B_s2, float *__restrict__ B_eps, float *__restrict__ B_delta, const bool TGN) {

    const float c0 = -1435.0f / 504.0f;
    const float c1 = 8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 = 8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 = 4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 = 4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;
    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float inv_dz2 = inv_dz * inv_dz;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = nx * nz;

    if (i >= total_size)
        return;

    const int iz = i / nx;
    const int ix = i % nx;

    AUc[i] = 0.0f;
    BUc[i] = 0.0f;
    QCxUc[i] = 0.0f;
    QCzUc[i] = 0.0f;

    if (ix < 4 || ix >= nx - 4 || iz < 4 || iz >= nz - 4) {
        return;
    }

    const float pxx = (c0 * P[i] 
        + c1 * (P[i + 1] + P[i - 1]) 
        + c2 * (P[i + 2] + P[i - 2]) 
        + c3 * (P[i + 3] + P[i - 3]) 
        + c4 * (P[i + 4] + P[i - 4])) * inv_dx2;

    const float pzz = (c0 * P[i] 
        + c1 * (P[i + nx] + P[i - nx]) 
        + c2 * (P[i + 2 * nx] + P[i - 2 * nx]) 
        + c3 * (P[i + 3 * nx] + P[i - 3 * nx]) 
        + c4 * (P[i + 4 * nx] + P[i - 4 * nx])) * inv_dz2;

    const float px = (a1 * (P[i + 1] - P[i - 1]) 
        + a2 * (P[i + 2] - P[i - 2]) 
        + a3 * (P[i + 3] - P[i - 3]) 
        + a4 * (P[i + 4] - P[i - 4])) * inv_dx;

    const float pz = (a1 * (P[i + nx] - P[i - nx]) 
        + a2 * (P[i + 2 * nx] - P[i - 2 * nx]) 
        + a3 * (P[i + 3 * nx] - P[i - 3 * nx]) 
        + a4 * (P[i + 4 * nx] - P[i - 4 * nx])) * inv_dz;

    const float eps = epsilon[i];
    const float del = delta[i];

    const float px2 = px * px;
    const float pz2 = pz * pz;
    const float px4 = px2 * px2;
    const float pz4 = pz2 * pz2;
    const float px2pz2 = px2 * pz2;

    const float num = -2.0f * (eps - del) * px2 * pz2;
    const float den = (1.0f + 2.0f * eps) * px4 + pz4 + 2.0f * (1.0f + del) * px2 * pz2;
    const float den_reg = den + 1e-37f;
    const float inv_den = 1.0f / den_reg;
    const float Sd = num * inv_den;

    const float dnum_dpx = -4.0f * (eps - del) * px * pz2;
    const float dnum_dpz = -4.0f * (eps - del) * px2 * pz;
    const float dden_dpx = 4.0f * (1.0f + 2.0f * eps) * px * px2 + 4.0f * (1.0f + del) * px * pz2;
    const float dden_dpz = 4.0f * pz * pz2 + 4.0f * (1.0f + del) * px2 * pz;
    const float Cx = (dnum_dpx - Sd * dden_dpx) * inv_den;
    const float Cz = (dnum_dpz - Sd * dden_dpz) * inv_den;

    float dSd_deps = 0.0;
    float dSd_ddelta = 0.0;
    if (multiparameter) {
        const float dnum_deps = -2.0f * px2pz2;
        const float dnum_ddelta = 2.0f * px2pz2;
        const float dden_deps = 2.0f * px4;
        const float dden_ddelta = 2.0f * px2pz2;
        dSd_deps = (dnum_deps - Sd * dden_deps) * inv_den;
        dSd_ddelta = (dnum_ddelta - Sd * dden_ddelta) * inv_den;
    }

    const float A = 1.0f + 2.0f * eps + Sd;
    const float B = 1.0f + Sd;
    const float Q = pxx + pzz;
    const float adj = Uc[i];
    AUc[i] = A * adj;
    BUc[i] = B * adj;
    QCxUc[i] = Q * Cx * adj;
    QCzUc[i] = Q * Cz * adj;

    if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc) {
        const int xf = ix - N_abc;
        const int zf = iz - N_abc;

        const int nxf = nx - 2 * N_abc;
        const int idx = zf * nxf + xf;

        const float vp2 = vp[i] * vp[i];
        const float s2_product = vp2 * (A * pxx + B * pzz);
        s2_grad[idx] += adj * s2_product;

        if (multiparameter) {
            const float dP_deps = -2.0f * pxx - dSd_deps * Q;
            const float dP_ddelta = -dSd_ddelta * Q;

            eps_grad[idx] += adj * dP_deps;
            delta_grad[idx] += adj * dP_ddelta;

            if (TGN) {
                B_s2[idx] += s2_product * s2_product * dt;
                B_eps[idx] += dP_deps * dP_deps * dt;
                B_delta[idx] += dP_ddelta * dP_ddelta * dt;
            }
        }
    }
}

__global__ void updateAdjointWaveEquationVTI(float *__restrict__ Uf, float *__restrict__ Uc, float *__restrict__ AUc, float *__restrict__ BUc, float *__restrict__ QCxUc, float *__restrict__ QCzUc, const int nx, const int nz, const float dt, const float dx, const float dz, const float *__restrict__ vp, float *__restrict__ A, int N_abc) {
    const float c0 = -1435.0f / 504.0f;
    const float c1 = 8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 = 8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;

    const float a1 = 4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 = 4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;

    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = nx * nz;

    if (i >= total_size)
        return;

    const int iz = i / nx;
    const int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) {
        const float dxx_AUc = (c0 * AUc[i] 
            + c1 * (AUc[i + 1] + AUc[i - 1]) 
            + c2 * (AUc[i + 2] + AUc[i - 2]) 
            + c3 * (AUc[i + 3] + AUc[i - 3]) 
            + c4 * (AUc[i + 4] + AUc[i - 4])) * inv_dx2;

        const float dzz_BUc = (c0 * BUc[i] 
            + c1 * (BUc[i + nx] + BUc[i - nx]) 
            + c2 * (BUc[i + 2 * nx] + BUc[i - 2 * nx]) 
            + c3 * (BUc[i + 3 * nx] + BUc[i - 3 * nx]) 
            + c4 * (BUc[i + 4 * nx] + BUc[i - 4 * nx])) * inv_dz2;

        const float dx_QCxUc = (a1 * (QCxUc[i + 1] - QCxUc[i - 1]) 
            + a2 * (QCxUc[i + 2] - QCxUc[i - 2]) 
            + a3 * (QCxUc[i + 3] - QCxUc[i - 3]) 
            + a4 * (QCxUc[i + 4] - QCxUc[i - 4])) * inv_dx;

        const float dz_QCzUc = (a1 * (QCzUc[i + nx] - QCzUc[i - nx]) 
            + a2 * (QCzUc[i + 2 * nx] - QCzUc[i - 2 * nx]) 
            + a3 * (QCzUc[i + 3 * nx] - QCzUc[i - 3 * nx]) 
            + a4 * (QCzUc[i + 4 * nx] - QCzUc[i - 4 * nx])) * inv_dz;

        const float spatial_operator = dxx_AUc + dzz_BUc - dx_QCxUc - dz_QCzUc;
        const float vp2dt2 = vp[i] * vp[i] * dt * dt;
        Uf[i] = 2.0f * Uc[i] - Uf[i] + vp2dt2 * spatial_operator;

        if (ix < N_abc) {
            Uf[i] *= A[ix];
            Uc[i] *= A[ix];
        }
        if (ix >= nx - N_abc) {
            Uf[i] *= A[nx - 1 - ix];
            Uc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc) {
            Uf[i] *= A[iz];
            Uc[i] *= A[iz];
        }
        if (iz >= nz - N_abc) {
            Uf[i] *= A[nz - 1 - iz];
            Uc[i] *= A[nz - 1 - iz];
        }
    }
}

__global__ void calculateAdjointTTIProductsAndGradients( const float *__restrict__ Uc, const float *__restrict__ P, float *__restrict__ AUc, float *__restrict__ BUc, float *__restrict__ HUc, float *__restrict__ QCxUc, float *__restrict__ QCzUc, float *__restrict__ s2_grad, float *__restrict__ eps_grad, float *__restrict__ delta_grad, float *__restrict__ theta_grad, const float *__restrict__ vp, const float *__restrict__ epsilon, const float *__restrict__ delta, const float *__restrict__ theta, const float dx, const float dz, const int nx, const int nz, const int N_abc, const bool multiparameter) {

    const float c0 = -1435.0f / 504.0f;
    const float c1 = 8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 = 8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 = 4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 = 4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;
    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dxdz = 1.0f / (dx * dz);

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = nx * nz;

    if (i >= total_size)
        return;

    const int iz = i / nx;
    const int ix = i % nx;

    AUc[i] = 0.0f;
    BUc[i] = 0.0f;
    HUc[i] = 0.0f;
    QCxUc[i] = 0.0f;
    QCzUc[i] = 0.0f;

    if (ix < 4 || ix >= nx - 4 || iz < 4 || iz >= nz - 4) {
        return;
    }

    const float pxx = (c0 * P[i] 
        + c1 * (P[i + 1] + P[i - 1]) 
        + c2 * (P[i + 2] + P[i - 2]) 
        + c3 * (P[i + 3] + P[i - 3]) 
        + c4 * (P[i + 4] + P[i - 4])) * inv_dx2;

    const float pzz = (c0 * P[i] 
        + c1 * (P[i + nx] + P[i - nx]) 
        + c2 * (P[i + 2 * nx] + P[i - 2 * nx]) 
        + c3 * (P[i + 3 * nx] + P[i - 3 * nx]) 
        + c4 * (P[i + 4 * nx] + P[i - 4 * nx])) * inv_dz2;

    const float px = (a1 * (P[i + 1] - P[i - 1]) 
        + a2 * (P[i + 2] - P[i - 2]) 
        + a3 * (P[i + 3] - P[i - 3]) 
        + a4 * (P[i + 4] - P[i - 4])) * inv_dx;

    const float pz = (a1 * (P[i + nx] - P[i - nx]) 
        + a2 * (P[i + 2 * nx] - P[i - 2 * nx]) 
        + a3 * (P[i + 3 * nx] - P[i - 3 * nx]) 
        + a4 * (P[i + 4 * nx] - P[i - 4 * nx])) * inv_dz;

    const float eps = epsilon[i];
    const float del = delta[i];
    const float th = theta[i];

    float s;
    float c;

    sincosf(th, &s, &c);

    const float xi = px * c - pz * s;
    const float eta = px * s + pz * c;
    const float xi2 = xi * xi;
    const float eta2 = eta * eta;
    const float xi4 = xi2 * xi2;
    const float eta4 = eta2 * eta2;
    const float xi2eta2 = xi2 * eta2;

    const float num = -2.0f * (eps - del) * xi2 * eta2;
    const float den = (1.0f + 2.0f * eps) * xi4 + eta4 + 2.0f * (1.0f + del) * xi2 * eta2;
    const float den_reg = den + 1e-37f;
    const float inv_den = 1.0f / den_reg;
    const float Sd = num * inv_den;

    const float dnum_dpx = -4.0f * (eps - del) * (xi * eta2 * c + xi2 * eta * s);
    const float dnum_dpz = -4.0f * (eps - del) * (xi * eta2 * (-s) + xi2 * eta * c);
    const float dden_dpx = 4.0f * (1.0f + 2.0f * eps) * xi * xi2 * c + 4.0f * eta2 * eta * s + 4.0f * (1.0f + del) * (xi * eta2 * c + xi2 * eta * s);
    const float dden_dpz = 4.0f * (1.0f + 2.0f * eps) * xi * xi2 * (-s) + 4.0f * eta * eta2 * c + 4.0f * (1.0f + del) * (xi * eta2 * (-s) + xi2 * eta * c);
    const float Cx = (dnum_dpx - Sd * dden_dpx) * inv_den;
    const float Cz = (dnum_dpz - Sd * dden_dpz) * inv_den;

    float dSd_deps = 0.0f;
    float dSd_ddelta = 0.0f;
    float dSd_dtheta = 0.0f;
    if (multiparameter) {
        const float dnum_deps = -2.0f * xi2eta2;
        const float dnum_ddelta = 2.0f * xi2eta2;
        const float dden_deps = 2.0f * xi4;
        const float dden_ddelta = 2.0f * xi2eta2;
        const float dnum_dtheta = -2.0f * (eps - del) * (-2.0f * xi * eta * eta * eta + 2.0f * xi * xi * xi * eta);
        const float dden_dtheta = -4.0f * (1.0f + 2.0f * eps) * xi * xi * xi * eta + 4.0f * xi * eta * eta * eta + 2.0f * (1.0f + del) * (-2.0f * xi * eta * eta * eta + 2.0f * xi * xi * xi * eta);

        dSd_deps = (dnum_deps - Sd * dden_deps) * inv_den;
        dSd_ddelta = (dnum_ddelta - Sd * dden_ddelta) * inv_den;
        dSd_dtheta = (dnum_dtheta - Sd * dden_dtheta) * inv_den;
    }

    const float cos2 = c * c;
    const float sin2 = s * s;
    const float sin2th = 2.0f * s * c;
    const float cos2th = cos2 - sin2;

    const float A = (1.0f + 2.0f * eps) * cos2 + sin2 + Sd;
    const float B = (1.0f + 2.0f * eps) * sin2 + cos2 + Sd;
    const float H = 2.0f * eps * sin2th;
    const float Q = pxx + pzz;
    const float adj = Uc[i];
    AUc[i] = A * adj;
    BUc[i] = B * adj;
    HUc[i] = H * adj;
    QCxUc[i] = Q * Cx * adj;
    QCzUc[i] = Q * Cz * adj;

    if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc) {
        const int xf = ix - N_abc;
        const int zf = iz - N_abc;
        const int nxf = nx - 2 * N_abc;
        const int idx = zf * nxf + xf;

        const float pxz = (a1 * a1 * (P[i + nx + 1] - P[i - nx + 1] + P[i - nx - 1] - P[i + nx - 1]) 
            + a1 * a2 * (P[i + 2 * nx + 1] - P[i - 2 * nx + 1] + P[i - 2 * nx - 1] - P[i + 2 * nx - 1]) 
            + a1 * a3 * (P[i + 3 * nx + 1] - P[i - 3 * nx + 1] + P[i - 3 * nx - 1] - P[i + 3 * nx - 1]) 
            + a1 * a4 * (P[i + 4 * nx + 1] - P[i - 4 * nx + 1] + P[i - 4 * nx - 1] - P[i + 4 * nx - 1]) +
            
            a2 * a1 * (P[i + nx + 2] - P[i - nx + 2] + P[i - nx - 2] - P[i + nx - 2]) 
            + a2 * a2 * (P[i + 2 * nx + 2] - P[i - 2 * nx + 2] + P[i - 2 * nx - 2] - P[i + 2 * nx - 2])
            + a2 * a3 * (P[i + 3 * nx + 2] - P[i - 3 * nx + 2] + P[i - 3 * nx - 2] - P[i + 3 * nx - 2]) 
            + a2 * a4 * (P[i + 4 * nx + 2] - P[i - 4 * nx + 2] + P[i - 4 * nx - 2] - P[i + 4 * nx - 2]) +
            
            a3 * a1 * (P[i + nx + 3] - P[i - nx + 3] + P[i - nx - 3] - P[i + nx - 3]) 
            + a3 * a2 * (P[i + 2 * nx + 3] - P[i - 2 * nx + 3] + P[i - 2 * nx - 3] - P[i + 2 * nx - 3]) 
            + a3 * a3 * (P[i + 3 * nx + 3] - P[i - 3 * nx + 3] + P[i - 3 * nx - 3] - P[i + 3 * nx - 3]) 
            + a3 * a4 * (P[i + 4 * nx + 3] - P[i - 4 * nx + 3] + P[i - 4 * nx - 3] - P[i + 4 * nx - 3]) +

            a4 * a1 * (P[i + nx + 4] - P[i - nx + 4] + P[i - nx - 4] - P[i + nx - 4]) 
            + a4 * a2 * (P[i + 2 * nx + 4] - P[i - 2 * nx + 4] + P[i - 2 * nx - 4] - P[i + 2 * nx - 4]) 
            + a4 * a3 * (P[i + 3 * nx + 4] - P[i - 3 * nx + 4] + P[i - 3 * nx - 4] - P[i + 3 * nx - 4]) 
            + a4 * a4 * (P[i + 4 * nx + 4] - P[i - 4 * nx + 4] + P[i - 4 * nx - 4] - P[i + 4 * nx - 4])) * inv_dxdz;

        const float vp2 = vp[i] * vp[i];
        const float s2_product = vp2 * (A * pxx + B * pzz - H * pxz);
        s2_grad[idx] += adj * s2_product;

        if (multiparameter) {

            const float dA_dtheta = 2.0f * eps * sin2th - dSd_dtheta;
            const float dB_dtheta = -2.0f * eps * sin2th - dSd_dtheta;
            const float dC_dtheta = 4.0f * eps * cos2th;
            const float dP_deps = -(2.0f * cos2 + dSd_deps) * pxx - (2.0f * sin2 + dSd_deps) * pzz +
                                  2.0f * sin2th * pxz;
            const float dP_ddelta = -dSd_ddelta * Q;
            const float dP_dtheta = dA_dtheta * pxx + dB_dtheta * pzz + dC_dtheta * pxz;

            eps_grad[idx] += adj * dP_deps;
            delta_grad[idx] += adj * dP_ddelta;
            theta_grad[idx] += adj * dP_dtheta;
        }
    }
}

__global__ void updateAdjointWaveEquationTTI( float *__restrict__ Uf, float *__restrict__ Uc, const float *__restrict__ AUc, const float *__restrict__ BUc, const float *__restrict__ HUc, const float *__restrict__ QCxUc, const float *__restrict__ QCzUc, const int nx, const int nz, const float dt, const float dx, const float dz, const float *__restrict__ vp, float *__restrict__ A, int N_abc) {
    const float c0 = -1435.0f / 504.0f;
    const float c1 = 8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 = 8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;

    const float a1 = 4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 = 4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;

    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);
    const float inv_dxdz = 1.0f / (dx * dz);

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_size = nx * nz;

    if (i >= total_size)
        return;

    const int iz = i / nx;
    const int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) {
        const float dxx_AUc = (c0 * AUc[i] 
            + c1 * (AUc[i + 1] + AUc[i - 1]) 
            + c2 * (AUc[i + 2] + AUc[i - 2]) 
            + c3 * (AUc[i + 3] + AUc[i - 3]) 
            + c4 * (AUc[i + 4] + AUc[i - 4])) * inv_dx2;

        const float dzz_BUc =(c0 * BUc[i] 
            + c1 * (BUc[i + nx] + BUc[i - nx]) 
            + c2 * (BUc[i + 2 * nx] + BUc[i - 2 * nx]) 
            + c3 * (BUc[i + 3 * nx] + BUc[i - 3 * nx]) 
            + c4 * (BUc[i + 4 * nx] + BUc[i - 4 * nx])) * inv_dz2;

        const float dxz_HUc =
            (a1 * a1 * (HUc[i + nx + 1] - HUc[i - nx + 1] + HUc[i - nx - 1] - HUc[i + nx - 1]) 
            + a1 * a2 * (HUc[i + 2 * nx + 1] - HUc[i - 2 * nx + 1] + HUc[i - 2 * nx - 1] - HUc[i + 2 * nx - 1]) 
            + a1 * a3 * (HUc[i + 3 * nx + 1] - HUc[i - 3 * nx + 1] + HUc[i - 3 * nx - 1] - HUc[i + 3 * nx - 1]) 
            + a1 * a4 * (HUc[i + 4 * nx + 1] - HUc[i - 4 * nx + 1] + HUc[i - 4 * nx - 1] - HUc[i + 4 * nx - 1]) +

            a2 * a1 * (HUc[i + nx + 2] - HUc[i - nx + 2] + HUc[i - nx - 2] - HUc[i + nx - 2]) 
            + a2 * a2 * (HUc[i + 2 * nx + 2] - HUc[i - 2 * nx + 2] + HUc[i - 2 * nx - 2] - HUc[i + 2 * nx - 2]) 
            + a2 * a3 * (HUc[i + 3 * nx + 2] - HUc[i - 3 * nx + 2] + HUc[i - 3 * nx - 2] - HUc[i + 3 * nx - 2]) 
            + a2 * a4 * (HUc[i + 4 * nx + 2] - HUc[i - 4 * nx + 2] + HUc[i - 4 * nx - 2] - HUc[i + 4 * nx - 2]) +

            a3 * a1 * (HUc[i + nx + 3] - HUc[i - nx + 3] + HUc[i - nx - 3] - HUc[i + nx - 3]) 
            + a3 * a2 * (HUc[i + 2 * nx + 3] - HUc[i - 2 * nx + 3] + HUc[i - 2 * nx - 3] - HUc[i + 2 * nx - 3]) 
            + a3 * a3 * (HUc[i + 3 * nx + 3] - HUc[i - 3 * nx + 3] + HUc[i - 3 * nx - 3] - HUc[i + 3 * nx - 3]) 
            + a3 * a4 * (HUc[i + 4 * nx + 3] - HUc[i - 4 * nx + 3] + HUc[i - 4 * nx - 3] - HUc[i + 4 * nx - 3]) +

            a4 * a1 * (HUc[i + nx + 4] - HUc[i - nx + 4] + HUc[i - nx - 4] - HUc[i + nx - 4]) 
            + a4 * a2 * (HUc[i + 2 * nx + 4] - HUc[i - 2 * nx + 4] + HUc[i - 2 * nx - 4] - HUc[i + 2 * nx - 4]) 
            + a4 * a3 * (HUc[i + 3 * nx + 4] - HUc[i - 3 * nx + 4] + HUc[i - 3 * nx - 4] - HUc[i + 3 * nx - 4]) 
            + a4 * a4 * (HUc[i + 4 * nx + 4] - HUc[i - 4 * nx + 4] + HUc[i - 4 * nx - 4] - HUc[i + 4 * nx - 4])) * inv_dxdz;

        const float dx_QCxUc = (a1 * (QCxUc[i + 1] - QCxUc[i - 1]) 
            + a2 * (QCxUc[i + 2] - QCxUc[i - 2]) 
            + a3 * (QCxUc[i + 3] - QCxUc[i - 3]) 
            + a4 * (QCxUc[i + 4] - QCxUc[i - 4])) * inv_dx;

        const float dz_QCzUc = (a1 * (QCzUc[i + nx] - QCzUc[i - nx]) 
            + a2 * (QCzUc[i + 2 * nx] - QCzUc[i - 2 * nx]) 
            + a3 * (QCzUc[i + 3 * nx] - QCzUc[i - 3 * nx]) 
            + a4 * (QCzUc[i + 4 * nx] - QCzUc[i - 4 * nx])) * inv_dz;

        const float spatial_operator = dxx_AUc + dzz_BUc - dxz_HUc - dx_QCxUc - dz_QCzUc;
        const float vp2dt2 = vp[i] * vp[i] * dt * dt;

        Uf[i] = 2.0f * Uc[i] - Uf[i] + vp2dt2 * spatial_operator;

        if (ix < N_abc) {
            Uf[i] *= A[ix];
            Uc[i] *= A[ix];
        }
        if (ix >= nx - N_abc) {
            Uf[i] *= A[nx - 1 - ix];
            Uc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc) {
            Uf[i] *= A[iz];
            Uc[i] *= A[iz];
        }
        if (iz >= nz - N_abc) {
            Uf[i] *= A[nz - 1 - iz];
            Uc[i] *= A[nz - 1 - iz];
        }
    }
}

__global__ void updateWaveEquationVTIBornTGN( float *__restrict__ dUf, float *__restrict__ dUc, float *__restrict__ U0f, float *__restrict__ U0c, const float *__restrict__ vp, const float *__restrict__ epsilon, const float *__restrict__ delta, const float *__restrict__ ds2, const float *__restrict__ deps, const float *__restrict__ ddelta, int nz, int nx, float dz, float dx, float dt, float *__restrict__ A, int N_abc) {
    const float c0 = -2.847222222222f;
    const float c1 = 1.6f;
    const float c2 = -0.2f;
    const float c3 = 0.02539682539f;
    const float c4 = -0.00178571428f;
    const float a1 = 0.8f;
    const float a2 = -0.2f;
    const float a3 = 0.03809523809f;
    const float a4 = -0.00357142857f;

    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);
    const float dt2 = dt * dt;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz * nx)
        return;

    int iz = i / nx;
    int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) {

        float u0_xx = (c0 * U0c[i] 
            + c1 * (U0c[i + 1] + U0c[i - 1]) 
            + c2 * (U0c[i + 2] + U0c[i - 2]) 
            + c3 * (U0c[i + 3] + U0c[i - 3]) 
            + c4 * (U0c[i + 4] + U0c[i - 4])) * inv_dx2;
        float u0_zz = (c0 * U0c[i] 
            + c1 * (U0c[i + nx] + U0c[i - nx]) 
            + c2 * (U0c[i + 2 * nx] + U0c[i - 2 * nx]) 
            + c3 * (U0c[i + 3 * nx] + U0c[i - 3 * nx]) 
            + c4 * (U0c[i + 4 * nx] + U0c[i - 4 * nx])) * inv_dz2;
        float u0_x = (a1 * (U0c[i + 1] - U0c[i - 1]) 
            + a2 * (U0c[i + 2] - U0c[i - 2]) 
            + a3 * (U0c[i + 3] - U0c[i - 3]) 
            + a4 * (U0c[i + 4] - U0c[i - 4])) * inv_dx;

        float u0_z = (a1 * (U0c[i + nx] - U0c[i - nx]) 
            + a2 * (U0c[i + 2 * nx] - U0c[i - 2 * nx]) 
            + a3 * (U0c[i + 3 * nx] - U0c[i - 3 * nx]) 
            + a4 * (U0c[i + 4 * nx] - U0c[i - 4 * nx])) *inv_dz;

        float du_xx = (c0 * dUc[i] 
            + c1 * (dUc[i + 1] + dUc[i - 1]) 
            + c2 * (dUc[i + 2] + dUc[i - 2]) 
            + c3 * (dUc[i + 3] + dUc[i - 3]) 
            + c4 * (dUc[i + 4] + dUc[i - 4])) * inv_dx2;

        float du_zz = (c0 * dUc[i] 
            + c1 * (dUc[i + nx] + dUc[i - nx]) 
            + c2 * (dUc[i + 2 * nx] + dUc[i - 2 * nx]) 
            + c3 * (dUc[i + 3 * nx] + dUc[i - 3 * nx]) 
            + c4 * (dUc[i + 4 * nx] + dUc[i - 4 * nx])) * inv_dz2;

        float du_x = (a1 * (dUc[i + 1] - dUc[i - 1]) 
            + a2 * (dUc[i + 2] - dUc[i - 2]) 
            + a3 * (dUc[i + 3] - dUc[i - 3]) 
            + a4 * (dUc[i + 4] - dUc[i - 4])) * inv_dx;

        float du_z = (a1 * (dUc[i + nx] - dUc[i - nx]) 
            + a2 * (dUc[i + 2 * nx] - dUc[i - 2 * nx]) 
            + a3 * (dUc[i + 3 * nx] - dUc[i - 3 * nx]) 
            + a4 * (dUc[i + 4 * nx] - dUc[i - 4 * nx])) *inv_dz;

        float eps = epsilon[i];
        float del = delta[i];
        float ds2_i = ds2[i];
        float deps_i = deps[i];
        float ddel_i = ddelta[i];

        float x2 = u0_x * u0_x;
        float z2 = u0_z * u0_z;
        float x4 = x2 * x2;
        float z4 = z2 * z2;
        float x2z2 = x2 * z2;

        float num = -2.0f * (eps - del) * x2z2;
        float den = (1.0f + 2.0f * eps) * x4 + z4 + 2.0f * (1.0f + del) * x2z2 + 1e-37f;
        const float inv_den = 1.0f / den;
        float Sd = num * inv_den;

        float num_x = -4.0f * (eps - del) * u0_x * z2;
        float num_z = -4.0f * (eps - del) * x2 * u0_z;
        float den_x = 4.0f * (1.0f + 2.0f * eps) * x2 * u0_x + 4.0f * (1.0f + del) * u0_x * z2;
        float den_z = 4.0f * z2 * u0_z + 4.0f * (1.0f + del) * x2 * u0_z;

        float Cx = (num_x - Sd * den_x) * inv_den;
        float Cz = (num_z - Sd * den_z) * inv_den;
        float dSd_u = Cx * du_x + Cz * du_z;

        float Sd_eps = (-2.0f * x2z2 - Sd * 2.0f * x4) * inv_den;
        float Sd_del = (2.0f * x2z2 - Sd * 2.0f * x2z2) * inv_den;
        float dSd_m = Sd_eps * deps_i + Sd_del * ddel_i;

        float vp2 = vp[i] * vp[i];
        float A0 = 1.0f + 2.0f * eps + Sd;
        float B0 = 1.0f + Sd;
        float spatial = A0 * u0_xx + B0 * u0_zz;

        U0f[i] = 2.0f * U0c[i] - U0f[i] + dt2 * vp2 * spatial;

        float propagation = A0 * du_xx + B0 * du_zz + dSd_u * (u0_xx + u0_zz);
        float u0_tt = vp2 * spatial;
        float born_source_s2 = -ds2_i * u0_tt;
        float born_source_anisotropy = 2.0f * deps_i * u0_xx + dSd_m * (u0_xx + u0_zz);

        dUf[i] = 2.0f * dUc[i] - dUf[i] + dt2 * vp2 * (propagation + born_source_s2 + born_source_anisotropy);

        if (ix < N_abc) {
            U0f[i] *= A[ix];
            U0c[i] *= A[ix];
            dUf[i] *= A[ix];
            dUc[i] *= A[ix];
        }
        if (ix >= nx - N_abc) {
            U0f[i] *= A[nx - 1 - ix];
            U0c[i] *= A[nx - 1 - ix];
            dUf[i] *= A[nx - 1 - ix];
            dUc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc) {
            U0f[i] *= A[iz];
            U0c[i] *= A[iz];
            dUf[i] *= A[iz];
            dUc[i] *= A[iz];
        }
        if (iz >= nz - N_abc) {
            U0f[i] *= A[nz - 1 - iz];
            U0c[i] *= A[nz - 1 - iz];
            dUf[i] *= A[nz - 1 - iz];
            dUc[i] *= A[nz - 1 - iz];
        }
    }
}
