#include "Inversion.cuh"
#include <cstring>
#include <algorithm>

Inversion::Inversion(Survey* parameters, Modeling* modeling, Migration* migration)
{
    pmt = parameters;
    mdl = modeling;
    mgt = migration;
}

void Inversion::InitializeInversionFields(){
    const int n_model = pmt->nx * pmt->nz;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec * pmt->nt_data; 
    
    XBlocks = (n_seis + nThreads - 1)/nThreads;
    
    cudaMalloc((void**)&X, sizeof(float));
    cudaMallocHost((void**)&X_h,sizeof(float));
    cudaMallocHost((void**)&obs_h,n_seis * sizeof(float));
    cudaMallocHost((void**)&obs_buffer,n_seis * sizeof(float));
    cudaMallocHost((void**)&vp_h,n_model * sizeof(float));
    cudaMallocHost((void**)&vpnew_h,n_model * sizeof(float));
    cudaMalloc((void**)&residual, n_seis * sizeof(float));
    cudaMalloc((void**)&residual_buffer, n_seis * sizeof(float));
    cudaMalloc((void**)&slowness2, n_model_exp * sizeof(float));

    if(!pmt->multiparameter){
        cudaMallocHost((void**)&grad_vp_h,n_model *sizeof(float));
        cudaMallocHost((void**)&grad_vpnew_h,n_model * sizeof(float));
        cudaMallocHost((void**)&p_vp,n_model * sizeof(float));
    }
    
    if (pmt->migration == "checkpoint"){
        cudaMalloc((void**)&past_field, n_model_exp * sizeof(float));
    }
    
    if(pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaMallocHost((void**)&eps_h,n_model * sizeof(float));
        cudaMallocHost((void**)&delta_h,n_model * sizeof(float));
        if(pmt->multiparameter){
            cudaMallocHost((void**)&epsnew_h,n_model * sizeof(float));
            cudaMallocHost((void**)&deltanew_h,n_model * sizeof(float));
            cudaMalloc((void**)&eps_grad, n_model * sizeof(float));
            cudaMalloc((void**)&delta_grad, n_model * sizeof(float));
            cudaMalloc((void**)&B_vp,n_model * sizeof(float));
            cudaMalloc((void**)&B_eps,n_model * sizeof(float));
            cudaMalloc((void**)&B_delta,n_model * sizeof(float));
            cudaMalloc((void**)&current_born,n_model_exp * sizeof(float));
            cudaMalloc((void**)&future_born,n_model_exp * sizeof(float));
            cudaMalloc((void**)&dvp, n_model_exp * sizeof(float));
            cudaMalloc((void**)&deps, n_model_exp * sizeof(float));
            cudaMalloc((void**)&ddelta, n_model_exp * sizeof(float));
        }
    }
    if(pmt->approximation == "TTI"){
        cudaMallocHost((void**)&theta_h,n_model * sizeof(float));
        if(pmt->multiparameter){
            cudaMallocHost((void**)&thetanew_h,n_model * sizeof(float));
            cudaMalloc((void**)&theta_grad, n_model * sizeof(float));
        }
    }
}

void Inversion::freeMemory(){
    cudaFree(X);
    cudaFreeHost(X_h);
    cudaFreeHost(obs_h);
    cudaFreeHost(obs_buffer);
    cudaFreeHost(vp_h);
    cudaFreeHost(vpnew_h);
    if(!pmt->multiparameter){
        cudaFreeHost(grad_vp_h);
        cudaFreeHost(grad_vpnew_h);
        cudaFreeHost(p_vp);
    }
    cudaFree(residual);
    cudaFree(residual_buffer);
    cudaFree(slowness2);

    if (pmt->migration == "checkpoint"){
        cudaFree(past_field);
    }
    
    if(pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaFreeHost(eps_h);
        cudaFreeHost(delta_h);
        if(pmt->multiparameter){
            cudaFreeHost(epsnew_h);
            cudaFreeHost(deltanew_h);
            cudaFree(eps_grad);
            cudaFree(delta_grad);
        }
    }
    if(pmt->approximation == "TTI"){
        cudaFreeHost(theta_h);
        if(pmt->multiparameter){
            cudaFreeHost(thetanew_h);
            cudaFree(theta_grad);
        } 
    }

    delete[] water_mask;
}

float Inversion::ObjectiveFunction(){
    const int n_seis = pmt->Nrec*pmt->nt_data;
    cudaMemset(X, 0, sizeof(float));
    slowness2ToVp<<<mdl->expBlocks, nThreads,0,mdl->compute_stream>>>(slowness2,mdl->vp,pmt->nx_abc,pmt->nz_abc);
    readObsSeismogram(0, obs_h);
    cudaMemcpyAsync(residual, obs_h, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
    cudaStreamSynchronize(mdl->copy_stream);
    for (int shot = 0; shot < pmt->Nshot; shot++){
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        if(shot > 0){
            cudaStreamSynchronize(mdl->compute_stream);    
            cudaStreamSynchronize(mdl->copy_stream);
            cudaMemcpyAsync(residual, residual_buffer, n_seis * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
            cudaStreamSynchronize(mdl->copy_stream);
        }

        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];
        mdl->resetFields();
        for (int k = 0; k < pmt->nt; k++){
            mdl->forward_step(k);
            injectSource<<<1, 1, 0,mdl->compute_stream>>>(mdl->future, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz, pmt->dt);
            if(k>=pmt->itlag){
                storeSeismogram<<<mdl->seisBlocks, nThreads,0,mdl->compute_stream>>>(mdl->current, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }
            std::swap(mdl->current, mdl->future);
        }
        computeObjectiveFunction<<<XBlocks, nThreads,0,mdl->compute_stream>>>(X, residual, mdl->seismogram, pmt->nt_data, pmt->Nrec);
        if(shot + 1 < pmt->Nshot){
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

void Inversion::readObsSeismogram(const int shot, float* obs_h){
    int n_seis = pmt->nt_data * pmt->Nrec;

    std::ostringstream fcut_stream;
    fcut_stream<<std::fixed<<std::setprecision(1)<<pmt->fcut;
    std::string seismogramFile = pmt->seismogramFolder+"seismogram_shot_"+std::to_string(shot+1)+"_Nt"+std::to_string(pmt->nt_data)+"_Nrec"+std::to_string(pmt->Nrec)+"_fcut"+fcut_stream.str()+".bin";
    mdl->importBin(seismogramFile, obs_h, n_seis);

}

void Inversion::resetGradients(){
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

void Inversion::backward_step(const int k, float* Pc, float* Pp, float* Pf, const bool TGN){
    if(pmt->approximation == "acoustic"){
        updateAdjointWaveEquationandGradient<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->futurebck,mgt->currentbck,Pp,Pc,Pf,mgt->image,mdl->vp,pmt->nz_abc,pmt->nx_abc,pmt->dz,pmt->dx,pmt->dt,mdl->A,pmt->N_abc);
    }
    else if(pmt->approximation == "VTI"){
        calculateAdjointVTIProductsAndGradients<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->currentbck, Pp, Pc, Pf, mgt->AUc, mgt->BUc, mgt->QCxUc, mgt->QCzUc, mgt->image, eps_grad, delta_grad, mdl->vp, mdl->epsilon, mdl->delta, pmt->dt, pmt->dx, pmt->dz, pmt->nx_abc, pmt->nz_abc, pmt->N_abc, pmt->multiparameter, B_vp, B_eps, B_delta, TGN);
        updateAdjointWaveEquationVTI<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->futurebck,mgt->currentbck,mgt->AUc,mgt->BUc,mgt->QCxUc,mgt->QCzUc,pmt->nx_abc,pmt->nz_abc,pmt->dt,pmt->dx,pmt->dz,mdl->vp,mdl->A,pmt->N_abc);
    }
    else if(pmt->approximation == "TTI"){
        calculateAdjointTTIProductsAndGradients<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->currentbck,Pp,Pc,Pf,mgt->AUc,mgt->BUc,mgt->HUc,mgt->QCxUc,mgt->QCzUc,mgt->image,eps_grad,delta_grad,theta_grad,mdl->epsilon,mdl->delta,mdl->theta,pmt->dt,pmt->dx,pmt->dz,pmt->nx_abc,pmt->nz_abc,pmt->N_abc,pmt->multiparameter);
        updateAdjointWaveEquationTTI<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->futurebck,mgt->currentbck,mgt->AUc,mgt->BUc,mgt->HUc,mgt->QCxUc,mgt->QCzUc,pmt->nx_abc,pmt->nz_abc,pmt->dt,pmt->dx,pmt->dz,mdl->vp,mdl->A,pmt->N_abc);
    }
}

float Inversion::calculateGradientOntheFly(bool TGN = false){
    std::cout << "info: Solving " + pmt->approximation + " Reverse Time Migration by " + pmt->migration + " method." << std::endl;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec*pmt->nt_data;
    cudaMemsetAsync(X, 0, sizeof(float), mdl->compute_stream);
    slowness2ToVp<<<mdl->expBlocks, nThreads, 0, mdl->compute_stream>>>(slowness2, mdl->vp, pmt->nx_abc, pmt->nz_abc);
    readObsSeismogram(0, obs_h);
    cudaMemcpyAsync(residual, obs_h, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
    cudaStreamSynchronize(mdl->copy_stream);
    resetGradients();
    for (int shot = 0; shot < pmt->Nshot; shot++){
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        if(shot > 0){
            cudaStreamSynchronize(mdl->compute_stream);    
            cudaStreamSynchronize(mdl->copy_stream);
            cudaMemcpyAsync(residual, residual_buffer, n_seis * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
            cudaStreamSynchronize(mdl->copy_stream);
        }

        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];
        mgt->resetFields();
        mdl->resetFields();
        for (int k = 0; k < pmt->nt; k++){
            mdl->forward_step(k);
            injectSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz, pmt->dt);
            if(k>=pmt->itlag){
                storeSeismogram<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>(mdl->current, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }
            cudaMemcpyAsync(mgt->savefield + k * n_model_exp,mdl->current,n_model_exp * sizeof(float),cudaMemcpyDeviceToDevice,mdl->compute_stream);
            std::swap(mdl->current, mdl->future);
        }
        computeObjectiveFunction<<<XBlocks, nThreads, 0, mdl->compute_stream>>>(X, residual, mdl->seismogram, pmt->nt_data, pmt->Nrec);
        if(shot + 1 < pmt->Nshot){
            readObsSeismogram(shot + 1, obs_buffer);
            cudaMemcpyAsync(residual_buffer, obs_buffer, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
        }
        cudaStreamSynchronize(mdl->compute_stream);
        for (int t = pmt->nt - 1; t >= 0; t--){
            const int previous_t = std::max(t - 1, 0);
            const int next_t = std::min(t + 1, pmt->nt - 1);

            float* Pc = mgt->savefield + t * n_model_exp;
            float* Pp = mgt->savefield + previous_t * n_model_exp;
            float* Pf = mgt->savefield + next_t * n_model_exp;

            backward_step(t, Pc, Pp, Pf, TGN);

            if (t >= pmt->itlag){
                int it = t - pmt->itlag;
                injectAdjointSource<<<mdl->seisBlocks, nThreads,0,mdl->compute_stream>>>(mgt->futurebck, residual, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz,pmt->dt);
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

float Inversion::calculateGradientCheckpoint(bool TGN = false){
    std::cout<<"info: Solving "+pmt->approximation+" Reverse Time Migration by "+pmt->migration+" method."<<std::endl;
    const int n_model_exp = pmt->nx_abc*pmt->nz_abc;
    const int n_seis = pmt->Nrec*pmt->nt_data;
    const int last_t = pmt->nt-1;
    const int last_checkpoint = last_t - pmt->step;
    cudaMemsetAsync(X,0,sizeof(float),mdl->compute_stream);
    slowness2ToVp<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(slowness2,mdl->vp,pmt->nx_abc,pmt->nz_abc);
    readObsSeismogram(0, obs_h);
    cudaMemcpyAsync(residual, obs_h, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
    cudaStreamSynchronize(mdl->copy_stream);
    resetGradients();
    for(int shot = 0; shot < pmt->Nshot; shot++){
        std::cout<<"info: Shot "<<shot+1<<" of "<<pmt->Nshot<<std::endl;
        if(shot > 0){
            cudaStreamSynchronize(mdl->compute_stream);    
            cudaStreamSynchronize(mdl->copy_stream);
            cudaMemcpyAsync(residual, residual_buffer, n_seis * sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
            cudaStreamSynchronize(mdl->copy_stream);
        }

        mdl->sx=pmt->sx[shot];
        mdl->sz=pmt->sz[shot];

        mgt->resetFields();
        mdl->resetFields();
        for(int k = 0; k < pmt->nt; k++){
            mdl->forward_step(k);
            injectSource<<< 1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz, pmt->dt);
            if(k >= pmt->itlag){
                storeSeismogram<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>(mdl->current, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }

            if((k < last_t) && ((last_t-k) % pmt->step == 0)){
                if(k >= pmt->step){
                    cudaStreamSynchronize(mdl->copy_stream);
                    mgt->saveCheckpoint(k - pmt->step);
                }

                cudaStreamSynchronize(mdl->compute_stream);
                cudaMemcpyAsync(mgt->d_current, mdl->current, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mgt->d_future, mdl->future, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                if(k != last_checkpoint){
                cudaMemcpyAsync(mgt->h_current, mgt->d_current, n_model_exp*sizeof(float), cudaMemcpyDeviceToHost, mdl->copy_stream);
                cudaMemcpyAsync(mgt->h_future, mgt->d_future, n_model_exp*sizeof(float), cudaMemcpyDeviceToHost, mdl->copy_stream);
                }
            }
            std::swap(mdl->current,mdl->future);
        }

        computeObjectiveFunction<<<XBlocks,nThreads,0,mdl->compute_stream>>>(X, residual, mdl->seismogram, pmt->nt_data, pmt->Nrec);
        if(shot + 1 < pmt->Nshot){
            readObsSeismogram(shot + 1, obs_buffer);
            cudaMemcpyAsync(residual_buffer, obs_buffer, n_seis * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
        }
        std::swap(mdl->current,mdl->future);
        for(int window_start = last_t; window_start >= 0; window_start -= pmt->step){
            const int window_end=std::max(0, window_start - pmt->step + 1);

            if(window_start != last_t){
                cudaStreamSynchronize(mdl->compute_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                cudaMemcpyAsync(mdl->current, mgt->d_current, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mdl->future, mgt->d_future, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                }

            for(int t = window_start; t >= window_end; t--){
                cudaMemcpyAsync(past_field, mdl->future, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->compute_stream);
                removeSource<<< 1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, t, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz, pmt->dt);
                mdl->forward_step(t);
                backward_step(t, mdl->current, mdl->future, past_field, TGN);
                if(t>=pmt->itlag){
                    const int it=t-pmt->itlag;
                    injectAdjointSource<<<mdl->seisBlocks,nThreads,0,mdl->compute_stream>>>(mgt->futurebck, residual, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz, pmt->dt);
                }
                
                std::swap(mdl->current,mdl->future);
                std::swap(mgt->currentbck,mgt->futurebck);
            }

            const int next_checkpoint = window_start-pmt->step;
            if((next_checkpoint >= 0) && (window_start!=last_t)){
                mgt->importCheckpoint(next_checkpoint, mgt->h_current_next, mgt->h_future_next);
                cudaMemcpyAsync(mgt->d_current, mgt->h_current_next, n_model_exp*sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mgt->d_future, mgt->h_future_next, n_model_exp*sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
            }
        }
        cudaStreamSynchronize(mdl->compute_stream);
        cudaStreamSynchronize(mdl->copy_stream);
    }

    cudaMemcpyAsync(X_h, X, sizeof(float), cudaMemcpyDeviceToHost, mdl->compute_stream);
    cudaStreamSynchronize(mdl->compute_stream);
    std::cout<<"info: Reverse Time Migration"<<std::endl;

    return *X_h;
}

float Inversion::dot(const float* a,const float* b){
    float result = 0.0;
    const int n_model = pmt->nx*pmt->nz;
    #pragma omp parallel for reduction(+:result)
    for(int i = 0; i < n_model; i++){
        result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }

    return result;
}

float Inversion::vector_dot(const std::vector<float>& a, const std::vector<float>& b){
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for(int i = 0; i < a.size(); ++i){
        result += static_cast<float>(a[i]) * b[i];
    }

    return result;
}

void Inversion::twoLoopRecursion(const float* gradient, float* p, const std::vector<std::vector<float>>& s_store, const std::vector<std::vector<float>>& y_store)
{
    if (s_store.size() != y_store.size()) {
        throw std::runtime_error("Info: Different L-BFGS history sizes.");
    }

    const int n_model = pmt->nx * pmt->nz;
    std::memcpy(p, gradient, n_model*sizeof(float));

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

    if (s_store.size() > 0){
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
    for(int j = 0; j < n_model; j++){
        p[j] = -p[j];
    }

}

void Inversion::applyModelStep(const std::string& parameter, const float* model, const float* p, float* model_new, const float alpha){
    const int n_model = pmt->nx * pmt->nz;
    float model_min = 0.0f;
    float model_max = 0.0f;

    if(parameter == "vp"){
        model_min = 1.0f/(pmt->vmax*pmt->vmax);
        model_max = 1.0f/(pmt->vmin*pmt->vmin);
    }
    else if(parameter == "epsilon"){
        model_min = pmt->epsmin;
        model_max = pmt->epsmax;
    }
    else if(parameter == "delta"){
        model_min = pmt->deltamin;
        model_max = pmt->deltamax;
    }
    else if(parameter == "theta"){
        model_min = pmt->thetamin;
        model_max = pmt->thetamax;
    }
    else{
        throw std::runtime_error("Info: Invalid inversion parameter: " + parameter);
    }

    #pragma omp parallel for
    for(int i = 0; i < n_model; i++){
        model_new[i] = std::clamp(model[i] + alpha*p[i],model_min,model_max);
    }
}

void Inversion::ExpandModelDevice(const std::string& parameter, const float* model){
    const int n_model_exp = pmt->nx_abc*pmt->nz_abc;
    float* model_exp = new float[n_model_exp];

    mdl->expandModel(model,model_exp);

    if(parameter == "vp"){
        cudaMemcpy(slowness2,model_exp,n_model_exp*sizeof(float),cudaMemcpyHostToDevice);
    }
    else if(parameter == "epsilon"){
        cudaMemcpy(mdl->epsilon,model_exp,n_model_exp*sizeof(float),cudaMemcpyHostToDevice);
    }
    else if(parameter == "delta"){
        cudaMemcpy(mdl->delta,model_exp,n_model_exp*sizeof(float),cudaMemcpyHostToDevice);
    }
    else if(parameter == "theta"){
        cudaMemcpy(mdl->theta,model_exp,n_model_exp*sizeof(float),cudaMemcpyHostToDevice);
    }
    else{
        delete[] model_exp;
        throw std::runtime_error("Info: Invalid inversion parameter: " + parameter);
    }

    delete[] model_exp;
}

float Inversion::calculateGradient(const std::string& parameter, float* gradient){
    const bool multiparameter = pmt->multiparameter;
    const float* gradient_device = nullptr;
    if(parameter == "vp"){
        pmt->multiparameter = false;
        gradient_device = mgt->image;
    }
    else if(parameter == "epsilon" && (pmt->approximation == "VTI" || pmt->approximation == "TTI")){
        pmt->multiparameter = true;
        gradient_device = eps_grad;
    }
    else if(parameter == "delta" && (pmt->approximation == "VTI" || pmt->approximation == "TTI")){
        pmt->multiparameter = true;
        gradient_device = delta_grad;
    }
    else if(parameter == "theta" && pmt->approximation == "TTI"){
        pmt->multiparameter = true;
        gradient_device = theta_grad;
    }
    else{
        throw std::runtime_error("Info: Invalid gradient parameter: " + parameter);
    }

    float X_current = 0.0f;

    if(pmt->migration == "onthefly"){
        X_current = calculateGradientOntheFly();
    }
    else if(pmt->migration == "checkpoint"){
        X_current = calculateGradientCheckpoint();
    }
    else{
        pmt->multiparameter = multiparameter;
        throw std::runtime_error("Info: FWI gradient is not implemented for migration method: " + pmt->migration);
    }

    const int n_model = pmt->nx*pmt->nz;
    cudaMemcpy(gradient,gradient_device,n_model*sizeof(float),cudaMemcpyDeviceToHost);
    
    #pragma omp parallel for
    for(int i = 0; i < n_model; i++){
        if(water_mask[i]){
            gradient[i] = 0.0f;
        }
    }

    pmt->multiparameter = multiparameter;
    return X_current;
}

float Inversion::calculateMultiparameterGradient(const bool update_eps, const bool update_delta, const bool update_theta, std::vector<float>& grad, const bool TGN){
    const bool multiparameter = pmt->multiparameter;
    const int n_model = pmt->nx * pmt->nz;
    pmt->multiparameter = update_eps || update_delta || update_theta;

    float X_current = 0.0;

    if(pmt->migration == "onthefly"){
        X_current = calculateGradientOntheFly(TGN);
    }
    else if(pmt->migration == "checkpoint"){
        X_current = calculateGradientCheckpoint(TGN);
    }
    else{
        pmt->multiparameter = multiparameter;
        throw std::runtime_error("Info: Invalid migration method for FWI gradient.");
    }

    cudaMemcpy(grad.data(), mgt->image, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    if(update_eps){
        cudaMemcpy(grad.data() + n_model, eps_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    }
    if(update_delta){
        cudaMemcpy(grad.data() + 2*n_model, delta_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    }

   #pragma omp parallel for
    for(int i = 0; i < n_model; i++){
        if(water_mask[i]){
            grad[i] = 0.0f;                  
            if(update_eps){
                grad[n_model + i] = 0.0f;
            }
            if(update_delta){
                grad[2*n_model + i] = 0.0f;
            }
        }
    }

    pmt->multiparameter = multiparameter;
    
    return X_current;
}

float Inversion::Backtracking(const std::string& parameter, const float* model0, const float* grad0, const float* p, const float X0, const bool empty, const float scale){
    std::cout << "Info: Starting Armijo line search for parameter " << parameter << std::endl;
    
    float model_min = 0.0f;
    float model_max = 0.0f;
    float beta;

    if(parameter == "vp"){
        model_min = 1.0f/(pmt->vmax*pmt->vmax);
        model_max = 1.0f/(pmt->vmin*pmt->vmin);
    }
    else if(parameter == "epsilon"){
        model_min = pmt->epsmin;
        model_max = pmt->epsmax;
    }
    else if(parameter == "delta"){
        model_min = pmt->deltamin;
        model_max = pmt->deltamax;
    }
    else if(parameter == "theta"){
        model_min = pmt->thetamin;
        model_max = pmt->thetamax;
    }
    else{
        throw std::runtime_error("Info: Invalid inversion parameter: " + parameter);
    }

    const float c1 = 1.0e-4;
    const int max_iters = 10;

    const float gp0 = static_cast<float>(scale) * dot(grad0,p);
    if(gp0 >= 0.0 || !std::isfinite(gp0)){
        throw std::runtime_error("Info: Armijo needs a descent direction.");
    }

    if (empty){
        beta = 0.01f * (model_max - model_min);
    }
    else {
        beta = 1.0f;
    }
    const int n_model = pmt->nx*pmt->nz;
    float* model_i = new float[n_model];

    for(int iter = 0; iter < max_iters; ++iter){
        applyModelStep(parameter,model0,p,model_i,beta);
        ExpandModelDevice(parameter,model_i);

        const float X_i = ObjectiveFunction();
        const float armijo_limit = X0 + c1*beta*gp0;

        std::cout << "Armijo " << parameter << std::endl
                  << "beta = " << beta << std::endl
                  << "X_new = " << X_i << std::endl
                  << "limit = " << armijo_limit << std::endl;

        if(X_i <= armijo_limit){
            ExpandModelDevice(parameter,model0);
            delete[] model_i;
            return beta;
        }

        beta *= 0.5f;
    }

    ExpandModelDevice(parameter,model0);
    delete[] model_i;

    std::cout << "warning: Armijo failed for " << parameter << std::endl;
    return 0.0f;
}

float Inversion::linesearch(const std::string& parameter, const float* model0, const float* grad0, const float* p, const float X0, float& X_new, float* grad_new, const bool empty, const float scale){
    std::cout << "Info: Starting Strong Wolfe with zoom line search for parameter " << parameter << std::endl;

    const int n_model = pmt->nx*pmt->nz;
    float model_min = 0.0f;
    float model_max = 0.0f;

    if (parameter == "vp"){
        model_min = 1.0f/(pmt->vmax*pmt->vmax);
        model_max = 1.0f/(pmt->vmin*pmt->vmin);
    }
    if (parameter == "epsilon"){
        model_min = pmt->epsmin;
        model_max = pmt->epsmax;
    }
    if (parameter == "delta"){
        model_min = pmt->deltamin;
        model_max = pmt->deltamax;
    }
    if (parameter == "theta"){
        model_min = pmt->thetamin;
        model_max = pmt->thetamax;
    }

    const float c1 = 1.0e-4;
    const float c2 = 0.9;
    const int max_iters = 10;
    const float gp0 = static_cast<float>(scale) * dot(grad0,p);

    if(gp0 >= 0.0 || !std::isfinite(gp0)){
        throw std::runtime_error("Info: Strong Wolfe needs a descent direction: g^T p < 0.");
    }
    
    float alpha_i;
    float alpha_max;

    if (empty){
        alpha_i = 0.01f * (model_max - model_min);
        alpha_max = 5.0f * (model_max - model_min);
    }
    else {
        alpha_i = 1.0f;
        alpha_max = 5.0f;
    }
    
    float alpha_past = 0.0f;
    float X_past = X0;

    float* model_i = new float[n_model];
    float* grad_i = new float[n_model];

    for(int i = 0; i < max_iters; i++){
        applyModelStep(parameter,model0,p,model_i,alpha_i);
        ExpandModelDevice(parameter,model_i);
        float X_i = ObjectiveFunction();

        std::cout << "parameter = " << parameter << std::endl;
        std::cout << "alpha = " << alpha_i << std::endl;
        std::cout << "X = " << X0 << std::endl;
        std::cout << "X_new = " << X_i << std::endl;
        std::cout << "gTp0 = " << gp0 << std::endl;
        std::cout << "Armijo limit = " << (X0 + c1*alpha_i*gp0) << std::endl;

        if((X_i > X0 + c1*alpha_i*gp0) || (i > 0 && X_i >= X_past)){
            const float alpha = zoom(parameter,model0,grad0,p,X0,gp0,alpha_past,alpha_i, X_past, X_new, grad_new, scale);
            delete[] model_i;
            delete[] grad_i;
            return alpha;
        }

        X_i = calculateGradient(parameter,grad_i);
        scaleGradient(grad_i, scale);
        const float gpi = static_cast<float>(scale) * dot(grad_i,p);
        std::cout << "gTp = " << gpi << std::endl;
        std::cout << "Curvature limit = " << -c2 * gp0 << std::endl;

        if(std::fabs(gpi) <= -c2*gp0){
            X_new = X_i;
            std::memcpy(grad_new,grad_i,n_model*sizeof(float));

            delete[] model_i;
            delete[] grad_i;
            return alpha_i;
        }

        if(gpi >= 0.0){
            const float alpha = zoom(parameter,model0,grad0,p,X0,gp0,alpha_i,alpha_past, X_i, X_new, grad_new,scale);
            delete[] model_i;
            delete[] grad_i;
            return alpha;
        }

        alpha_past = alpha_i;
        X_past = X_i;
        alpha_i = alpha_i + 0.5f*(alpha_max-alpha_i);
    }
    
    ExpandModelDevice(parameter, model0);
    X_new = X0;
    std::memcpy(grad_new, grad0, n_model * sizeof(float));

    delete[] model_i;
    delete[] grad_i;

    std::cout << "warning: Strong Wolfe did not reduce the objective." << std::endl;
    return 0.0f;
}

float Inversion::zoom(const std::string& parameter,const float* model0, const float* grad0, const float* p, const float X0, const float gp0, float alpha_lo, float alpha_hi,const float X_lo_initial, float& X_new, float* grad_new, const float scale){
    std::cout << "Info: Starting zoom for parameter " << parameter << std::endl;

    const int n_model = pmt->nx*pmt->nz;
    const float c1 = 1.0e-4;
    const float c2 = 0.9;
    const int max_iters = 10;

    float* model_i = new float[n_model];
    float* grad_i = new float[n_model];

    float X_lo = X_lo_initial;
    float alpha_i;

    for(int i = 0; i < max_iters; i++){

        alpha_i = 0.5f*(alpha_lo + alpha_hi);

        applyModelStep(parameter,model0,p,model_i,alpha_i);
        ExpandModelDevice(parameter,model_i);
        float X_i = ObjectiveFunction();

        std::cout << "alpha_lo = " << alpha_lo << std::endl;
        std::cout << "alpha_hi = " << alpha_hi << std::endl;
        std::cout << "alpha = " << alpha_i << std::endl;
        std::cout << "X = " << X0 << std::endl;
        std::cout << "X_new = " << X_i << std::endl;


        if(X_i > X0 + c1*alpha_i*gp0 || X_i >= X_lo){
            alpha_hi = alpha_i;
        }
        else{

            X_i = calculateGradient(parameter,grad_i);
            scaleGradient(grad_i, scale);

            const float gpi = static_cast<float>(scale) * dot(grad_i, p);

            std::cout << "gTp = " << gpi << std::endl;
            std::cout << "Curvature limit = " << -c2 * gp0 << std::endl;

            if(std::fabs(gpi) <= -c2*gp0){

                X_new = X_i;
                std::memcpy(grad_new,grad_i,n_model*sizeof(float));

                delete[] model_i;
                delete[] grad_i;

                return alpha_i;
            }

            if(gpi*(alpha_hi-alpha_lo) >= 0.0){
                alpha_hi = alpha_lo;
            }

            alpha_lo = alpha_i;
            X_lo = X_i;
        }
    }

    ExpandModelDevice(parameter,model0);
    X_new = X0;
    std::memcpy(grad_new,grad0,n_model*sizeof(float));

    delete[] model_i;
    delete[] grad_i;

    std::cout << "warning: Zoom reached the iteration limit." << std::endl;
    return 0.0f;
}

float Inversion::linesearchMultiparameter(const float scale_vp, const float scale_eps, const float scale_delta, const float scale_theta, const bool update_eps, const bool update_delta, const bool update_theta, const float X0){
    std::cout << "Info: Starting multiparameter line search " << std::endl;
    const float c1 = 1.0e-4;
    const int max_iters = 10;

    const int n_model = pmt->nx * pmt->nz;

    float* vp_i    = new float[n_model];
    float* eps_i   = new float[n_model];
    float* delta_i = new float[n_model];
    float* theta_i = nullptr;

    if(pmt->approximation == "TTI"){
        theta_i = new float[n_model];
    }

    float gp0 = static_cast<float>(beta_vp) * static_cast<float>(scale_vp) * dot(g_vp, p_vp);

    if(update_eps){
        gp0 += beta_eps * scale_eps * dot(g_eps, p_eps);
    }

    if(update_delta){
        gp0 += beta_delta * scale_delta * dot(g_delta, p_delta);
    }

    if(update_theta){
        gp0 += beta_theta * scale_theta * dot(g_theta, p_theta);
    }

    if(gp0 >= 0.0 || !std::isfinite(gp0)){
        delete[] vp_i;
        delete[] eps_i;
        delete[] delta_i;
        delete[] theta_i;

        throw std::runtime_error(
            "Info: Multiparameter direction is not a descent direction."
        );
    }

    float alpha = 1.0f;

    for(int iter = 0; iter < max_iters; iter++){
        applyMultiparameterStep(update_eps, update_delta, update_theta,vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, vp_i, eps_i, delta_i, theta_i, alpha, beta_vp, beta_eps, beta_delta, beta_theta);
        ExpandModelDevice("vp", vp_i);
        if(update_eps){
            ExpandModelDevice("epsilon", eps_i);
        }
        if(update_delta){
            ExpandModelDevice("delta", delta_i);
        }
        if(update_theta){
            ExpandModelDevice("theta", theta_i);
        }

        const float X_i = ObjectiveFunction();

        const float armijo_limit = X0 + c1 * alpha * gp0;

        std::cout << "alpha = " << alpha << std::endl;
        std::cout << "X = " << X0 << std::endl;
        std::cout << "X_new = " << X_i << std::endl;
        std::cout << "gTp0 = " << gp0 << std::endl;
        std::cout << "Armijo limit = " << armijo_limit << std::endl;

        if(X_i <= armijo_limit){

            delete[] vp_i;
            delete[] eps_i;
            delete[] delta_i;
            delete[] theta_i;

            return alpha;
        }

        alpha *= 0.5f;
    }

    ExpandModelDevice("vp", vp);
    if(update_eps){
        ExpandModelDevice("epsilon", epsilon);
    }
    if(update_delta){
        ExpandModelDevice("delta", delta);
    }
    if(update_theta){
        ExpandModelDevice("theta", theta);
    }

    delete[] vp_i;
    delete[] eps_i;
    delete[] delta_i;
    delete[] theta_i;

    std::cout << "warning: Multiparameter Armijo failed." << std::endl;

    return 0.0f;
}

void Inversion::applyMultiparameterStep(const bool update_eps, const bool update_delta, const bool update_theta,const float* vp, const float* epsilon, const float* delta, const float* theta, const float* p_vp, const float* p_eps, const float* p_delta, const float* p_theta, float* vp_new, float* epsilon_new, float* delta_new, float* theta_new, const float alpha, const float beta_vp, const float beta_eps, const float beta_delta, const float beta_theta){
    const int n_model = pmt->nx * pmt->nz;

    const float vp_min = 1.0f / (pmt->vmax * pmt->vmax);
    const float vp_max = 1.0f / (pmt->vmin * pmt->vmin);

    #pragma omp parallel for
    for (int i = 0; i < n_model; i++){
        vp_new[i] = std::clamp(vp[i] + alpha * beta_vp * p_vp[i], vp_min, vp_max);
        if (update_eps){
            epsilon_new[i] = std::clamp(epsilon[i] + alpha * beta_eps * p_eps[i], pmt->epsmin, pmt->epsmax);
        }
        if (update_delta){
            delta_new[i] = std::clamp(delta[i] + alpha * beta_delta * p_delta[i], pmt->deltamin, pmt->deltamax);
        }
        if (update_theta){ 
            theta_new[i] = std::clamp(theta[i] + alpha * beta_theta * p_theta[i], pmt->thetamin, pmt->thetamax);
        }
    }
}

void Inversion::adjustfmax(const float fmax){
    const float pi = 3.14159265358979323846f;
    const int n_model_exp = pmt->nx_abc*pmt->nz_abc;

    pmt->fcut = fmax;
    pmt->tlag = 2.0f*std::sqrt(pi)/pmt->fcut;
    pmt->itlag = std::round(pmt->tlag/pmt->dt);
    pmt->nt = pmt->itlag + pmt->nt_data;
    cudaFree(mdl->source);
    cudaMalloc(&mdl->source, pmt->nt * sizeof(float));
    mdl->createWavelet();
    if (pmt->migration == "onthefly"){
        cudaFree(mgt->savefield);
        cudaMalloc(&mgt->savefield, pmt->nt*n_model_exp*sizeof(float));
    }
}

void Inversion::saveModel(const std::string& file_name, const float* model){
    const int n = pmt->nx * pmt->nz;
    std::ofstream file(file_name,std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("Info: Could not open file. Please verify the file path.");
    }

    file.write((char*)model,n*sizeof(float));
    file.close();

    std::cout << "Info: File saved to " << file_name << std::endl;
}

void Inversion::setModel(){
    const int n_model = pmt->nx * pmt->nz;

    float* v_h = new float[n_model]();

    mdl->importBin(pmt->vpFile, v_h, n_model);
    water_mask = mgt->createMask(v_h);
    mgt->smoothModel(v_h, water_mask, false);
    const std::string vp_smooth_file = pmt->modelFolder + "fwi_vp_smooth_"+ pmt->approximation+ "_Nx"+std::to_string(pmt->nx)+ "_Nz"+std::to_string(pmt->nz)+".bin";
    saveModel(vp_smooth_file,v_h);

    #pragma omp parallel for
    for(int i = 0; i < n_model; i++){
        vp_h[i] = 1.0f/(v_h[i]*v_h[i]);
    }

    ExpandModelDevice("vp",vp_h);

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        mdl->importBin(pmt->epsilonFile, eps_h, n_model);
        mgt->smoothModel(eps_h, water_mask, true);

        ExpandModelDevice("epsilon",eps_h);

        mdl->importBin(pmt->deltaFile, delta_h, n_model);
        mgt->smoothModel(delta_h, water_mask, true);

        ExpandModelDevice("delta",delta_h);
        
        saveModel(pmt->modelFolder+"fwi_epsilon_smooth_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+".bin",eps_h);
        saveModel(pmt->modelFolder+"fwi_delta_smooth_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+".bin",delta_h);
    }

    if (pmt->approximation == "TTI"){
        mdl->importBin(pmt->thetaFile, theta_h, n_model);
        mgt->smoothModel(theta_h, water_mask, true);

        #pragma omp parallel for
        for (int i = 0; i < n_model; i++){
            const float pi = 3.14159265358979323846f;
            theta_h[i] = theta_h[i] * pi / 180.0f;
        }

        ExpandModelDevice("theta",theta_h);
        saveModel(pmt->modelFolder+"fwi_theta_smooth_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+".bin",theta_h);
    }

    delete[] v_h;
}

float Inversion::getGradientScale(const float* gradient){
    const int n_model = pmt->nx * pmt->nz;
    float scale = 0.0f;
    for (int i = 0; i < n_model; i++) {
        scale = std::max(scale,std::fabs(gradient[i]));  
    }

    return scale;
}

void Inversion::scaleGradient(float* gradient, const float scale){
    const int n_model = pmt->nx * pmt->nz;
    const float inv_scale = 1.0f / scale;

    #pragma omp parallel for
    for (int i = 0; i < n_model; i++){
        gradient[i] *= inv_scale;
    }
}

void Inversion::updateLBFGSHistory(const float* model, const float* model_new, const float* gradient, const float* gradient_new, std::vector<std::vector<float>>& s_store, std::vector<std::vector<float>>& y_store){
    const int n_model = pmt->nx * pmt->nz;

    s_store.emplace_back(n_model);
    y_store.emplace_back(n_model);

    #pragma omp parallel for
    for(int i = 0; i < n_model; i++){
        s_store.back()[i] = model_new[i] - model[i];
        y_store.back()[i] = gradient_new[i] - gradient[i];
    }

    const float sy = dot(s_store.back().data(), y_store.back().data());
    if(sy <= 0.0 || !std::isfinite(sy)){
        std::cout << "warning: Invalid L-BFGS pair, sTy = " << sy << std::endl;
        s_store.pop_back();
        y_store.pop_back();

        return;
    }

    if(s_store.size() > 5){
        s_store.erase(s_store.begin());
        y_store.erase(y_store.begin());
    }
}

void Inversion::solveFullWaveformInversionMonoparameter(){
    std::cout << "info: Solving Full Waveform Inversion" << std::endl;

    const int n_model = pmt->nx*pmt->nz;

    setModel();

    float* final_model = new float[n_model];

    const std::string history_file = "../outputs/history.txt";
    std::ofstream history_stream(history_file);

    mdl->createCerjanVector();
    for(const float fmax : pmt->freqs){
        std::cout << "info: FWI frequency " << fmax <<std::endl;
        adjustfmax(fmax);
        s_vp_store.clear();
        y_vp_store.clear();

        float X_current = calculateGradient("vp",grad_vp_h);

        const float gmax0 = getGradientScale(grad_vp_h);
        scaleGradient(grad_vp_h, gmax0);

        const float X_freq0 = X_current;
        history_stream << X_current/X_freq0 << " " << fmax << std::endl;

        for(int itr = 0; itr < pmt->niter; itr++){
            std::cout << "info: FWI iteration " << itr + 1 << "/" << pmt->niter << " for frequency " << fmax << std::endl;
            std::ostringstream fcut_stream;
            fcut_stream<<std::fixed<<std::setprecision(1)<<fmax;
            const std::string gradient_file = pmt->gradientsFolder+"vp_gradient_fwi_iter_"+std::to_string(itr+1)+"_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_freq"+fcut_stream.str()+".bin";
            saveModel(gradient_file,grad_vp_h);

            twoLoopRecursion(grad_vp_h,p_vp,s_vp_store,y_vp_store);

            float X_new;
            const bool empty = s_vp_store.empty();
            const float alpha_vp = linesearch("vp",vp_h,grad_vp_h,p_vp,X_current, X_new, grad_vpnew_h,empty, gmax0);

            applyModelStep("vp",vp_h,p_vp,vpnew_h,alpha_vp);

            history_stream << X_new/X_freq0 << " " << fmax << std::endl;

            updateLBFGSHistory(vp_h,  vpnew_h, grad_vp_h, grad_vpnew_h, s_vp_store, y_vp_store);

            std::swap(vp_h, vpnew_h);
            std::swap(grad_vp_h, grad_vpnew_h);
            X_current = X_new;

            #pragma omp parallel for
            for(int i = 0; i < n_model; i++){
                final_model[i] = 1.0f/std::sqrt(vp_h[i]);
            }

            const std::string model_file = pmt->estimatedmodelsFolder+"fwi_vp_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_itr"+std::to_string(itr+1)+"_freq"+fcut_stream.str()+".bin";
            saveModel(model_file,final_model);
        }
    }

    delete[] final_model;

    history_stream.close();
    std::cout << "info: FWI history saved to " << history_file << std::endl;
}

void Inversion::bornforward_step(const int k){
    if (pmt->approximation == "acoustic"){
        updateWaveEquationBorn<<<expBlocks, nThreads, 0, compute_stream>>>(future_born, current_born,mdl->future, mdl->current,mdl->vp,dvp, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt, mdl->A, pmt->N_abc);
    }
    else if (pmt->approximation == "VTI"){
        updateWaveEquationVTIBorn<<<expBlocks, nThreads, 0, compute_stream>>>(future_born, current_born,mdl->future, mdl->current,mdl->vp, mdl->epsilon, mdl->delta, dvp, deps, ddelta, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt, mdl->A, pmt->N_abc);
    }
}

void Inversion::computeHessianVectorProduct(){
    std::cout << "info: Solving Hessian Vector Product." << std::endl;
    const int n_model_exp = pmt->nx_abc*pmt->nz_abc;
    const int n_seis = pmt->Nrec*pmt->nt_data;
    const int last_t = pmt->nt-1;
    const int last_checkpoint = last_t - pmt->step;
    slowness2ToVp<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(slowness2,mdl->vp,pmt->nx_abc,pmt->nz_abc);
    resetGradients();
    for(int shot = 0; shot < pmt->Nshot; shot++){
        std::cout<<"info: Shot "<<shot+1<<" of "<<pmt->Nshot<<std::endl;
        mdl->sx=pmt->sx[shot];
        mdl->sz=pmt->sz[shot];

        mgt->resetFields();
        mdl->resetFields();
        cudaMemset(current_born, 0, n_model_exp * sizeof(float));
        cudaMemset(future_born, 0, n_model_exp * sizeof(float));
        for(int k = 0; k < pmt->nt; k++){
            bornforward_step(k);
            injectSource<<< 1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz, pmt->dt);
            if(k >= pmt->itlag){
                storeSeismogram<<<mdl->seisBlocks, nThreads, 0, mdl->compute_stream>>>(current_born, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }

            if((k < last_t) && ((last_t-k) % pmt->step == 0)){
                if(k >= pmt->step){
                    cudaStreamSynchronize(mdl->copy_stream);
                    mgt->saveCheckpoint(k - pmt->step);
                }

                cudaStreamSynchronize(mdl->compute_stream);
                cudaMemcpyAsync(mgt->d_current, mdl->current, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mgt->d_future, mdl->future, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                if(k != last_checkpoint){
                cudaMemcpyAsync(mgt->h_current, mgt->d_current, n_model_exp*sizeof(float), cudaMemcpyDeviceToHost, mdl->copy_stream);
                cudaMemcpyAsync(mgt->h_future, mgt->d_future, n_model_exp*sizeof(float), cudaMemcpyDeviceToHost, mdl->copy_stream);
                }
            }
            std::swap(current_born, future_born);
            std::swap(mdl->current,mdl->future);
        }
        std::swap(mdl->current,mdl->future);
        for(int window_start = last_t; window_start >= 0; window_start -= pmt->step){
            const int window_end=std::max(0, window_start - pmt->step + 1);

            if(window_start != last_t){
                cudaStreamSynchronize(mdl->compute_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                cudaMemcpyAsync(mdl->current, mgt->d_current, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mdl->future, mgt->d_future, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                }

            for(int t = window_start; t >= window_end; t--){
                cudaMemcpyAsync(past_field, mdl->future, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->compute_stream);
                removeSource<<< 1, 1, 0, mdl->compute_stream>>>(mdl->future, mdl->source, t, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz, pmt->dt);
                mdl->forward_step(t);
                backward_step(t, mdl->current, mdl->future, past_field, false);
                if(t>=pmt->itlag){
                    const int it=t-pmt->itlag;
                    injectAdjointSource<<<mdl->seisBlocks,nThreads,0,mdl->compute_stream>>>(mgt->futurebck, mdl->seismogram, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz, pmt->dt);
                }
                
                std::swap(mdl->current,mdl->future);
                std::swap(mgt->currentbck,mgt->futurebck);
            }

            const int next_checkpoint = window_start-pmt->step;
            if((next_checkpoint >= 0) && (window_start!=last_t)){
                mgt->importCheckpoint(next_checkpoint, mgt->h_current_next, mgt->h_future_next);
                cudaMemcpyAsync(mgt->d_current, mgt->h_current_next, n_model_exp*sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
                cudaMemcpyAsync(mgt->d_future, mgt->h_future_next, n_model_exp*sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
            }
        }
        cudaStreamSynchronize(mdl->compute_stream);
        cudaStreamSynchronize(mdl->copy_stream);
    }
    std::cout<<"info: Reverse Time Migration"<<std::endl;
}

void Inversion::setTGN(){
    const int n_model = pmt->nx * pmt->nz;

    Q.resize(3 * n_model);
    p.resize(3 * n_model);
    x.resize(3 * n_model);
    Hx.assign(3 * n_model, 0.0f);
    dm.assign(3 * n_model, 0.0f);

    cudaMemcpy(Q.data(), B_vp, n_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q.data() + n_model, B_eps, n_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q.data() + 2*n_model, B_delta, n_model * sizeof(float), cudaMemcpyDeviceToHost);

    const int s_vp = 1.0;
    const int s_eps = 2.0;
    const int s_delta = 8.0;

    #pragma omp parallel for
    for (int i = 0; i < 3 * n_model; ++i){
        Q[i] =  1.0f / (s_m2 * Q[i] + reg_m2);
        Q[n_model + i] = 1.0f / (s_eps * Q[n_model + i] + reg_eps);
        Q[2*n_model + i] = 1.0f / (s_delta * Q[2*n_model + i] + reg_delta);
        p[i] = Q[i] * grad[i];
        x[i] = -p[i];
    }
}

void Inversion::setBornPerturbation(const std::vector<float>& x){
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;

    float* temp = new float[n_model_exp];
    mdl->expandModel(x.data(), temp);
    cudaMemcpy(dvp, temp, n_model_exp*sizeof(float), cudaMemcpyHostToDevice);

    mdl->expandModel(x.data() + n_model, temp);
    cudaMemcpy(dvp, temp, n_model_exp*sizeof(float), cudaMemcpyHostToDevice);

    mdl->expandModel(x.data() + 2*n_model, temp);
    cudaMemcpy(dvp, temp, n_model_exp*sizeof(float), cudaMemcpyHostToDevice);
    delete[] temp;
}

void Inversion::solveGaussNewton(const float X0){
    float X = X0; 
    while (X > e){
        cudaMemset(B_vp,    0, n_model*sizeof(float));
        cudaMemset(B_eps,   0, n_model*sizeof(float));
        cudaMemset(B_delta, 0, n_model*sizeof(float));

        X = calculateMultiparameterGradient(true,true,true, grad, true);
        setTGN();
        while(condition){
            setBornPerturbation(x);
            computeHessianVectorProduct();
            cudaMemcpy(Hx.data(), mgt->image, n_model*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(Hx.data() + n_model, eps_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(Hx.data() + 2*n_model, delta_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
            
            beta1 = dot_vector(x, Hx);
            if(beta1 <= 0.0) break;
            beta2 = dot_vector(grad, p);
            for (int i = 0; i < 3 * n_model; ++i) {
                dm[i] += (beta2/beta1) * x[i];
                grad[i] += (beta2/beta1) * Hx[i];
                p[i] = Q[i] * grad[i];
            }
        
            beta2_new = dot(grad,p);
            for (int i = 0; i < 3 * n_model; ++i) {
                x[i] = -p[i] + (beta2_new/beta2) * x[i];
            }  
        }
        linesearchMultiparameter
        applyMultiparameterStep
    }
}

void Inversion::solveFullWaveformInversionMultiparameterHierarchical(){
    std::cout << "info: Solving hierarchical multiparameter FWI" << std::endl;

    if(pmt->approximation != "VTI" && pmt->approximation != "TTI"){
        throw std::runtime_error("Hierarchical multiparameter FWI requires VTI or TTI.");
    }

    if(!pmt->multiparameter){
        throw std::runtime_error("Hierarchical multiparameter FWI requires multiparameter=true.");
    }

    const int n_model = pmt->nx*pmt->nz;
    const float eps_start = 0.50f;
    const float delta_start = 0.75f;
    const float theta_start = 0.85f;

    const int eps_first_itr = 1 + static_cast<int>(std::ceil(eps_start*(pmt->niter - 1)));
    const int delta_first_itr = 1 + static_cast<int>(std::ceil(delta_start*(pmt->niter - 1)));
    const int theta_first_itr = 1 + static_cast<int>(std::ceil(theta_start*(pmt->niter - 1)));

    setModel();

    float* final_model = new float[n_model];

    const std::string history_file = "../outputs/history.txt";
    std::ofstream history_stream(history_file);

    mdl->createCerjanVector();
    cudaMemset(B_vp,    0, n_model * sizeof(float));
    cudaMemset(B_eps,   0, n_model * sizeof(float));
    cudaMemset(B_delta, 0, n_model * sizeof(float));
    for(const float fmax : pmt->freqs){
        std::cout << std::defaultfloat << "info: FWI frequency " << fmax << std::endl;

        adjustfmax(fmax);

        s_vp_store.clear();
        y_vp_store.clear();

        s_eps_store.clear();
        y_eps_store.clear();

        s_delta_store.clear();
        y_delta_store.clear();

        s_theta_store.clear();
        y_theta_store.clear();

        float X_current = calculateMultiparameterGradient(false,false,false,grad_vp_h,grad_eps_h,grad_delta_h,grad_theta_h);
        const float g_vp_max0 = getGradientScale(grad_vp_h);
        scaleGradient(grad_vp_h,g_vp_max0);

        float g_eps_max0 = 1.0f;
        float g_delta_max0 = 1.0f;
        float g_theta_max0 = 1.0f;

        const float X_freq0 = X_current;

        history_stream << X_current/X_freq0 << " " << fmax << std::endl;

        for(int itr = 0; itr < pmt->niter; itr++){
            const int iteration = itr + 1;

            std::cout << std::defaultfloat << "info: FWI iteration " << iteration << "/" << pmt->niter << " for frequency " << fmax << std::endl;

            std::ostringstream fcut_stream;
            fcut_stream << std::fixed << std::setprecision(1) << fmax;

            const bool update_eps = iteration >= eps_first_itr;
            const bool update_delta = iteration >= delta_first_itr;
            const bool update_theta = pmt->approximation == "TTI" && iteration >= theta_first_itr;

            std::cout << "info: Hierarchical stage" << std::endl;
            std::cout << "Vp update: true" << std::endl;
            std::cout << "Epsilon update: " << update_eps << std::endl;
            std::cout << "Delta update: " << update_delta << std::endl;

            if(pmt->approximation == "TTI"){
                std::cout << "Theta update: " << update_theta << std::endl;
            }

            if(iteration == eps_first_itr || iteration == delta_first_itr || (pmt->approximation == "TTI" && iteration == theta_first_itr)){
                X_current = calculateMultiparameterGradient(update_eps,update_delta,update_theta,grad_vp_h,grad_eps_h,grad_delta_h,grad_theta_h);

                scaleGradient(grad_vp_h,g_vp_max0);

                if(iteration == eps_first_itr){
                    g_eps_max0 = getGradientScale(grad_eps_h);
                }

                if(iteration == delta_first_itr){
                    g_delta_max0 = getGradientScale(grad_delta_h);
                }

                if(pmt->approximation == "TTI" && iteration == theta_first_itr){
                    g_theta_max0 = getGradientScale(grad_theta_h);
                }

                if(update_eps){
                    scaleGradient(grad_eps_h,g_eps_max0);
                }

                if(update_delta){
                    scaleGradient(grad_delta_h,g_delta_max0);
                }

                if(update_theta){
                    scaleGradient(grad_theta_h,g_theta_max0);
                }
            }

            const std::string gradient_vp_file = pmt->gradientsFolder+"vp_gradient_fwi_iter_"+std::to_string(iteration)+"_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_freq"+fcut_stream.str()+".bin";
            saveModel(gradient_vp_file,grad_vp_h);

            if(update_eps){
                const std::string gradient_eps_file = pmt->gradientsFolder+"epsilon_gradient_fwi_iter_"+std::to_string(iteration)+"_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_freq"+fcut_stream.str()+".bin";
                saveModel(gradient_eps_file,grad_eps_h);
            }

            if(update_delta){
                const std::string gradient_delta_file = pmt->gradientsFolder+"delta_gradient_fwi_iter_"+std::to_string(iteration)+"_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_freq"+fcut_stream.str()+".bin";
                saveModel(gradient_delta_file,grad_delta_h);
            }

            if(update_theta){
                const std::string gradient_theta_file = pmt->gradientsFolder+"theta_gradient_fwi_iter_"+std::to_string(iteration)+"_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_freq"+fcut_stream.str()+".bin";
                saveModel(gradient_theta_file,grad_theta_h);
            }

            twoLoopRecursion(grad_vp_h,p_vp,s_vp_store,y_vp_store);

            if(update_eps){
                twoLoopRecursion(grad_eps_h,p_eps,s_eps_store,y_eps_store);
            }

            if(update_delta){
                twoLoopRecursion(grad_delta_h,p_delta,s_delta_store,y_delta_store);
            }

            if(update_theta){
                twoLoopRecursion(grad_theta_h,p_theta,s_theta_store,y_theta_store);
            }

            const float beta_vp = armijolinesearch("vp",vp_h,grad_vp_h,p_vp,X_current,s_vp_store.empty(),g_vp_max0);

            float beta_eps = 0.0f;
            float beta_delta = 0.0f;
            float beta_theta = 0.0f;

            if(update_eps){
                beta_eps = armijolinesearch("epsilon",eps_h,grad_eps_h,p_eps,X_current,s_eps_store.empty(),g_eps_max0);
            }

            if(update_delta){
                beta_delta = armijolinesearch("delta",delta_h,grad_delta_h,p_delta,X_current,s_delta_store.empty(),g_delta_max0);
            }

            if(update_theta){
                beta_theta = armijolinesearch("theta",theta_h,grad_theta_h,p_theta,X_current,s_theta_store.empty(),g_theta_max0);
            }

            if(beta_vp == 0.0f && beta_eps == 0.0f && beta_delta == 0.0f && beta_theta == 0.0f){
                std::cout << "warning: All parameter Armijo searches failed. Stopping this frequency." << std::endl;
                break;
            }

            float alpha = 1.0f;

            if(update_eps || update_delta || update_theta){
                alpha = linesearchMultiparameter(vp_h,eps_h,delta_h,theta_h,p_vp,p_eps,p_delta,p_theta,grad_vp_h,grad_eps_h,grad_delta_h,grad_theta_h,beta_vp,beta_eps,beta_delta,beta_theta,g_vp_max0,g_eps_max0,g_delta_max0,g_theta_max0,update_eps,update_delta,update_theta,X_current);
            }

            if(alpha == 0.0f){
                std::cout << "warning: Multiparameter Armijo failed. Stopping this frequency." << std::endl;
                break;
            }

            applyMultiparameterStep(update_eps,update_delta,update_theta,vp_h,eps_h,delta_h,theta_h,p_vp,p_eps,p_delta,p_theta,vpnew_h,epsnew_h,deltanew_h,thetanew_h,alpha,beta_vp,beta_eps,beta_delta,beta_theta);

            ExpandModelDevice("vp",vpnew_h);

            if(update_eps){
                ExpandModelDevice("epsilon",epsnew_h);
            }

            if(update_delta){
                ExpandModelDevice("delta",deltanew_h);
            }

            if(update_theta){
                ExpandModelDevice("theta",thetanew_h);
            }

            const float X_new = calculateMultiparameterGradient(update_eps,update_delta,update_theta,grad_vpnew_h,grad_epsnew_h,grad_deltanew_h,grad_thetanew_h);

            scaleGradient(grad_vpnew_h,g_vp_max0);

            if(update_eps){
                scaleGradient(grad_epsnew_h,g_eps_max0);
            }

            if(update_delta){
                scaleGradient(grad_deltanew_h,g_delta_max0);
            }

            if(update_theta){
                scaleGradient(grad_thetanew_h,g_theta_max0);
            }

            updateLBFGSHistory(vp_h,vpnew_h,grad_vp_h,grad_vpnew_h,s_vp_store,y_vp_store);

            if(update_eps){
                updateLBFGSHistory(eps_h,epsnew_h,grad_eps_h,grad_epsnew_h,s_eps_store,y_eps_store);
            }

            if(update_delta){
                updateLBFGSHistory(delta_h,deltanew_h,grad_delta_h,grad_deltanew_h,s_delta_store,y_delta_store);
            }

            if(update_theta){
                updateLBFGSHistory(theta_h,thetanew_h,grad_theta_h,grad_thetanew_h,s_theta_store,y_theta_store);
            }

            std::swap(vp_h,vpnew_h);
            std::swap(grad_vp_h,grad_vpnew_h);

            if(update_eps){
                std::swap(eps_h,epsnew_h);
                std::swap(grad_eps_h,grad_epsnew_h);
            }

            if(update_delta){
                std::swap(delta_h,deltanew_h);
                std::swap(grad_delta_h,grad_deltanew_h);
            }

            if(update_theta){
                std::swap(theta_h,thetanew_h);
                std::swap(grad_theta_h,grad_thetanew_h);
            }

            X_current = X_new;

            history_stream << X_current/X_freq0 << " " << fmax << std::endl;

            #pragma omp parallel for
            for(int i = 0; i < n_model; i++){
                final_model[i] = 1.0f/std::sqrt(vp_h[i]);
            }

            const std::string vp_model_file = pmt->estimatedmodelsFolder+"fwi_vp_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_itr"+std::to_string(iteration)+"_freq"+fcut_stream.str()+".bin";
            saveModel(vp_model_file,final_model);

            if(update_eps){
                const std::string eps_model_file = pmt->estimatedmodelsFolder+"fwi_epsilon_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_itr"+std::to_string(iteration)+"_freq"+fcut_stream.str()+".bin";
                saveModel(eps_model_file,eps_h);
            }

            if(update_delta){
                const std::string delta_model_file = pmt->estimatedmodelsFolder+"fwi_delta_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_itr"+std::to_string(iteration)+"_freq"+fcut_stream.str()+".bin";
                saveModel(delta_model_file,delta_h);
            }

            if(update_theta){
                float* theta_deg = new float[n_model];

                const float rad2deg = 180.0f / 3.14159265358979323846f;

                #pragma omp parallel for
                for(int i = 0; i < n_model; i++){
                    theta_deg[i] = theta_h[i] * rad2deg;
                }
                const std::string theta_model_file = pmt->estimatedmodelsFolder+"fwi_theta_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_itr"+std::to_string(iteration)+"_freq"+fcut_stream.str()+".bin";
                saveModel(theta_model_file,theta_deg);
                delete[] theta_deg;
            }
        }
    }

    delete[] final_model;
    history_stream.close();

    std::cout << "info: FWI history saved to " << history_file << std::endl;
}

__global__ void computeObjectiveFunction(float* X, float* __restrict__ residual, const float* __restrict__ calculated,const int nt, const int Nrec){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_seis = nt * Nrec;

    if (i < n_seis){
        const float r = residual[i] - calculated[i];
        residual[i] = r;
        atomicAdd(X, 0.5f * static_cast<float>(r) * static_cast<float>(r));
    }
}

__global__ void slowness2ToVp(const float* __restrict__ slowness2,float* __restrict__ vp, int nx, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_model = nx * nz; 
    if (i < n_model){
        vp[i] = rsqrtf(slowness2[i]);
    }
}

__global__ void updateAdjointWaveEquationandGradient(float* __restrict__ Uf, float* __restrict__ Uc, const float* __restrict__ Pp, const float* __restrict__ Pc, const float* __restrict__ Pf,
float* __restrict__ vp_grad, const float* __restrict__ vp, const int nz, const int nx, const float dz, const float dx, const float dt, const float* __restrict__ A, const int N_abc){
    const float c0 = -2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;

    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);

    const float dt2 = dt * dt;
    const float inv_dt2 = 1.0f / dt2;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_size = nx * nz;

    if (i >= total_size)
        return;

    const int iz = i / nx;
    const int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) 
    {
        const float vp2 = vp[i]*vp[i];

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) * inv_dx2;

        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx] + Uc[i - nx])
            + c2 * (Uc[i + 2*nx] + Uc[i - 2*nx])
            + c3 * (Uc[i + 3*nx] + Uc[i - 3*nx])
            + c4 * (Uc[i + 4*nx] + Uc[i - 4*nx])) * inv_dz2;

        Uf[i] = vp2 * dt2 * (pxx + pzz) + 2.0f * Uc[i] - Uf[i];

        if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc){
            const int xf = ix - N_abc;
            const int zf = iz - N_abc;

            const int nxf = nx - 2 * N_abc;
            const int idx = zf * nxf + xf;

            const float ptt =(Pf[i] - 2.0f * Pc[i] + Pp[i]) * inv_dt2;
            vp_grad[idx] += Uc[i] * ptt;
        }   

        if (ix < N_abc){
            Uf[i] *= A[ix];
            Uc[i] *= A[ix];
        }
        if (ix >=  nx - N_abc){
            Uf[i] *= A[nx - 1 - ix];
            Uc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc){
            Uf[i] *= A[iz];
            Uc[i] *= A[iz];
        }
        if (iz >= nz - N_abc){
            Uf[i] *= A[nz - 1 - iz];
            Uc[i] *= A[nz - 1 - iz];
        }
    }
}

__global__ void calculateAdjointVTIProductsAndGradients(const float* __restrict__ Uc, const float* __restrict__ Pp, const float* __restrict__ Pc, const float* __restrict__ Pf, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc,
float* __restrict__ vp_grad, float* __restrict__ eps_grad, float* __restrict__ delta_grad,const float* __restrict__ vp, const float* __restrict__ epsilon, const float* __restrict__ delta, const float dt, const float dx, const float dz,
const int nx, const int nz, const int N_abc, const bool multiparameter, float* __restrict__ B_m, float* __restrict__ B_eps, float* __restrict__ B_delta, const bool TGN){

    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;
    const float inv_dx  = 1.0f / dx;
    const float inv_dz  = 1.0f / dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dt2 = 1.0f / (dt * dt);
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = nx * nz;

    if (i >= total_size)
        return;

    const int iz = i / nx;
    const int ix = i % nx;

    AUc[i]   = 0.0f;
    BUc[i]   = 0.0f;
    QCxUc[i] = 0.0f;
    QCzUc[i] = 0.0f;


    if (ix < 4 || ix >= nx - 4 ||
        iz < 4 || iz >= nz - 4)
    {
        return;
    }
    
    const float pxx =(c0 * Pc[i]
            + c1 * (Pc[i + 1] + Pc[i - 1])
            + c2 * (Pc[i + 2] + Pc[i - 2])
            + c3 * (Pc[i + 3] + Pc[i - 3])
            + c4 * (Pc[i + 4] + Pc[i - 4])) * inv_dx2;

    const float pzz =(c0 * Pc[i]
            + c1 * (Pc[i + nx] + Pc[i - nx])
            + c2 * (Pc[i + 2 * nx] + Pc[i - 2 * nx])
            + c3 * (Pc[i + 3 * nx] + Pc[i - 3 * nx])
            + c4 * (Pc[i + 4 * nx] + Pc[i - 4 * nx])) * inv_dz2;

    const float px =(a1 * (Pc[i + 1] - Pc[i - 1])
            + a2 * (Pc[i + 2] - Pc[i - 2])
            + a3 * (Pc[i + 3] - Pc[i - 3])
            + a4 * (Pc[i + 4] - Pc[i - 4])) * inv_dx;

    const float pz =(a1 * (Pc[i + nx] - Pc[i - nx])
            + a2 * (Pc[i + 2*nx] - Pc[i - 2*nx])
            + a3 * (Pc[i + 3*nx] - Pc[i - 3*nx])
            + a4 * (Pc[i + 4*nx] - Pc[i - 4*nx])) * inv_dz;

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
    if(multiparameter){
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

    if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc){
        const int xf = ix - N_abc;
        const int zf = iz - N_abc;

        const int nxf = nx - 2 * N_abc;
        const int idx = zf * nxf + xf;

        const float ptt =(Pf[i] - 2.0f * Pc[i] + Pp[i]) * inv_dt2;
        vp_grad[idx] += adj * ptt;

        if (multiparameter){
            const float dP_deps = (-2.0f - dSd_deps) * pxx - dSd_deps * pzz;
            const float dP_ddelta = -dSd_ddelta * Q;

            eps_grad[idx] += adj * dP_deps;
            delta_grad[idx] += adj * dP_ddelta;

            if(TGN){
                B_vp[idx] += ptt * ptt * dt;
                B_eps[idx] += dP_deps * dP_deps * dt;
                B_delta[idx] += dP_ddelta * dP_ddelta * dt;
            }   
        }
    }
}

__global__ void updateAdjointWaveEquationVTI(float* __restrict__ Uf, float* __restrict__ Uc, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp, float* __restrict__ A, int N_abc)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;

    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
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


    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4)
    {
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


        const float dx_QCxUc = (a1 * (QCxUc[i + 1] - QCxUc[i - 1])
                + a2 * (QCxUc[i + 2] - QCxUc[i - 2])
                + a3 * (QCxUc[i + 3] - QCxUc[i - 3])
                + a4 * (QCxUc[i + 4] - QCxUc[i - 4])) * inv_dx;

        const float dz_QCzUc =(a1 * (QCzUc[i + nx] - QCzUc[i - nx])
                + a2 * (QCzUc[i + 2*nx] - QCzUc[i - 2*nx])
                + a3 * (QCzUc[i + 3*nx] - QCzUc[i - 3*nx])
                + a4 * (QCzUc[i + 4*nx] - QCzUc[i - 4*nx])) * inv_dz;

        const float spatial_operator = dxx_AUc + dzz_BUc - dx_QCxUc - dz_QCzUc;
        const float vp2dt2 = vp[i] * vp[i] * dt * dt;
        Uf[i] = 2.0f * Uc[i] - Uf[i] + vp2dt2 * spatial_operator;

        if (ix < N_abc){
            Uf[i] *= A[ix];
            Uc[i] *= A[ix];
        }
        if (ix >=  nx - N_abc){
            Uf[i] *= A[nx - 1 - ix];
            Uc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc){
            Uf[i] *= A[iz];
            Uc[i] *= A[iz];
        }
        if (iz >= nz - N_abc){
            Uf[i] *= A[nz - 1 - iz];
            Uc[i] *= A[nz - 1 - iz];
        }
    }
}

__global__ void calculateAdjointTTIProductsAndGradients(const float* __restrict__ Uc, const float* __restrict__ Pp, const float* __restrict__ Pc, const float* __restrict__ Pf, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ HUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc,
float* __restrict__ vp_grad, float* __restrict__ eps_grad, float* __restrict__ delta_grad, float* __restrict__ theta_grad, const float* __restrict__ epsilon, const float* __restrict__ delta, const float* __restrict__ theta,
const float dt, const float dx, const float dz, const int nx, const int nz, const int N_abc, const bool multiparameter){ 

    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;
    const float inv_dx  = 1.0f / dx;
    const float inv_dz  = 1.0f / dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dxdz = 1.0f / (dx * dz);
    const float inv_dt2 = 1.0f / (dt * dt);

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size =nx * nz;

    if (i >= total_size)
        return;

    const int iz = i / nx;
    const int ix = i % nx;

    AUc[i]   = 0.0f;
    BUc[i]   = 0.0f;
    HUc[i]   = 0.0f;
    QCxUc[i] = 0.0f;
    QCzUc[i] = 0.0f;

    if (ix < 4 || ix >= nx - 4 || iz < 4 || iz >= nz - 4)
    {
        return;
    }

    const float pxx =(c0 * Pc[i]
            + c1 * (Pc[i + 1] + Pc[i - 1])
            + c2 * (Pc[i + 2] + Pc[i - 2])
            + c3 * (Pc[i + 3] + Pc[i - 3])
            + c4 * (Pc[i + 4] + Pc[i - 4])) * inv_dx2;

    const float pzz =(c0 * Pc[i]
            + c1 * (Pc[i + nx] + Pc[i - nx])
            + c2 * (Pc[i + 2 * nx] + Pc[i - 2 * nx])
            + c3 * (Pc[i + 3 * nx] + Pc[i - 3 * nx])
            + c4 * (Pc[i + 4 * nx] + Pc[i - 4 * nx])) * inv_dz2;

    const float px =(a1 * (Pc[i + 1] - Pc[i - 1])
            + a2 * (Pc[i + 2] - Pc[i - 2])
            + a3 * (Pc[i + 3] - Pc[i - 3])
            + a4 * (Pc[i + 4] - Pc[i - 4])) * inv_dx;

    const float pz =(a1 * (Pc[i + nx] - Pc[i - nx])
            + a2 * (Pc[i + 2*nx] - Pc[i - 2*nx])
            + a3 * (Pc[i + 3*nx] - Pc[i - 3*nx])
            + a4 * (Pc[i + 4*nx] - Pc[i - 4*nx])) * inv_dz;


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

    const float num = -2.0f * (eps - del) * xi2* eta2;
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

    float dSd_deps   = 0.0f;
    float dSd_ddelta = 0.0f;
    float dSd_dtheta = 0.0f;
    if (multiparameter)
    {
        const float dnum_deps = -2.0f * xi2eta2;
        const float dnum_ddelta =  2.0f * xi2eta2;
        const float dden_deps =  2.0f * xi4;
        const float dden_ddelta =  2.0f * xi2eta2;
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

    const float Acoef = (1.0f + 2.0f * eps) * cos2 + sin2 + Sd;
    const float Bcoef = (1.0f + 2.0f * eps) * sin2 + cos2 + Sd;
    const float Hcoef = 2.0f * eps * sin2th;
    const float Q = pxx + pzz;
    const float adj = Uc[i];
    AUc[i] = Acoef * adj;
    BUc[i] = Bcoef * adj;
    HUc[i] = Hcoef * adj;
    QCxUc[i] = Q * Cx * adj;
    QCzUc[i] = Q * Cz * adj;

    if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc)
    { 
        const int xf = ix - N_abc;
        const int zf = iz - N_abc;
        const int nxf = nx - 2 * N_abc;
        const int idx = zf * nxf + xf;

        const float ptt = (Pf[i] - 2.0f * Pc[i] + Pp[i]) * inv_dt2;
        vp_grad[idx] += adj * ptt;

        if (multiparameter)
        {
            const float pxz = (
            a1*a1*(Pc[i + nx + 1]     - Pc[i - nx + 1]     + Pc[i - nx - 1]     - Pc[i + nx - 1]) +
            a1*a2*(Pc[i + 2*nx + 1]   - Pc[i - 2*nx + 1]   + Pc[i - 2*nx - 1]   - Pc[i + 2*nx - 1]) +
            a1*a3*(Pc[i + 3*nx + 1]   - Pc[i - 3*nx + 1]   + Pc[i - 3*nx - 1]   - Pc[i + 3*nx - 1]) +
            a1*a4*(Pc[i + 4*nx + 1]   - Pc[i - 4*nx + 1]   + Pc[i - 4*nx - 1]   - Pc[i + 4*nx - 1]) +

            a2*a1*(Pc[i + nx + 2]     - Pc[i - nx + 2]     + Pc[i - nx - 2]     - Pc[i + nx - 2]) +
            a2*a2*(Pc[i + 2*nx + 2]   - Pc[i - 2*nx + 2]   + Pc[i - 2*nx - 2]   - Pc[i + 2*nx - 2]) +
            a2*a3*(Pc[i + 3*nx + 2]   - Pc[i - 3*nx + 2]   + Pc[i - 3*nx - 2]   - Pc[i + 3*nx - 2]) +
            a2*a4*(Pc[i + 4*nx + 2]   - Pc[i - 4*nx + 2]   + Pc[i - 4*nx - 2]   - Pc[i + 4*nx - 2]) +

            a3*a1*(Pc[i + nx + 3]     - Pc[i - nx + 3]     + Pc[i - nx - 3]     - Pc[i + nx - 3]) +
            a3*a2*(Pc[i + 2*nx + 3]   - Pc[i - 2*nx + 3]   + Pc[i - 2*nx - 3]   - Pc[i + 2*nx - 3]) +
            a3*a3*(Pc[i + 3*nx + 3]   - Pc[i - 3*nx + 3]   + Pc[i - 3*nx - 3]   - Pc[i + 3*nx - 3]) +
            a3*a4*(Pc[i + 4*nx + 3]   - Pc[i - 4*nx + 3]   + Pc[i - 4*nx - 3]   - Pc[i + 4*nx - 3]) +

            a4*a1*(Pc[i + nx + 4]     - Pc[i - nx + 4]     + Pc[i - nx - 4]     - Pc[i + nx - 4]) +
            a4*a2*(Pc[i + 2*nx + 4]   - Pc[i - 2*nx + 4]   + Pc[i - 2*nx - 4]   - Pc[i + 2*nx - 4]) +
            a4*a3*(Pc[i + 3*nx + 4]   - Pc[i - 3*nx + 4]   + Pc[i - 3*nx - 4]   - Pc[i + 3*nx - 4]) +
            a4*a4*(Pc[i + 4*nx + 4]   - Pc[i - 4*nx + 4]   + Pc[i - 4*nx - 4]   - Pc[i + 4*nx - 4])) * inv_dxdz;


            const float dA_dtheta = 2.0f * eps * sin2th - dSd_dtheta;
            const float dB_dtheta = -2.0f * eps * sin2th - dSd_dtheta;
            const float dC_dtheta = 4.0f * eps * cos2th;
            const float dP_deps = -(2.0f * cos2 + dSd_deps) * pxx-(2.0f * sin2 + dSd_deps) * pzz + 2.0f * sin2th * pxz;
            const float dP_ddelta = -dSd_ddelta * Q;
            const float dP_dtheta = dA_dtheta * pxx + dB_dtheta * pzz + dC_dtheta * pxz;

            eps_grad[idx] += adj * dP_deps;
            delta_grad[idx] += adj * dP_ddelta;
            theta_grad[idx] += adj * dP_dtheta;
        }  
    }
}

__global__ void updateAdjointWaveEquationTTI(float* __restrict__ Uf, float* __restrict__ Uc, const float* __restrict__ AUc, const float* __restrict__ BUc, const float* __restrict__ HUc, const float* __restrict__ QCxUc, const float* __restrict__ QCzUc, const int nx, const int nz, const float dt, const float dx, const float dz, const float* __restrict__ vp, float* __restrict__ A, int N_abc)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;

    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
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

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4)
    {
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
    
        const float dxz_HUc = (
                a1*a1*(HUc[i + nx + 1]     - HUc[i - nx + 1]     + HUc[i - nx - 1]     - HUc[i + nx - 1]) +
                a1*a2*(HUc[i + 2*nx + 1]   - HUc[i - 2*nx + 1]   + HUc[i - 2*nx - 1]   - HUc[i + 2*nx - 1]) +
                a1*a3*(HUc[i + 3*nx + 1]   - HUc[i - 3*nx + 1]   + HUc[i - 3*nx - 1]   - HUc[i + 3*nx - 1]) +
                a1*a4*(HUc[i + 4*nx + 1]   - HUc[i - 4*nx + 1]   + HUc[i - 4*nx - 1]   - HUc[i + 4*nx - 1]) +

                a2*a1*(HUc[i + nx + 2]     - HUc[i - nx + 2]     + HUc[i - nx - 2]     - HUc[i + nx - 2]) +
                a2*a2*(HUc[i + 2*nx + 2]   - HUc[i - 2*nx + 2]   + HUc[i - 2*nx - 2]   - HUc[i + 2*nx - 2]) +
                a2*a3*(HUc[i + 3*nx + 2]   - HUc[i - 3*nx + 2]   + HUc[i - 3*nx - 2]   - HUc[i + 3*nx - 2]) +
                a2*a4*(HUc[i + 4*nx + 2]   - HUc[i - 4*nx + 2]   + HUc[i - 4*nx - 2]   - HUc[i + 4*nx - 2]) +

                a3*a1*(HUc[i + nx + 3]     - HUc[i - nx + 3]     + HUc[i - nx - 3]     - HUc[i + nx - 3]) +
                a3*a2*(HUc[i + 2*nx + 3]   - HUc[i - 2*nx + 3]   + HUc[i - 2*nx - 3]   - HUc[i + 2*nx - 3]) +
                a3*a3*(HUc[i + 3*nx + 3]   - HUc[i - 3*nx + 3]   + HUc[i - 3*nx - 3]   - HUc[i + 3*nx - 3]) +
                a3*a4*(HUc[i + 4*nx + 3]   - HUc[i - 4*nx + 3]   + HUc[i - 4*nx - 3]   - HUc[i + 4*nx - 3]) +

                a4*a1*(HUc[i + nx + 4]     - HUc[i - nx + 4]     + HUc[i - nx - 4]     - HUc[i + nx - 4]) +
                a4*a2*(HUc[i + 2*nx + 4]   - HUc[i - 2*nx + 4]   + HUc[i - 2*nx - 4]   - HUc[i + 2*nx - 4]) +
                a4*a3*(HUc[i + 3*nx + 4]   - HUc[i - 3*nx + 4]   + HUc[i - 3*nx - 4]   - HUc[i + 3*nx - 4]) +
                a4*a4*(HUc[i + 4*nx + 4]   - HUc[i - 4*nx + 4]   + HUc[i - 4*nx - 4]   - HUc[i + 4*nx - 4])) * inv_dxdz;

        const float dx_QCxUc = (a1 * (QCxUc[i + 1] - QCxUc[i - 1])
                + a2 * (QCxUc[i + 2] - QCxUc[i - 2])
                + a3 * (QCxUc[i + 3] - QCxUc[i - 3])
                + a4 * (QCxUc[i + 4] - QCxUc[i - 4])) * inv_dx;


        const float dz_QCzUc =(a1 * (QCzUc[i + nx] - QCzUc[i - nx])
                + a2 * (QCzUc[i + 2*nx] - QCzUc[i - 2*nx])
                + a3 * (QCzUc[i + 3*nx] - QCzUc[i - 3*nx])
                + a4 * (QCzUc[i + 4*nx] - QCzUc[i - 4*nx])) * inv_dz;

        const float spatial_operator = dxx_AUc + dzz_BUc - dxz_HUc - dx_QCxUc - dz_QCzUc;
        const float vp2dt2 = vp[i] * vp[i] * dt * dt;

        Uf[i] = 2.0f * Uc[i] - Uf[i] + vp2dt2 * spatial_operator;

        if (ix < N_abc){
            Uf[i] *= A[ix];
            Uc[i] *= A[ix];
        }
        if (ix >=  nx - N_abc){
            Uf[i] *= A[nx - 1 - ix];
            Uc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc){
            Uf[i] *= A[iz];
            Uc[i] *= A[iz];
        }
        if (iz >= nz - N_abc){
            Uf[i] *= A[nz - 1 - iz];
            Uc[i] *= A[nz - 1 - iz];
        }
    }
}

__global__ void updateWaveEquationBorn(float* __restrict__ dUf,  float* __restrict__ dUc, float* __restrict__ U0f, float* __restrict__ U0c, const float* __restrict__ vp, const float* __restrict__ dvp, int nz, int nx, float dz, float dx, float dt, float* __restrict__ A, int N_abc){
    const float c0 = -2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;

    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);
    const float dt2 = dt * dt;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz * nx) return;

    int iz = i/nx;
    int ix = i%nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4){
        
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

        float vp2 = vp[i] * vp[i];
        float vp4 = vp2 * vp2;
        float propagation = vp2 * (du_xx + du_zz);
        
        float U0p = U0f[i];
        U0f[i] = vp2 * dt2 * (u0_xx + u0_zz) + 2.0f * U0c[i] - U0f[i];
        const float u0_tt = (U0f[i] - 2.0f * U0c[i] + U0p) * inv_dt2;
        float born_source = -vp2 * dvp[i] * u0_tt;
        dUf[i] = 2.0f * dUc[i] - dUf[i] + dt2 * (propagation + born_source);

        if (ix < N_abc){
            U0f[i] *= A[ix];
            U0c[i] *= A[ix];
            dUf[i] *= A[ix];
            dUc[i] *= A[ix];
        }
        if (ix >=  nx - N_abc){
            U0f[i] *= A[nx - 1 - ix];
            U0c[i] *= A[nx - 1 - ix];
            dUf[i] *= A[nx - 1 - ix];
            dUc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc){
            U0f[i] *= A[iz];
            U0c[i] *= A[iz];
            dUf[i] *= A[iz];
            dUc[i] *= A[iz];
        }
        if (iz >= nz - N_abc){
            U0f[i] *= A[nz - 1 - iz];
            U0c[i] *= A[nz - 1 - iz];
            dUf[i] *= A[nz - 1 - iz];
            dUc[i] *= A[nz - 1 - iz];
        }
    }
}

__global__ void updateWaveEquationVTIBorn(float* __restrict__ dUf, float* __restrict__ dUc, float* __restrict__ U0f, float* __restrict__ U0c, const float* __restrict__ vp, const float* __restrict__ epsilon, const float* __restrict__ delta, const float* __restrict__ dvp, const float* __restrict__ deps, const float* __restrict__ ddelta, int nz, int nx, float dz, float dx, float dt, float* __restrict__ A, int N_abc){
    const float c0 = -2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;
    const float a1 =  0.8f;
    const float a2 = -0.2f;
    const float a3 =  0.03809523809f;
    const float a4 = -0.00357142857f;

    const float inv_dx  = 1.0f / dx;
    const float inv_dz  = 1.0f / dz;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);
    const float dt2 = dt * dt;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz * nx) return;

    int iz = i / nx;
    int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4){
        
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
        float u0_x = (a1*(U0c[i+1] - U0c[i-1]) +
                    a2*(U0c[i+2] - U0c[i-2]) +
                    a3*(U0c[i+3] - U0c[i-3]) +
                    a4*(U0c[i+4] - U0c[i-4])) * inv_dx;

        float u0_z = (a1 * (U0c[i + nx] - U0c[i - nx]) +
                    a2 * (U0c[i + 2*nx] - U0c[i - 2*nx]) +
                    a3 * (U0c[i + 3*nx] - U0c[i - 3*nx]) +
                    a4 * (U0c[i + 4*nx] - U0c[i - 4*nx])) * inv_dz;

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
        float du_x = (a1*(dUc[i+1] - dUc[i-1]) +
                    a2*(dUc[i+2] - dUc[i-2]) +
                    a3*(dUc[i+3] - dUc[i-3]) +
                    a4*(dUc[i+4] - dUc[i-4])) * inv_dx;
        float du_z = (a1 * (dUc[i + nx] - dUc[i - nx]) +
                    a2 * (dUc[i + 2*nx] - dUc[i - 2*nx]) +
                    a3 * (dUc[i + 3*nx] - dUc[i - 3*nx]) +
                    a4 * (dUc[i + 4*nx] - dUc[i - 4*nx])) * inv_dz;;

        float eps  = epsilon[i];
        float del  = delta[i];
        float deps = deps[i];
        float ddel = ddelta[i];

        float x2 = u0_x * u0_x;
        float z2 = u0_z * u0_z;
        float x4 = x2 * x2;
        float z4 = z2 * z2;
        float x2z2 = x2 * z2;


        float num = -2.0f * (eps - del) * x2z2;
        float den = (1.0f + 2.0f * eps) * x4 + z4 + 2.0f * (1.0f + del) * x2z2 + 1e-37f;
        float Sd = num/den;

        float dnum_u = -4.0f * (eps - del) * (u0_x * z2 * du_x + x2 * u0_z * du_z);
        float dden_u = 4.0f * (1.0f + 2.0f * eps) * x2 * u0_x * du_x + 4.0f * z2 * u0_z * du_z + 4.0f * (1.0f + del) * (u0_x * z2 * du_x + x2 * u0_z * du_z);
        float dSd_u = (dnum_u - Sd * dden_u) / den;

        float dnum_m = -2.0f * (deps - ddel) * x2z2;
        float dden_m = 2.0f * deps * x4+ 2.0f * ddel * x2z2;
        float dSd_m = (dnum_m - Sd * dden_m) / den;

        float vp2 = vp[i] * vp[i];
        float vp4 = vp2 * vp2;

        float A0 = 1.0f + 2.0f * eps + Sd;
        float B0 = 1.0f + Sd;
        float H0 = A0 * u0_xx + B0 * u0_zz;
        float propagation = vp2 * (A0 * du_xx + B0 * du_zz + dSd_u * (u0_xx + u0_zz));
        
        float U0p = U0f[i];
        U0f[i] = 2.0f * U0c[i] - U0f[i] + vp2 * dt2 * ((1.0f+ 2.0f*eps) + Sd) * u0_xx + vp2 * dt2 *(1.0f + Sd) * u0_zz;
        const float u0_tt = (U0f[i] - 2.0f * U0c[i] + U0p) * inv_dt2;
        float born_source_m = -vp2 * dvp[i] * u0_tt;
        float born_source_anisotropy = vp2 * (2.0f * deps * u0_xx + dSd_m * (u0_xx + u0_zz));
        dUf[i] = 2.0f * dUc[i] - dUf[i] + dt2 * (propagation + born_source_m + born_source_anisotropy);
        
        if (ix < N_abc){
            U0f[i] *= A[ix];
            U0c[i] *= A[ix];
            dUf[i] *= A[ix];
            dUc[i] *= A[ix];
        }
        if (ix >=  nx - N_abc){
            U0f[i] *= A[nx - 1 - ix];
            U0c[i] *= A[nx - 1 - ix];
            dUf[i] *= A[nx - 1 - ix];
            dUc[i] *= A[nx - 1 - ix];
        }
        if (iz < N_abc){
            U0f[i] *= A[iz];
            U0c[i] *= A[iz];
            dUf[i] *= A[iz];
            dUc[i] *= A[iz];
        }
        if (iz >= nz - N_abc){
            U0f[i] *= A[nz - 1 - iz];
            U0c[i] *= A[nz - 1 - iz];
            dUf[i] *= A[nz - 1 - iz];
            dUc[i] *= A[nz - 1 - iz];
        }
    
    }
}
