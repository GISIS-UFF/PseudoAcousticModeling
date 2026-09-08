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

    cudaMalloc((void**)&X, sizeof(double));
    cudaMallocHost((void**)&X_h,sizeof(double));
    cudaMallocHost((void**)&obs_h,n_seis * sizeof(float));
    cudaMallocHost((void**)&obs_buffer,n_seis * sizeof(float));
    cudaMallocHost((void**)&vp_h,n_model * sizeof(float));
    cudaMallocHost((void**)&vpnew_h,n_model * sizeof(float));
    cudaMallocHost((void**)&grad_vp_h,n_model *sizeof(float));
    cudaMallocHost((void**)&grad_vpnew_h,n_model * sizeof(float));
    cudaMallocHost((void**)&p_vp,n_model * sizeof(float));
    cudaMalloc((void**)&residual, n_seis * sizeof(float));
    cudaMalloc((void**)&residual_buffer, n_seis * sizeof(float));
    cudaMalloc((void**)&slowness2, n_model_exp * sizeof(float));

    if (pmt->migration == "checkpoint"){
        cudaMalloc((void**)&past_field, n_model_exp * sizeof(float));
    }
    
    if(pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaMallocHost((void**)&eps_h,n_model * sizeof(float));
        cudaMallocHost((void**)&delta_h,n_model * sizeof(float));
        if(pmt->multiparameter){
            cudaMallocHost((void**)&epsnew_h,n_model * sizeof(float));
            cudaMallocHost((void**)&deltanew_h,n_model * sizeof(float));
            cudaMallocHost((void**)&grad_eps_h,n_model *sizeof(float));
            cudaMallocHost((void**)&grad_epsnew_h,n_model * sizeof(float));
            cudaMallocHost((void**)&grad_delta_h,n_model *sizeof(float));
            cudaMallocHost((void**)&grad_deltanew_h,n_model * sizeof(float));
            cudaMallocHost((void**)&p_eps,n_model * sizeof(float));
            cudaMallocHost((void**)&p_delta,n_model * sizeof(float));
            cudaMalloc((void**)&eps_grad, n_model * sizeof(float));
            cudaMalloc((void**)&delta_grad, n_model * sizeof(float));
        }
    }
    if(pmt->approximation == "TTI"){
        cudaMallocHost((void**)&theta_h,n_model * sizeof(float));
        if(pmt->multiparameter){
            cudaMallocHost((void**)&thetanew_h,n_model * sizeof(float));
            cudaMallocHost((void**)&grad_theta_h,n_model *sizeof(float));
            cudaMallocHost((void**)&grad_thetanew_h,n_model * sizeof(float));
            cudaMallocHost((void**)&p_theta,n_model * sizeof(float));
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
    cudaFreeHost(grad_vp_h);
    cudaFreeHost(grad_vpnew_h);
    cudaFreeHost(p_vp);
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
            cudaFreeHost(grad_eps_h);
            cudaFreeHost(grad_epsnew_h);
            cudaFreeHost(grad_delta_h);
            cudaFreeHost(grad_deltanew_h);
            cudaFreeHost(p_eps);
            cudaFreeHost(p_delta);
            cudaFree(eps_grad);
            cudaFree(delta_grad);
        }
    }
    if(pmt->approximation == "TTI"){
        cudaFreeHost(theta_h);
        if(pmt->multiparameter){
            cudaFreeHost(thetanew_h);
            cudaFreeHost(grad_theta_h);
            cudaFreeHost(grad_thetanew_h);
            cudaFreeHost(p_theta);
            cudaFree(theta_grad);
        } 
    }

    delete[] water_mask;
}

double Inversion::ObjectiveFunction(){
    const int n_seis = pmt->Nrec*pmt->nt_data;
    cudaMemset(X, 0, sizeof(double));
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
            injectSource<<<1, 1, 0,mdl->compute_stream>>>(mdl->current, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz);
            mdl->forward_step(k);
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

    cudaMemcpyAsync(X_h, X, sizeof(double), cudaMemcpyDeviceToHost, mdl->compute_stream);
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
    cudaMemset(mgt->ilum, 0, n_model * sizeof(float));
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

void Inversion::backward_step(const int k, float* Pc, float* Pp, float* Pf){
    if(pmt->approximation == "acoustic"){
        updateAdjointWaveEquationandGradient<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->futurebck,mgt->currentbck,Pp,Pc,Pf,mgt->ilum,mgt->image,mdl->vp,pmt->nz_abc,pmt->nx_abc,pmt->dz,pmt->dx,pmt->dt,mdl->A,pmt->N_abc);
    }
    else if(pmt->approximation == "VTI"){
        calculateAdjointVTIProductsAndGradients<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->currentbck,Pp,Pc,Pf,mgt->ilum,mgt->AUc,mgt->BUc,mgt->QCxUc,mgt->QCzUc,mgt->image,eps_grad,delta_grad,mdl->epsilon,mdl->delta,pmt->dt,pmt->dx,pmt->dz,pmt->nx_abc,pmt->nz_abc,pmt->N_abc,pmt->multiparameter);
        updateAdjointWaveEquationVTI<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->futurebck,mgt->currentbck,mgt->AUc,mgt->BUc,mgt->QCxUc,mgt->QCzUc,pmt->nx_abc,pmt->nz_abc,pmt->dt,pmt->dx,pmt->dz,mdl->vp,mdl->A,pmt->N_abc);
    }
    else if(pmt->approximation == "TTI"){
        calculateAdjointTTIProductsAndGradients<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->currentbck,Pp,Pc,Pf,mgt->ilum,mgt->AUc,mgt->BUc,mgt->HUc,mgt->QCxUc,mgt->QCzUc,mgt->image,eps_grad,delta_grad,theta_grad,mdl->epsilon,mdl->delta,mdl->theta,pmt->dt,pmt->dx,pmt->dz,pmt->nx_abc,pmt->nz_abc,pmt->N_abc,pmt->multiparameter);
        updateAdjointWaveEquationTTI<<<mdl->expBlocks,nThreads,0,mdl->compute_stream>>>(mgt->futurebck,mgt->currentbck,mgt->AUc,mgt->BUc,mgt->HUc,mgt->QCxUc,mgt->QCzUc,pmt->nx_abc,pmt->nz_abc,pmt->dt,pmt->dx,pmt->dz,mdl->vp,mdl->A,pmt->N_abc);
    }
}

double Inversion::calculateGradientOntheFly(){
    std::cout << "info: Solving " + pmt->approximation + " Reverse Time Migration by " + pmt->migration + " method." << std::endl;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec*pmt->nt_data;
    cudaMemsetAsync(X, 0, sizeof(double), mdl->compute_stream);
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
            injectSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->current, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz);
            mdl->forward_step(k);
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
            if (t >= pmt->itlag){
                int it = t - pmt->itlag;
                injectAdjointSource<<<mdl->seisBlocks, nThreads,0,mdl->compute_stream>>>(mgt->currentbck, residual, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz);
            }

            const int previous_t = std::max(t - 1, 0);
            const int next_t = std::min(t + 1, pmt->nt - 1);

            float* Pc = mgt->savefield + t * n_model_exp;
            float* Pp = mgt->savefield + previous_t * n_model_exp;
            float* Pf = mgt->savefield + next_t * n_model_exp;

            backward_step(t, Pc, Pp, Pf);
            std::swap(mgt->currentbck, mgt->futurebck);
        }
        cudaStreamSynchronize(mdl->compute_stream);
    }
    normalizeImage<<<mgt->nBlocks,nThreads,0,mdl->compute_stream>>>(mgt->image,mgt->ilum,pmt->nx,pmt->nz);
    if(pmt->multiparameter){
        normalizeImage<<<mgt->nBlocks,nThreads,0,mdl->compute_stream>>>(eps_grad,mgt->ilum,pmt->nx,pmt->nz);
        normalizeImage<<<mgt->nBlocks,nThreads,0,mdl->compute_stream>>>(delta_grad,mgt->ilum,pmt->nx,pmt->nz);
    
        if(pmt->approximation == "TTI"){
            normalizeImage<<<mgt->nBlocks,nThreads,0,mdl->compute_stream>>>(theta_grad,mgt->ilum,pmt->nx,pmt->nz);
        }
    }
    cudaMemcpy(X_h, X, sizeof(double), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(mdl->compute_stream);
    std::cout << "info: Reverse Time Migration" << std::endl;
    return *X_h;
}

double Inversion::calculateGradientCheckpoint(){
    std::cout<<"info: Solving "+pmt->approximation+" Reverse Time Migration by "+pmt->migration+" method."<<std::endl;
    const int n_model_exp = pmt->nx_abc*pmt->nz_abc;
    const int n_seis = pmt->Nrec*pmt->nt_data;
    const int last_t = pmt->nt-1;
    const int last_checkpoint = last_t - pmt->step;
    cudaMemsetAsync(X,0,sizeof(double),mdl->compute_stream);
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
            injectSource<<< 1, 1, 0, mdl->compute_stream>>>(mdl->current, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz);
            mdl->forward_step(k);
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
                if(t>=pmt->itlag){
                    const int it=t-pmt->itlag;
                    injectAdjointSource<<<mdl->seisBlocks,nThreads,0,mdl->compute_stream>>>(mgt->currentbck, residual, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz);
                }

                cudaMemcpyAsync(past_field, mdl->future, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl->compute_stream);
                mdl->forward_step(t);
                backward_step(t, mdl->current, mdl->future, past_field);
                removeSource<<< 1, 1, 0, mdl->compute_stream>>>(mdl->current, mdl->source, t, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz,pmt->dx, pmt->dz);
                
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

    normalizeImage<<<mgt->nBlocks,nThreads,0,mdl->compute_stream>>>(mgt->image,mgt->ilum,pmt->nx,pmt->nz);
    if(pmt->multiparameter){
        normalizeImage<<<mgt->nBlocks,nThreads,0,mdl->compute_stream>>>(eps_grad,mgt->ilum,pmt->nx,pmt->nz);
        normalizeImage<<<mgt->nBlocks,nThreads,0,mdl->compute_stream>>>(delta_grad,mgt->ilum,pmt->nx,pmt->nz);
        
        if(pmt->approximation == "TTI"){
            normalizeImage<<<mgt->nBlocks,nThreads,0,mdl->compute_stream>>>(theta_grad,mgt->ilum,pmt->nx,pmt->nz);
        }
    }
    cudaMemcpyAsync(X_h, X, sizeof(double), cudaMemcpyDeviceToHost, mdl->compute_stream);
    cudaStreamSynchronize(mdl->compute_stream);
    std::cout<<"info: Reverse Time Migration"<<std::endl;

    return *X_h;
}

double Inversion::dot(const float* a,const float* b){
    double result = 0.0;
    const int n_model = pmt->nx*pmt->nz;
    #pragma omp parallel for reduction(+:result)
    for(int i = 0; i < n_model; i++){
        result += static_cast<double>(a[i]) * static_cast<double>(b[i]);
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

    std::vector<double> alpha(s_store.size(), 0.0);
    std::vector<double> rho(s_store.size(), 0.0);

    for (int i = s_store.size() - 1; i >= 0; --i) {
        const double sy = dot(s_store[i].data(), y_store[i].data());

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

    double gamma = 1.0;

    if (s_store.size() > 0){
        const double sy = dot(s_store.back().data(), y_store.back().data());
        const double yy = dot(y_store.back().data(), y_store.back().data());

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
        double beta = rho[i] * dot(y_store[i].data(), p);

        double coefficient = alpha[i] - beta;

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

double Inversion::calculateGradient(const std::string& parameter, float* gradient){
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

    double X_current = 0.0f;

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

double Inversion::calculateMultiparameterGradient(const bool update_eps, const bool update_delta, const bool update_theta, float* gradient_vp, float* gradient_eps, float* gradient_delta, float* gradient_theta){
    const bool multiparameter = pmt->multiparameter;
    const int n_model = pmt->nx * pmt->nz;
    pmt->multiparameter = update_eps || update_delta || update_theta;

    double X_current = 0.0;

    if(pmt->migration == "onthefly"){
        X_current = calculateGradientOntheFly();
    }
    else if(pmt->migration == "checkpoint"){
        X_current = calculateGradientCheckpoint();
    }
    else{
        pmt->multiparameter = multiparameter;
        throw std::runtime_error("Info: Invalid migration method for FWI gradient.");
    }

    cudaMemcpy(gradient_vp, mgt->image, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    if(update_eps){
        cudaMemcpy(gradient_eps, eps_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    }
    if(update_delta){
        cudaMemcpy(gradient_delta, delta_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    }
    if(update_theta){
        cudaMemcpy(gradient_theta, theta_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    }

    #pragma omp parallel for
    for(int i = 0; i < n_model; i++){
        if(water_mask[i]){
            gradient_vp[i] = 0.0f;
            if(update_eps){
                gradient_eps[i] = 0.0f;
            }
            if(update_delta){
                gradient_delta[i] = 0.0f;
            }
            if(update_theta){
                gradient_theta[i] = 0.0f;
            }
        }
    }

    pmt->multiparameter = multiparameter;
    
    return X_current;
}

float Inversion::armijolinesearch(const std::string& parameter, const float* model0, const float* grad0, const float* p, const double X0, const bool empty, const float scale){
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

    const double c1 = 1.0e-4;
    const int max_iters = 10;

    const double gp0 = static_cast<double>(scale) * dot(grad0,p);
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

        const double X_i = ObjectiveFunction();
        const double armijo_limit = X0 + c1*beta*gp0;

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

float Inversion::linesearch(const std::string& parameter, const float* model0, const float* grad0, const float* p, const double X0, double& X_new, float* grad_new, const bool empty, const float scale){
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

    const double c1 = 1.0e-4;
    const double c2 = 0.9;
    const int max_iters = 10;
    const double gp0 = static_cast<double>(scale) * dot(grad0,p);

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
    double X_past = X0;

    float* model_i = new float[n_model];
    float* grad_i = new float[n_model];

    for(int i = 0; i < max_iters; i++){
        applyModelStep(parameter,model0,p,model_i,alpha_i);
        ExpandModelDevice(parameter,model_i);
        double X_i = ObjectiveFunction();

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
        const double gpi = static_cast<double>(scale) * dot(grad_i,p);
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

float Inversion::zoom(const std::string& parameter,const float* model0, const float* grad0, const float* p, const double X0, const double gp0, float alpha_lo, float alpha_hi,const double X_lo_initial, double& X_new, float* grad_new, const float scale){
    std::cout << "Info: Starting zoom for parameter " << parameter << std::endl;

    const int n_model = pmt->nx*pmt->nz;
    const double c1 = 1.0e-4;
    const double c2 = 0.9;
    const int max_iters = 10;

    float* model_i = new float[n_model];
    float* grad_i = new float[n_model];

    double X_lo = X_lo_initial;
    float alpha_i;

    for(int i = 0; i < max_iters; i++){

        alpha_i = 0.5f*(alpha_lo + alpha_hi);

        applyModelStep(parameter,model0,p,model_i,alpha_i);
        ExpandModelDevice(parameter,model_i);
        double X_i = ObjectiveFunction();

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

            const double gpi = static_cast<double>(scale) * dot(grad_i, p);

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

float Inversion::linesearchMultiparameter(const float* vp, const float* epsilon, const float* delta, const float* theta, const float* p_vp, const float* p_eps, const float* p_delta, const float* p_theta,
const float* g_vp, const float* g_eps, const float* g_delta, const float* g_theta, const float beta_vp, const float beta_eps, const float beta_delta, const float beta_theta, const float scale_vp, const float scale_eps, const float scale_delta, const float scale_theta,
const bool update_eps, const bool update_delta, const bool update_theta, const double X0){
    std::cout << "Info: Starting multiparameter line search " << std::endl;
    const double c1 = 1.0e-4;
    const int max_iters = 10;

    const int n_model = pmt->nx * pmt->nz;

    float* vp_i    = new float[n_model];
    float* eps_i   = new float[n_model];
    float* delta_i = new float[n_model];
    float* theta_i = nullptr;

    if(pmt->approximation == "TTI"){
        theta_i = new float[n_model];
    }

    double gp0 = static_cast<double>(beta_vp) * static_cast<double>(scale_vp) * dot(g_vp, p_vp);

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

        const double X_i = ObjectiveFunction();

        const double armijo_limit = X0 + c1 * alpha * gp0;

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
    pmt->itlag = static_cast<int>(pmt->tlag/pmt->dt);
    pmt->nt = pmt->itlag + pmt->nt_data;
    cudaMemset((void**)&mdl->source, 0, pmt->nt * sizeof(float));
    mdl->createWavelet();
    if (pmt->migration == "onthefly"){
        cudaMemset((void**)&mgt->savefield, 0, pmt->nt * n_model_exp * sizeof(float));
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

    const double sy = dot(s_store.back().data(), y_store.back().data());
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

    if (pmt->ABC == "cerjan"){
        mdl->createCerjanVector();
    }

    for(const float fmax : pmt->freqs){
        std::cout << "info: FWI frequency " << fmax <<std::endl;
        adjustfmax(fmax);
        s_vp_store.clear();
        y_vp_store.clear();

        double X_current = calculateGradient("vp",grad_vp_h);

        const float gmax0 = getGradientScale(grad_vp_h);
        scaleGradient(grad_vp_h, gmax0);

        const double X_freq0 = X_current;
        history_stream << X_current/X_freq0 << " " << fmax << std::endl;

        for(int itr = 0; itr < pmt->niter; itr++){
            std::cout << "info: FWI iteration " << itr + 1 << "/" << pmt->niter << " for frequency " << fmax << std::endl;
            std::ostringstream fcut_stream;
            fcut_stream<<std::fixed<<std::setprecision(1)<<fmax;
            const std::string gradient_file = pmt->gradientsFolder+"vp_gradient_fwi_iter_"+std::to_string(itr+1)+"_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_freq"+fcut_stream.str()+".bin";
            saveModel(gradient_file,grad_vp_h);

            twoLoopRecursion(grad_vp_h,p_vp,s_vp_store,y_vp_store);

            double X_new;
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

void Inversion::solveFullWaveformInversionMultiparameterHierarchical(){
    std::cout << "info: Solving hierarchical multiparameter FWI" << std::endl;

    if(pmt->approximation != "VTI" && pmt->approximation != "TTI"){
        throw std::runtime_error("Hierarchical multiparameter FWI requires VTI or TTI.");
    }

    if(!pmt->multiparameter){
        throw std::runtime_error("Hierarchical multiparameter FWI requires multiparameter=true.");
    }

    const int n_model = pmt->nx*pmt->nz;
    const float eps_start = 0.5f;
    const float delta_start = 0.75f;
    const float theta_start = 0.85f;

    const int eps_first_itr = 1 + static_cast<int>(std::ceil(eps_start*(pmt->niter - 1)));
    const int delta_first_itr = 1 + static_cast<int>(std::ceil(delta_start*(pmt->niter - 1)));
    const int theta_first_itr = 1 + static_cast<int>(std::ceil(theta_start*(pmt->niter - 1)));

    setModel();

    float* final_model = new float[n_model];

    const std::string history_file = "../outputs/history.txt";
    std::ofstream history_stream(history_file);

    if(pmt->ABC == "cerjan"){
        mdl->createCerjanVector();
    }

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

        double X_current = calculateMultiparameterGradient(false,false,false,grad_vp_h,grad_eps_h,grad_delta_h,grad_theta_h);

        const float g_vp_max0 = getGradientScale(grad_vp_h);
        scaleGradient(grad_vp_h,g_vp_max0);

        float g_eps_max0 = 1.0f;
        float g_delta_max0 = 1.0f;
        float g_theta_max0 = 1.0f;

        const double X_freq0 = X_current;

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

            const double X_new = calculateMultiparameterGradient(update_eps,update_delta,update_theta,grad_vpnew_h,grad_epsnew_h,grad_deltanew_h,grad_thetanew_h);

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
                const std::string theta_model_file = pmt->estimatedmodelsFolder+"fwi_theta_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+"_itr"+std::to_string(iteration)+"_freq"+fcut_stream.str()+".bin";
                saveModel(theta_model_file,theta_h);
            }
        }
    }

    delete[] final_model;
    history_stream.close();

    std::cout << "info: FWI history saved to " << history_file << std::endl;
}

__global__ void computeObjectiveFunction(double* X, float* __restrict__ residual, const float* __restrict__ calculated,const int nt, const int Nrec){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_seis = nt * Nrec;

    if (i < n_seis){
        const float r = residual[i] - calculated[i];
        residual[i] = r;
        atomicAdd(X, 0.5f * static_cast<double>(r) * static_cast<double>(r));
    }
}

__global__ void slowness2ToVp(const float* __restrict__ slowness2,float* __restrict__ vp, int nx, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_model = nx * nz; 
    if (i < n_model){
        vp[i] = rsqrtf(slowness2[i]);
    }
}

__global__ void updateAdjointWaveEquationandGradient(float* __restrict__ Uf, float* __restrict__ Uc, const float* __restrict__ Pp, const float* __restrict__ Pc, const float* __restrict__ Pf, float* __restrict__ ilum,
float* __restrict__ vp_grad, const float* __restrict__ vp, const int nz, const int nx, const float dz, const float dx, const float dt, const float* __restrict__ A, const int N_abc){
    const float c0 = -2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;

    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);

    const float dt2     = dt * dt;
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

            const float d2Pdt2 =(Pf[i] - 2.0f * Pc[i] + Pp[i]) * inv_dt2;
            vp_grad[idx] += Uc[i] * d2Pdt2;
            ilum[idx] += Pc[i] * Pc[i];
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

__global__ void calculateAdjointVTIProductsAndGradients(const float* __restrict__ Uc, const float* __restrict__ Pp, const float* __restrict__ Pc, const float* __restrict__ Pf, float* __restrict__ ilum, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc,
float* __restrict__ vp_grad, float* __restrict__ eps_grad, float* __restrict__ delta_grad, const float* __restrict__ epsilon, const float* __restrict__ delta, const float dt, const float dx, const float dz,
const int nx, const int nz, const int N_abc, const bool multiparameter){

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


    if (ix < 4 || ix >= nx - 4 || iz < 4 || iz >= nz - 4)
    {
        return;
    }

    const double pxx =(c0 * Pc[i]
            + c1 * (Pc[i + 1] + Pc[i - 1])
            + c2 * (Pc[i + 2] + Pc[i - 2])
            + c3 * (Pc[i + 3] + Pc[i - 3])
            + c4 * (Pc[i + 4] + Pc[i - 4])) * inv_dx2;

    const double pzz =(c0 * Pc[i]
            + c1 * (Pc[i + nx] + Pc[i - nx])
            + c2 * (Pc[i + 2 * nx] + Pc[i - 2 * nx])
            + c3 * (Pc[i + 3 * nx] + Pc[i - 3 * nx])
            + c4 * (Pc[i + 4 * nx] + Pc[i - 4 * nx])) * inv_dz2;

    const double px =(a1 * (Pc[i + 1] - Pc[i - 1])
            + a2 * (Pc[i + 2] - Pc[i - 2])
            + a3 * (Pc[i + 3] - Pc[i - 3])
            + a4 * (Pc[i + 4] - Pc[i - 4])) * inv_dx;

    const double pz =(a1 * (Pc[i + nx] - Pc[i - nx])
            + a2 * (Pc[i + 2*nx] - Pc[i - 2*nx])
            + a3 * (Pc[i + 3*nx] - Pc[i - 3*nx])
            + a4 * (Pc[i + 4*nx] - Pc[i - 4*nx])) * inv_dz;

    const float eps = epsilon[i];
    const float del = delta[i];

    const double px2 = px * px;
    const double pz2 = pz * pz;

    const double px4 = px2 * px2;
    const double pz4 = pz2 * pz2;

    const double px2pz2 = px2 * pz2;

    const double num = -2.0f * (eps - del) * px2pz2;
    const double den = (1.0f + 2.0f * eps) * px4 + pz4 + 2.0f * (1.0f + del) * px2pz2;

    double Sd = 0.0f;
    double Cx = 0.0f;
    double Cz = 0.0f;
    double dSd_deps   = 0.0f;
    double dSd_ddelta = 0.0f;

    if(std::abs(den) > 1.0e-25){
        const double inv_den = 1.0f / den;
        const double inv_den2 = inv_den * inv_den;
        Sd = num * inv_den;

        const double factor = 4.0f * (eps - del) * ((1.0f + 2.0f * eps) * px4 - pz4 ) * inv_den2;
    
        Cx = factor * px * pz2;
        Cz = -factor * px2 * pz;

        if(multiparameter){
            const double dnum_deps = -2.0f * px2pz2;
            const double dnum_ddelta = 2.0f * px2pz2;
            const double dden_deps = 2.0f * px4;
            const double dden_ddelta = 2.0f * px2pz2;
            dSd_deps = (dnum_deps * den - num * dden_deps) * inv_den2;
            dSd_ddelta = (dnum_ddelta * den - num * dden_ddelta) * inv_den2;
        }
    }

    const double A = 1.0f + 2.0f * eps + Sd;
    const double B = 1.0f + Sd;
    const double Q = pxx + pzz;
    const double adj = Uc[i];
    AUc[i] = A * adj;
    BUc[i] = B * adj;
    QCxUc[i] = Q * Cx * adj;
    QCzUc[i] = Q * Cz * adj;

    if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc){
        const int xf = ix - N_abc;
        const int zf = iz - N_abc;

        const int nxf = nx - 2 * N_abc;
        const int idx = zf * nxf + xf;

        const double d2Pdt2 =(Pf[i] - 2.0f * Pc[i] + Pp[i]) * inv_dt2;
        ilum[idx] += Pc[i] * Pc[i];
        vp_grad[idx] += adj * d2Pdt2;

        if (multiparameter){
            const double dP_deps = (-2.0f - dSd_deps) * pxx - dSd_deps * pzz;
            const double dP_ddelta = -dSd_ddelta * Q;

            eps_grad[idx] += adj * dP_deps;
            delta_grad[idx] += adj * dP_ddelta;
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

__global__ void calculateAdjointTTIProductsAndGradients(const float* __restrict__ Uc, const float* __restrict__ Pp, const float* __restrict__ Pc, const float* __restrict__ Pf, float* __restrict__ ilum, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ HUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc,
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

    const double pxx =(c0 * Pc[i]
            + c1 * (Pc[i + 1] + Pc[i - 1])
            + c2 * (Pc[i + 2] + Pc[i - 2])
            + c3 * (Pc[i + 3] + Pc[i - 3])
            + c4 * (Pc[i + 4] + Pc[i - 4])) * inv_dx2;

    const double pzz =(c0 * Pc[i]
            + c1 * (Pc[i + nx] + Pc[i - nx])
            + c2 * (Pc[i + 2 * nx] + Pc[i - 2 * nx])
            + c3 * (Pc[i + 3 * nx] + Pc[i - 3 * nx])
            + c4 * (Pc[i + 4 * nx] + Pc[i - 4 * nx])) * inv_dz2;

    const double px =(a1 * (Pc[i + 1] - Pc[i - 1])
            + a2 * (Pc[i + 2] - Pc[i - 2])
            + a3 * (Pc[i + 3] - Pc[i - 3])
            + a4 * (Pc[i + 4] - Pc[i - 4])) * inv_dx;

    const double pz =(a1 * (Pc[i + nx] - Pc[i - nx])
            + a2 * (Pc[i + 2*nx] - Pc[i - 2*nx])
            + a3 * (Pc[i + 3*nx] - Pc[i - 3*nx])
            + a4 * (Pc[i + 4*nx] - Pc[i - 4*nx])) * inv_dz;


    const float eps = epsilon[i];
    const float del = delta[i];
    const float th = theta[i];

    float s;
    float c;

    sincosf(th, &s, &c);

    const double xi = px * c - pz * s;
    const double eta = px * s + pz * c;
    const double xi2 = xi * xi;
    const double eta2 = eta * eta;
    const double xi4 = xi2 * xi2;
    const double eta4 = eta2 * eta2;
    const double xi2eta2 = xi2 * eta2;

    const double num = -2.0f * (eps - del) * xi2eta2;
    const double den = (1.0f + 2.0f * eps) * xi4 + eta4 + 2.0f * (1.0f + del) * xi2eta2;

    double Sd = 0.0f;
    double Cx = 0.0f;
    double Cz = 0.0f;
    double dSd_deps   = 0.0f;
    double dSd_ddelta = 0.0f;
    double dSd_dtheta = 0.0f;


    if(std::abs(den) > 1.0e-25){
        const double inv_den = 1.0f / den;
        const double inv_den2 = inv_den * inv_den;
        Sd = num * inv_den;

        const double K = (1.0f + 2.0f * eps) * xi4 - eta4;
        const double factor = 4.0f * (eps - del) * xi * eta * K * inv_den2;

        Cx = factor * pz;
        Cz = -factor * px;

        if (multiparameter)
        {
            const double dnum_deps = -2.0f * xi2eta2;
            const double dnum_ddelta =  2.0f * xi2eta2;
            const double dden_deps =  2.0f * xi4;
            const double dden_ddelta =  2.0f * xi2eta2;
            const double dnum_dtheta = -2.0f * (eps - del) * (-2.0f * xi * eta * eta * eta + 2.0f * xi * xi * xi * eta);
            const double dden_dtheta = -4.0f * (1.0f + 2.0f * eps) * xi * xi * xi * eta + 4.0f * xi * eta * eta * eta + 2.0f * (1.0f + del) * (-2.0f * xi * eta * eta * eta + 2.0f * xi * xi * xi * eta);

            dSd_deps = (dnum_deps * den - num * dden_deps) * inv_den2;
            dSd_ddelta = (dnum_ddelta * den - num * dden_ddelta) * inv_den2;
            dSd_dtheta = (dnum_dtheta * den - num * dden_dtheta) * inv_den2;
        }
    }

    const float cos2 = c * c;
    const float sin2 = s * s;
    const float sin2th = 2.0f * s * c;
    const float cos2th = cos2 - sin2;

    const double Acoef = (1.0f + 2.0f * eps) * cos2 + sin2 + Sd;
    const double Bcoef = (1.0f + 2.0f * eps) * sin2 + cos2 + Sd;
    const double Hcoef = 2.0f * eps * sin2th;
    const double Q = pxx + pzz;
    const double adj = Uc[i];
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

        const double d2Pdt2 = (Pf[i] - 2.0f * Pc[i] + Pp[i]) * inv_dt2;
        ilum[idx] += Pc[i] * Pc[i];
        vp_grad[idx] += adj * d2Pdt2;

        if (multiparameter)
        {
            const double pxz = (
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


            const double dA_dtheta = 2.0f * eps * sin2th - dSd_dtheta;
            const double dB_dtheta = -2.0f * eps * sin2th - dSd_dtheta;
            const double dC_dtheta = 4.0f * eps * cos2th;
            const double dP_deps = -(2.0f * cos2 + dSd_deps) * pxx-(2.0f * sin2 + dSd_deps) * pzz + 2.0f * sin2th * pxz;
            const double dP_ddelta = -dSd_ddelta * Q;
            const double dP_dtheta = dA_dtheta * pxx + dB_dtheta * pzz + dC_dtheta * pxz;

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