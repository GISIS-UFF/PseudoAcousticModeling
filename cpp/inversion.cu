#include "Inversion.cuh"

inversion::Inversion(Survey* parameters, Modeling* modeling, Migration* migration)
{
    pmt = parameters;
    mdl = modeling;
    mgt = migration;
}

void Inversion::InitializeInversionFields(){
    const int n_model = pmt->nx * pmt->nz;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec * pmt->nt_data; 

    cudaMalloc((void**)&X, sizeof(float));
    cudaMalloc((void**)&residual, n_seis * sizeof(float))
    cudaMalloc((void**)&slowness2, n_model_exp * sizeof(float));
    if(pmt->multiparameter){
        if(pmt->approximation == "VTI" || pmt->approximation == "TTI"){
            cudaMalloc((void**)&eps_grad, n_model * sizeof(float));
            cudaMalloc((void**)&delta_grad, n_model * sizeof(float));
        }
        if(pmt->approximation == "TTI"){
            cudaMalloc((void**)&theta_grad, n_model * sizeof(float));
        }
    }
}

void Migration::loadObsSeismogram(const int shot){
    int n_seis = pmt->nt_data * pmt->Nrec;
    float* seismogram_h = new float[n_seis];

    std::ostringstream fcut_stream;
    fcut_stream<<std::fixed<<std::setprecision(1)<<pmt->fcut;
    std::string seismogramFile = pmt->seismogramFolder+"seismogram_shot_"+std::to_string(shot+1)+"_Nt"+std::to_string(pmt->nt_data)+"_Nrec"+std::to_string(pmt->Nrec)+"_fcut"+fcut_stream.str()+".bin";
    mdl->importBin(seismogramFile, seismogram_h, n_seis);
    Mute(seismogram_h, shot);
    
    cudaMemcpy(residual,seismogram_h,n_seis * sizeof(float), cudaMemcpyHostToDevice);
    delete[] seismogram_h;
}

void Inversion::ObjectiveFunction(){
    cudaMemset(X, 0, sizeof(float));
    slowness2ToVp<<<mdl->expBlocks, mdl->nThreads>>>(slowness2,mdl->vp,pmt->nx_abc,pmt->nz_abc)
    for (int shot = 0; shot < pmt->Nshot; shot++){
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        loadObsSeismogram(shot);
        sx = pmt->sx[shot];
        sz = pmt->sz[shot];
        mdl->resetFields();
        for (int k = 0; k < pmt->nt; k++){
            mdl->injectSource<<<1, 1>>>(mdl->current, mdl->source, k, pmt->nt, pmt->nx_abc, sx, sz);
            mdl->forward_step(k);
            if(k>=pmt->itlag){
                mdl->storeSeismogram<<<mdl->BlocksSeis, mdl->nThreads>>>(mdl->current, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }
            std::swap(mdl->current, mdl->future);
        }
        computeObjectiveFunction<<<mdl->seisBlocks, mdl->nThreads>>>(residual,mdl->seismogram, pmt->nt_data, pmt->Nrec)
    }
    std::cout << "info: Wave equation solved" << std::endl;
}

void Migration::backward_step(const int k, float* Pc, float* Pp, float* Pf){
    if (pmt->approximation == "acoustic"){
        updateAdjointWaveEquationandGradient<<<expBlocks, nThreads,0,mdl->compute_stream>>>(futurebck, currentbck, Pp, Pc, Pf, image, ilum, mdl->vp, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt, mdl->A, pmt->N_abc);
    }
    else if (pmt->approximation == "VTI"){
        calculateAdjointVTIProducts<<<expBlocks, nThreads,0,mdl->compute_stream>>>(currentbck, P, AUc, BUc, QCxUc, QCzUc, pmt->nx_abc, pmt->nz_abc, pmt->dx, pmt->dz, mdl->epsilon, mdl->delta);
        updateAdjointWaveEquationVTI<<<expBlocks, nThreads,0,mdl->compute_stream>>>(futurebck, currentbck, P, image, ilum, AUc, BUc, QCxUc, QCzUc, pmt->nx_abc, pmt->nz_abc,pmt->dt, pmt->dx, pmt->dz, mdl->vp, mdl->A, pmt->N_abc);
    }
    else if (pmt->approximation == "TTI"){
        calculateAdjointTTIProducts<<<expBlocks, nThreads,0,mdl->compute_stream>>>(currentbck, P, AUc, BUc, HUc, QCxUc, QCzUc, pmt->nx_abc, pmt->nz_abc, pmt->dx, pmt->dz, mdl->epsilon, mdl->delta, mdl->theta);
        updateAdjointWaveEquationTTI<<<expBlocks, nThreads,0,mdl->compute_stream>>>(futurebck, currentbck, P, image, ilum, AUc, BUc, HUc, QCxUc, QCzUc, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, mdl->vp, mdl->A, pmt->N_abc);
    }
}

void Inversion::calculateGradient(){
    std::cout << "info: Solving " + pmt->approximation + " Reverse Time Migration by " + pmt->migration + " method." << std::endl;

    mgt->initializeMigrationFields();
    mdl->createWavelet();
    if (pmt->ABC == "cerjan"){
        mdl->createCerjanVector();
    }
    for (int shot = 0; shot < pmt->Nshot; shot++){
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];
        resetFields();
        mdl->resetFields();
        loadObsSeismogram(shot);
        const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
        for (int k = 0; k < pmt->nt; k++){
            mdl->injectSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->current, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz);
            mdl->forward_step(k);
            if(k>=pmt->itlag){
                mdl->storeSeismogram<<<mdl->BlocksSeis, mdl->nThreads, 0, mdl->compute_stream>>>(mdl->current, mdl->seismogram, mdl->rx, mdl->rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }
            cudaMemcpyAsync(mgt->savefield + k * n_model_exp,mdl->current,n_model_exp * sizeof(float),cudaMemcpyDeviceToDevice,mdl->copy_stream);
            std::swap(mdl->current, mdl->future);
        }
        computeObjectiveFunction<<<mdl->seisBlocks, mdl->nThreads, mdl->compute_stream>>>(residual,mdl->seismogram, pmt->nt_data, pmt->Nrec)
        cudaStreamSynchronize(mdl->compute_stream);
        for (int t = pmt->nt - 1; t >= 0; t--){
            if (t >= pmt->itlag){
                int it = t - pmt->itlag;
                mgt->injectAdjointSource<<<mdl->seisBlocks, mdl->nThreads,0,mdl->compute_stream>>>(mgt->currentbck, residual, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz);
            }
            backward_step(t,mgt->savefield + t * n_model_exp);
            std::swap(mgt->currentbck, mgt->futurebck);
        }
        cudaStreamSynchronize(mdl->compute_stream);
    }

    mgt->normalizeImage<<<nBlocks, nThreads>>>(image, ilum, pmt->nx, pmt->nz);   
    saveImage();
    std::cout << "info: Reverse Time Migration" << std::endl;
}


__global__ void computeObjectiveFunction(float* X, float* __restrict__ residual, const float* __restrict__ calculated,const int nt, const int Nrec){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_seis = nt * Nrec;

    if (i < n_seis){
        float r = residual[i] - calculated[i];
        residual[i] = r;
        atomicAdd(X, 0.5f * r * r);
    }
}

__global__ void slowness2ToVp(float* __restrict__ slowness2,float* __restrict__ vp, int nx, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_model = nx * nz; 
    if (i < n_model){
        vp[i] = rsqrtf(slowness2[i]);
    }
}

__global__ void updateAdjointWaveEquationandGradient(float* __restrict__ Uf, float* __restrict__ Uc, const float* __restrict__ Pp, const float* __restrict__ Pc, const float* __restrict__ Pf,
float* __restrict__ vp_grad, float* __restrict__ ilum, const float* __restrict__ vp, const int nz, const int nx, const float dz, const float dx, const float dt, const float* __restrict__ A, const int N_abc){
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
            ilum[idx] += Pc[i] * Pc[i];
            vp_grad[idx] += Uc[i] * d2Pdt2;
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
float* __restrict__ vp_grad, float* __restrict__ eps_grad, float* __restrict__ delta_grad, const float* __restrict__ epsilon, const float* __restrict__ delta, const float dt, const float dx, const float dz,
const int nx, const int nz, const int N_abc, const bool multiparameter){
    
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

    const float inv_dx  = 1.0f / dx;
    const float inv_dz  = 1.0f / dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dt2 = 1.0f / (dt * dt);

    const float pxx =(c0 * P[i]
            + c1 * (P[i + 1] + P[i - 1])
            + c2 * (P[i + 2] + P[i - 2])
            + c3 * (P[i + 3] + P[i - 3])
            + c4 * (P[i + 4] + P[i - 4])) * inv_dx2;

    const float pzz =(c0 * P[i]
            + c1 * (P[i + nx] + P[i - nx])
            + c2 * (P[i + 2 * nx] + P[i - 2 * nx])
            + c3 * (P[i + 3 * nx] + P[i - 3 * nx])
            + c4 * (P[i + 4 * nx] + P[i - 4 * nx])) * inv_dz2;

    const float px =(a1 * (P[i + 1] - P[i - 1])
            + a2 * (P[i + 2] - P[i - 2])
            + a3 * (P[i + 3] - P[i - 3])
            + a4 * (P[i + 4] - P[i - 4])) * inv_dx;

    const float pz =(a1 * (P[i + nx] - P[i - nx])
            + a2 * (P[i + 2*nx] - P[i - 2*nx])
            + a3 * (P[i + 3*nx] - P[i - 3*nx])
            + a4 * (P[i + 4*nx] - P[i - 4*nx])) * inv_dz;


    const float px2 = px * px;
    const float pz2 = pz * pz;

    const float px4 = px2 * px2;
    const float pz4 = pz2 * pz2;

    const float px2pz2 = px2 * pz2;

    const float num = -2.0f * (eps - delta) * px2pz2;
    const float den = (1.0f + 2.0f * eps) * px4 + pz4 + 2.0f * (1.0f + delta) * px2pz2;

    float Sd = 0.0f;
    float Cx = 0.0f;
    float Cz = 0.0f;
    float dSd_deps   = 0.0f;
    float dSd_ddelta = 0.0f;

    if (fabsf(den) > 1.0e-12f)
    {
        const float inv_den = 1.0f / den;
        const float inv_den2 = inv_den * inv_den;
        Sd = num * inv_den;

        const float factor = 4.0f * (eps - delta) * ((1.0f + 2.0f * eps) * px4 - pz4 ) * inv_den2;
    
        Cx = factor * px * pz2;
        Cz = -factor * px2 * pz;

        if (multiparameter)
        {
            const float dnum_deps = -2.0f * px2pz2;
            const float dnum_ddelta = 2.0f * px2pz2;
            const float dden_deps = 2.0f * px4;
            const float dden_ddelta = 2.0f * px2pz2;
            dSd_deps = (dnum_deps * den - num * dden_deps) * inv_den2;
            dSd_ddelta = (dnum_ddelta * den - num * dden_ddelta) * inv_den2;
        }
    }

    const float A = 1.0f + 2.0f * eps + Sd;
    const float B = 1.0f + Sd;
    const float Q = pxx + pzz;
    const float adj = Uc[i];
    AUc[i] = A * adj;
    BUc[i] = B * adj;
    QCxUc[i] = Q * Cx * adj;
    QCzUc[i] = Q * Cz * adj;

    if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc)
    {
        const int xf = ix - N_abc;
        const int zf = iz - N_abc;

        const int nxf = nx - 2 * N_abc;
        const int idx = zf * nxf + xf;

        const float d2Pdt2 =(Pf[i] - 2.0f * Pc[i] + Pp[i]) * inv_dt2;
        vp_grad[idx] += adj * d2Pdt2;

        if (multiparameter)
        {
            const float dP_deps = (-2.0f - dSd_deps) * pxx - dSd_deps * pzz;
            const float dP_ddelta = -dSd_ddelta * Q;
            eps_grad[idx] += adj * dP_deps;
            delta_grad[idx] += adj * dP_ddelta;
        }
    }
}

__global__ void calculateAdjointTTIProductsAndGradients(const float* __restrict__ Uc, const float* __restrict__ Pp, const float* __restrict__ Pc, const float* __restrict__ Pf, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ HUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc,
float* __restrict__ vp_grad, float* __restrict__ eps_grad, float* __restrict__ delta_grad, float* __restrict__ theta_grad, const float* __restrict__ epsilon, const float* __restrict__ delta, const float* __restrict__ theta,
const float dt, const float dx, const float dz, const int nx, const int nz, const int N_abc, const bool multiparameter){ 
    
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

    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
    const float inv_dx2 = inv_dx * inv_dx;
    const float inv_dz2 = inv_dz * inv_dz;
    const float inv_dxdz = inv_dx * inv_dz;
    const float inv_dt2 = 1.0f / (dt * dt);

    const float pxx =(c0 * P[i]
            + c1 * (P[i + 1] + P[i - 1])
            + c2 * (P[i + 2] + P[i - 2])
            + c3 * (P[i + 3] + P[i - 3])
            + c4 * (P[i + 4] + P[i - 4])) * inv_dx2;

    const float pzz =(c0 * P[i]
            + c1 * (P[i + nx] + P[i - nx])
            + c2 * (P[i + 2 * nx] + P[i - 2 * nx])
            + c3 * (P[i + 3 * nx] + P[i - 3 * nx])
            + c4 * (P[i + 4 * nx] + P[i - 4 * nx])) * inv_dz2;

    const float px =(a1 * (P[i + 1] - P[i - 1])
            + a2 * (P[i + 2] - P[i - 2])
            + a3 * (P[i + 3] - P[i - 3])
            + a4 * (P[i + 4] - P[i - 4])) * inv_dx;

    const float pz =(a1 * (P[i + nx] - P[i - nx])
            + a2 * (P[i + 2*nx] - P[i - 2*nx])
            + a3 * (P[i + 3*nx] - P[i - 3*nx])
            + a4 * (P[i + 4*nx] - P[i - 4*nx])) * inv_dz;


    const float eps = epsilon[i];
    const float delta = delta[i];
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

    const float num = -2.0f * (eps - delta) * xi2eta2;
    const float den = (1.0f + 2.0f * eps) * xi4 + eta4 + 2.0f * (1.0f + delta) * xi2eta2;

    float Sd = 0.0f;
    float Cx = 0.0f;
    float Cz = 0.0f;
    float dSd_deps   = 0.0f;
    float dSd_ddelta = 0.0f;
    float dSd_dtheta = 0.0f;


    if (fabsf(den) > 1.0e-12f)
    {
        const float inv_den = 1.0f / den;
        const float inv_den2 = inv_den * inv_den;
        Sd = num * inv_den;

        const float K = (1.0f + 2.0f * eps) * xi4 - eta4;
        const float factor = 4.0f * (eps - delta) * xi * eta * K * inv_den2;

        Cx = factor * pz;
        Cz = -factor * px;

        if (multiparameter)
        {
            const float dnum_deps = -2.0f * xi2eta2;
            const float dnum_ddelta =  2.0f * xi2eta2;
            const float dden_deps =  2.0f * xi4;
            const float dden_ddelta =  2.0f * xi2eta2;
            const float dnum_dtheta = -2.0f * (eps - delta) * (-2.0f * xi * eta * eta * eta + 2.0f * xi * xi * xi * eta);
            const float dden_dtheta = -4.0f * (1.0f + 2.0f * eps) * xi * xi * xi * eta + 4.0f * xi * eta * eta * eta + 2.0f * (1.0f + delta) * (-2.0f * xi * eta * eta * eta + 2.0f * xi * xi * xi * eta);

            dSd_deps = (dnum_deps * den - num * dden_deps ) * inv_den2;
            dSd_ddelta = (dnum_ddelta * den - num * dden_ddelta ) * inv_den2;
            dSd_dtheta = (dnum_dtheta * den - num * dden_dtheta ) * inv_den2;
        }
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

        const float d2Pdt2 = (Pf[i] - 2.0f * Pc[i] + Pp[i]) * inv_dt2;
        vp_grad[idx] += adj * d2Pdt2;

        if (multiparameter)
        {
            const float pxz = (
            a1*a1*(Uc[i + nx + 1]     - Uc[i - nx + 1]     + Uc[i - nx - 1]     - Uc[i + nx - 1]) +
            a1*a2*(Uc[i + 2*nx + 1]   - Uc[i - 2*nx + 1]   + Uc[i - 2*nx - 1]   - Uc[i + 2*nx - 1]) +
            a1*a3*(Uc[i + 3*nx + 1]   - Uc[i - 3*nx + 1]   + Uc[i - 3*nx - 1]   - Uc[i + 3*nx - 1]) +
            a1*a4*(Uc[i + 4*nx + 1]   - Uc[i - 4*nx + 1]   + Uc[i - 4*nx - 1]   - Uc[i + 4*nx - 1]) +

            a2*a1*(Uc[i + nx + 2]     - Uc[i - nx + 2]     + Uc[i - nx - 2]     - Uc[i + nx - 2]) +
            a2*a2*(Uc[i + 2*nx + 2]   - Uc[i - 2*nx + 2]   + Uc[i - 2*nx - 2]   - Uc[i + 2*nx - 2]) +
            a2*a3*(Uc[i + 3*nx + 2]   - Uc[i - 3*nx + 2]   + Uc[i - 3*nx - 2]   - Uc[i + 3*nx - 2]) +
            a2*a4*(Uc[i + 4*nx + 2]   - Uc[i - 4*nx + 2]   + Uc[i - 4*nx - 2]   - Uc[i + 4*nx - 2]) +

            a3*a1*(Uc[i + nx + 3]     - Uc[i - nx + 3]     + Uc[i - nx - 3]     - Uc[i + nx - 3]) +
            a3*a2*(Uc[i + 2*nx + 3]   - Uc[i - 2*nx + 3]   + Uc[i - 2*nx - 3]   - Uc[i + 2*nx - 3]) +
            a3*a3*(Uc[i + 3*nx + 3]   - Uc[i - 3*nx + 3]   + Uc[i - 3*nx - 3]   - Uc[i + 3*nx - 3]) +
            a3*a4*(Uc[i + 4*nx + 3]   - Uc[i - 4*nx + 3]   + Uc[i - 4*nx - 3]   - Uc[i + 4*nx - 3]) +

            a4*a1*(Uc[i + nx + 4]     - Uc[i - nx + 4]     + Uc[i - nx - 4]     - Uc[i + nx - 4]) +
            a4*a2*(Uc[i + 2*nx + 4]   - Uc[i - 2*nx + 4]   + Uc[i - 2*nx - 4]   - Uc[i + 2*nx - 4]) +
            a4*a3*(Uc[i + 3*nx + 4]   - Uc[i - 3*nx + 4]   + Uc[i - 3*nx - 4]   - Uc[i + 3*nx - 4]) +
            a4*a4*(Uc[i + 4*nx + 4]   - Uc[i - 4*nx + 4]   + Uc[i - 4*nx - 4]   - Uc[i + 4*nx - 4])) * inv_dxdz;


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


