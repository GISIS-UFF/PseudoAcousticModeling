#include "Migration.cuh"
#include <algorithm>
#include <cmath>

Migration:Migration(Survey* parameters, Modeling* modeling)
{
    pmt = parameters;
    mdl = modeling;
}

void Migration::initializeMigrationFields()
{
    const int n_model = pmt->nx * pmt->nz;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc

    cudaMalloc((void**)&image, n_model * sizeof(float));
    cudaMalloc((void**)&ilum, n_model * sizeof(float));
    cudaMalloc((void**)&currentbck, n_model_exp * sizeof(float));
    cudaMalloc((void**)&futurebck, n_model_exp * sizeof(float));

    if (pmt->migration == "onthefly"){
        if (pmt->approximation == "acoustic"){
            cudaMalloc((void**)&savefield, n_model * sizeof(float));
        }
        if else(pmt->approximation == "VTI" || pmt->approximation == "TTI"){
            cudaMalloc((void**)&savefield, n_model_exp * sizeof(float));
        }
        
    }

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaMalloc((void**)&AUc, n_model_exp * sizeof(float));
        cudaMalloc((void**)&BUc, n_model_exp * sizeof(float));
        cudaMalloc((void**)&QCxUc, n_model_exp * sizeof(float));
        cudaMalloc((void**)&QCzUc, n_model_exp * sizeof(float));
        if (pt->approximation == "TTI"){
            cudaMalloc((void**)&HUc, n_model_exp * sizeof(float)) ;   
        }
    }

    if (pmt->fwi && pmt->multiparameter){
        cudaMalloc((void**)&eps_grad, n_model * sizeof(float));
        cudaMalloc((void**)&delta_grad, n_model * sizeof(float));
        cudaMalloc((void**)&theta_grad, n_model * sizeof(float));
    } 

}

void Migration::freeMemory(){
    cudaFree(image);
    cudaFree(ilum);
    cudaFree(currentbck);
    cudaFree(futurebck);
    if (pmt->migration == "onthefly"){
        cudaFree(savefield)
    }
    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaFree(AUc);
        cudaFree(BUc);
        cudaFree(QCxUc);
        cudaFree(QCzUc);
        if (pt->approximation == "TTI"){
            cudaFree(HUc);   
        }
    }

    if (pmt->fwi && pmt->multiparameter){
        cudaFree(eps_grad);
        cudaFree(delta_grad);
        cudaFree(theta_grad);
    } 
}

void Migration::resetFields(){
    int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    cudaMemset(currentbck, 0, n_model_exp * sizeof(float));
    cudaMemset(futurebck, 0, n_model_exp * sizeof(float));
}

void Migration::Mute(float* seismogram, const int shot)
{
    for (int irec = 0; irec < pmt->Nrec; irec++)
    {
        float dz = pmt->rec_z[irec] - pmt->sz;
        float dx = pmt->rec_x[irec] - pmt->sx;

        float dist = sqrtf(dx * dx + dz * dz);

        float traveltime = (dist / pmt->v0) + pmt->shift;

        float t1 = traveltime;
        float t2 = t1 + pmt->window;

        for (int it = 0; it < pmt->Nt; it++)
        {
            float t = it * pmt->dt;
            int index = it * pmt->Nrec + irec;

            if (t < t1)
            {
                seismogram[index] = 0.0f;
            }
            else if (t < t2)
            {
                seismogram[index] *= (t - t1) / (t2 - t1);
            }
        }
    }
}

void Migration::loadSeismogram(const int shot){
    n_seis = pmt->Nt * pmt->Nrec;
    seismogram_h = new float[n_seis];
    
    if (pmt->fwi){
        std::ostringstream fcut_stream;
        fcut_stream<<std::fixed<<std::setprecision(1)<<pmt->fcut;
        std::string seismogramFile = pmt->seismogramFolder+"residual_shot_"+std::to_string(shot+1)+"_Nt"+std::to_string(pmt->nt_data)+"_Nrec"+std::to_string(pmt->Nrec)+"_fcut"+fcut_stream.str()+".bin";
        mdl->importBin(seismogramFile, seismogram_h, n_seis);
    }
    else{
        std::ostringstream fcut_stream;
        fcut_stream<<std::fixed<<std::setprecision(1)<<pmt->fcut;
        std::string seismogramFile = pmt->seismogramFolder+"seismogram_shot_"+std::to_string(shot+1)+"_Nt"+std::to_string(pmt->nt_data)+"_Nrec"+std::to_string(pmt->Nrec)+"_fcut"+fcut_stream.str()+".bin";
        mdl->importBin(seismogramFile, seismogram_h, n_seis);
        Mute(seismogram_h, shot)
    }

    cudaMemcpy(seismogram,seismogram_h,n_seis * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<unsigned char> Migration::createMask(float* f){
    int n_model = pmt->nx * pmt->nz;
    float f_min = *std::min_element(f, f + n);

    bool* mask = new bool[n_model];

    for (int i = 0; i < n_model; i++)
    {
        mask[i] = std::fabs(f[i] - f_min) < 1e-3f;
    }

    return mask;
}

float Migration::gaussianKernel(const int x, const int z)
{
    const float pi = 3.14159265358979323846f;

    float fator = 1.0f / (2.0f * pi * pmt->sigma * pmt->sigma);
    float expoente = -(x * x + z * z) / (2.0f * pmt->sigma * pmt->sigma);

    return fator * std::exp(expoente);
}


std::vector<float> Migration::gaussianFilter2D()
{
    int kernelSize = std::ceil(6.0f * pmt->sigma + 1.0f);

    if (kernelSize % 2 == 0)
        kernelSize++;

    std::vector<float> kernel(kernelSize * kernelSize, 0.0f);

    float total = 0.0f;

    for (int lin = 0; lin < kernelSize; lin++)
    {
        for (int col = 0; col < kernelSize; col++)
        {
            int x = lin - kernelSize / 2;
            int z = col - kernelSize / 2;

            float val = gaussianKernel(x, z, pmt->sigma);

            kernel[lin * kernelSize + col] = val;

            total += val;
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; i++)
        kernel[i] /= total;

    return kernel;
}

void Migration::smoothModel(float* f, const bool* mask, const bool parameter = false)
{
    const int n = pmt->nz * pmt->nx;

    std::vector<float> s(n);
    std::vector<float> sOld(n);

    if (parameter)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            s[i] = f[i];
    }
    else
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            s[i] = 1.0f / f[i];
    }

    sOld = s;

    int kernelSize;

    std::vector<float> kernel = gaussianFilter2D(pmt->sigma, kernelSize);

    const int half = kernelSize / 2;

    #pragma omp parallel for
    for (int z = half; z < pmt->nz - half; z++)
    {
        for (int x = half; x < pmt->nx - half; x++)
        {
            float newValue = 0.0f;
            float total = 0.0f;

            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    int zz = z + i - half;
                    int xx = x + j - half;

                    float weight = kernel[i * kernelSize + j];
                    newValue += weight * sOld[zz * pmt->nx + xx];
                    total += weight;
                }
            }

            if (total > 0.0f)
            {
                s[z * pmt->nx + x] =newValue / total;
            }
        }
    }

    for (int z = 0; z < half; z++)
    {
        for (int x = 0; x < pmt->nx; x++)
        {
            s[z * pmt->nx + x] = s[half * pmt->nx + x];

            s[(pmt->nz - 1 - z) * pmt->nx + x] = s[(pmt->nz - 1 - half) * pmt->nx + x];
        }
    }

    for (int x = 0; x < half; x++)
    {
        for (int z = 0; z < pmt->nz; z++)
        {
            s[z * pmt->nx + x] = s[z * pmt->nx + half];

            s[z * pmt->nx + (pmt->nx - 1 - x)] = s[z * pmt->nx + (pmt->nx - 1 - half)];
        }
    }

    #pragma omp parallel for
    for (int x = 0; x < pmt->nx; x++)
    {
        for (int z = 0; z < pmt->nz; z++)
        {
            int index = z * pmt->nx + x;

            if (mask[index])
            {
                s[index] = sOld[index];
            }
            else
            {
                break;
            }
        }
    }

    if (parameter)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            f[i] = s[i];
    }
    else
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            f[i] = 1.0f / s[i];
    }
}


void Migration::backward_step(const int k){
    if (k >= pmt->itlag){
        injectAdjointSource <<<1, 1>>>(currentbck, muted_seismogram, rx, rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz);
    }
    if (pmt->approximation == "acoustic"){
        mdl->updateWaveEquation<<<nBlocks, nThreads>>>(futurebck, currentbck, vp, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt, A, pmt->N_abc);
    }
    else if (pmt->approximation == "VTI"){
        calculateAdjointVTIProducts<<<nBlocks, nThreads>>>(currentbck, mdl->current, AUc, BUc, QCxUc, QCzUc, pmt->nx_abc, pmt->nz_abc, pmt->dx, pmt->dz, epsilon, delta)
        updateAdjointWaveEquationVTI<<<nBlocks, nThreads>>>(futurebck, currentbck, AUc, BUc, QCxUc, QCzUc, pmt->dt, pmt->dx, pmt->dz, pmt->nx_abc, pmt->nz_abc, vp, epsilon, delta, mdl->A, pmt->N_abc);
    }
    else if (pmt->approximation == "TTI"){
        calculateAdjointTTIProducts<<<nBlocks, nThreads>>>(currentbck, mdl->current, AUc, BUc, HUc, QCxUc, QCzUc, pmt->nx_abc, pmt->nz_abc, pmt->dx, pmt->dz, epsilon, delta, theta)
        updateAdjointWaveEquationTTI<<<nBlocks, nThreads>>>(futurebck, currentbck, AUc, BUc, HUc, QCxUc, QCzUc, pmt->dt, pmt->dx, pmt->dz, pmt->nx_abc, pmt->nz_abc, vp, epsilon, delta, theta, mdl->A, pmt->N_abc);
    }
}

void Migration::setModel(){
    int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    int n_model = pmt->nx * pmt->nz;

    float* vp_h = new float[n_model]();
    float* vp_exp_h = new float[n_model_exp]();
    float* epsilon_h = nullptr;
    float* epsilon_exp_h = nullptr;
    float* delta_h = nullptr;
    float* delta_exp_h = nullptr;
    float* theta_h = nullptr;
    float* theta_exp_h = nullptr;

    mdl->importBin(pmt->vpFile, vp_h, n_model);
    if(!pmt->fwi){
        auto vp_mask = createMask(vp_h);
        smoothModel(vp_h,vp_mask);
    }
    mdl->expandModel(vp_h, vp_exp_h);
    cudaMemcpy(vp, vp_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        epsilon_h = new float[n_model]();
        epsilon_exp_h = new float[n_model_exp]();
        mdl->importBin(pmt->epsilonFile, epsilon_h, n_model);
        if(!pmt->fwi){
            auto eps_mask = createMask(epsilon_h);
            smoothModel(epsilon_h,eps_mask);
        }
        mdl->expandModel(epsilon_h, epsilon_exp_h);
        cudaMemcpy(epsilon, epsilon_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);

        delta_h = new float[n_model]();
        delta_exp_h = new float[n_model_exp]();
        mdl->importBin(pmt->deltaFile, delta_h, n_model);
        if(!pmt->fwi){
            auto delta_mask = createMask(delta_h);
            smoothModel(delta_h,delta_mask);
        }
        mdl->expandModel(delta_h, delta_exp_h);
        cudaMemcpy(delta, delta_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (pmt->approximation == "TTI"){
        theta_h = new float[n_model]();
        theta_exp_h = new float[n_model_exp]();
        mdl->importBin(pmt->thetaFile, theta_h, n_model);
        if(!pmt->fwi){
            auto theta_mask = createMask(theta_h);
            smoothModel(theta_h,theta_mask);
        }
        #pragma omp parallel for
        for (int i = 0; i < n_model; i++){
            theta_h[i] = theta_h[i] * pi / 180.0f;
        }
        mdl->expandModel(theta_h, theta_exp_h);
        cudaMemcpy(theta, theta_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    }

    delete[] vp_h;
    delete[] vp_exp_h;
    delete[] epsilon_h;
    delete[] epsilon_exp_h;
    delete[] delta_h;
    delete[] delta_exp_h;
    delete[] theta_h;
    delete[] theta_exp_h;
}

void Migration::solveReverseTimeMigrationOntheFly(){
    initializeMigrationFields();
    mdl->createWavelet();
    if (pmt->ABC == "cerjan"){
        mdl->createCerjanVector();
    }
    setModel();
    for (int shot = 0; shot < pmt->Nshot; shot++){
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        sx = pmt->sx[shot];
        sz = pmt->sz[shot];
        resetFields();
        for (int k = 0; k < pmt->nt; k++){
            foward_step(k);
            if (pmt->approximation == "acoustic"){
                cudaMemcpy2D(save_field + k * pmt->Nz * pmt->Nx, pmt->Nx * sizeof(float), mdl->current + pmt->N_abc * pmt->Nx_abc + pmt->N_abc, pmt->Nx_abc * sizeof(float),pmt->Nx * sizeof(float),pmt->Nz,cudaMemcpyDeviceToDevice);
            }
            if else (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
                cudaMemcpy(save_field, mdl->current, pmt->Nz_abc * pmt->Nx_abc * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            saveSnapshot(shot, k, current);
            std::swap(current, future);
        }
        saveSeismogram(shot);
        std::cout << "info: Wave equation solved" << std::endl;
    }


    
}

__global__ void updateIllumination(
    float* ilum,
    const float* save_field,
    const int k,
    const int Nz,
    const int Nx
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && z < Nz)
    {
        int idx = z * Nx + x;

        int idx_save =
            k * Nz * Nx +
            z * Nx +
            x;

        float value = save_field[idx_save];

        ilum[idx] += value * value;
    }
}

__global__ void updateIlluminationABC(
    float* ilum,
    const float* save_field,
    const int k,
    const int Nz,
    const int Nx,
    const int Nz_abc,
    const int Nx_abc,
    const int N_abc
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && z < Nz)
    {
        // índice de ilum_partial
        int idx = z * Nx + x;

        // posição correspondente dentro do campo com ABC
        int z_abc = z + N_abc;
        int x_abc = x + N_abc;

        int idx_save =
            k * Nz_abc * Nx_abc +
            z_abc * Nx_abc +
            x_abc;

        float value = save_field[idx_save];

        ilum[idx] += value * value;
    }
}

__global__ void injectAdjointSource(const float* currentbck, float* muted_seismogram, const int* rx, const int* rz, int k, int itlag, int Nrec, int nx_abc, float dx, float dz){
    int irec = blockIdx.x * blockDim.x + threadIdx.x;

    if (irec >= Nrec){
        return;
    }

    int it = k - itlag;
    inv_dxdz = 1.0f/(dx*dz);
    currentbck[rz[irec] * nx_abc + rx[irec]] = (muted_seismogram[it * Nrec + irec] * inv_dxdz);
}

__global__ void calculateAdjointVTIProducts(const float* Uc, const float* P, float* AUc, float* BUc, float* QCxUc, float* QCzUc, const int nx, const int nz, const float dx, const float dz, const float* epsilon, const float* delta)
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

    AUc[i]   = 0.0f;
    BUc[i]   = 0.0f;
    QCxUc[i] = 0.0f;
    QCzUc[i] = 0.0f;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4){
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

        const float eps  = epsilon[i];
        const float delt = delta[i];

        const float px2 = px * px;
        const float pz2 = pz * pz;

        const float px4 = px2 * px2;
        const float pz4 = pz2 * pz2;

        const float num = -2.0f * (eps - delt) * px2 * pz2;

        const float den = (1.0f + 2.0f * eps) * px4 + pz4 + 2.0f * (1.0f + delt) * px2 * pz2;

        float Sd = 0.0f;
        float Cx = 0.0f;
        float Cz = 0.0f;

        if (fabsf(den) > 1.0e-12f)
        {
            Sd = num / den;

            const float inv_den2 = 1 / (den * den);

            const float factor = 4.0f * (eps - delt) * ((1.0f + 2.0f * eps) * px4 - pz4)* inv_den2;

            Cx = factor * px * pz2;
            Cz = -factor * px2 * pz;
        }

        const float A = 1.0f + 2.0f * eps + Sd;
        const float B = 1.0f + Sd;
        const float Q = pxx + pzz;

        AUc[i]   = A * Uc[i];
        BUc[i]   = B * Uc[i];
        QCxUc[i] = Q * Cx * Uc[i];
        QCzUc[i] = Q * Cz * Uc[i];
    }
}

__global__ void updateAdjointWaveEquationVTI(float* Uf, const float* Uc, float* AUc, float* BUc, float* QCxUc, float* QCzUc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* vp)
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
    const float dt2 = dt * dt;

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

__global__ void calculateAdjointTTIProducts(const float* Uc, const float* P, float* AUc, float* BUc, float* HUc, float* QCxUc, float* QCzUc, const int nx, const int nz, const float dx, const float dz, const float* epsilon, const float* delta, const float* theta)
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

    AUc[i]   = 0.0f;
    BUc[i]   = 0.0f;
    HUc[i]   = 0.0f;
    QCxUc[i] = 0.0f;
    QCzUc[i] = 0.0f;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4)
    {
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

        const float eps  = epsilon[i];
        const float delt = delta[i];

        const float xi = px * cosf(theta[i]) - pz * sinf(theta[i]);
        const float eta = px * sinf(theta[i]) + pz * cosf(theta[i]);

        const float xi2  = xi * xi;
        const float eta2 = eta * eta;

        const float xi4  = xi2 * xi2;
        const float eta4 = eta2 * eta2;

        const float num = -2.0f * (eps - delt) * xi2* eta2;
        const float den = (1.0f + 2.0f * eps) * xi4 + eta4 + 2.0f * (1.0f + delt) * xi2 * eta2;

        float Sd = 0.0f;
        float Cx = 0.0f;
        float Cz = 0.0f;

        if (fabsf(den) > 1.0e-12f)
        {
            Sd = num / den;
            const float den2 = den * den;
            const float K = (1.0f + 2.0f * eps) * xi4 - eta4;
            const float factor = 4.0f * (eps - delt) * xi * eta * K / den2;

            Cx = factor * pz;
            Cz = -factor * px;
        }

        const float cos2 = cosf(theta[i]) * cosf(theta[i]);
        const float sin2 = sinf(theta[i]) * sinf(theta[i]);

        const float A = (1.0f + 2.0f * eps) * cos2 + sin2 + Sd;
        const float B = (1.0f + 2.0f * eps) * sin2 + cos2 + Sd;
        const float H = 2.0f * eps * sinf(2.0f * theta[i]);
        const float Q = pxx + pzz;

        AUc[i]   = A * Uc[i];
        BUc[i]   = B * Uc[i];
        HUc[i]   = H * Uc[i];
        QCxUc[i] = Q * Cx * Uc[i];
        QCzUc[i] = Q * Cz * Uc[i];
    }
}    

__global__ void updateAdjointWaveEquationTTI(float* Uf, const float* Uc, const float* AUc, const float* BUc, const float* HUc, const float* QCxUc, const float* QCzUc, const int nx, const int nz, const float dt, const float dx, const float dz, const float* vp)
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


__global__ void calculateGradientVTICuda(const float* current,const float* adj,float* epsilon_partial,float* delta_partial,const float dx,const float dz,const int nx,const int nz,const float* epsilon,const float* delta)
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

    double dSd_deps = 0.0f;
    double dSd_ddelta = 0.0f;
    double dP_deps = 0.0f;
    double dP_ddelta = 0.0f;

    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i/nx;
    int ix = i%nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) 
    {
        double pxx = (c0 * current[i]
                    + c1 * (current[i + 1] + current[i - 1])
                    + c2 * (current[i + 2] + current[i - 2])
                    + c3 * (current[i + 3] + current[i - 3])
                    + c4 * (current[i + 4] + current[i - 4])) * inv_dx2;

        double pzz = (c0 * current[i]
                    + c1 * (current[i + nx] + current[i - nx])
                    + c2 * (current[i + 2*nx] + current[i - 2*nx])
                    + c3 * (current[i + 3*nx] + current[i - 3*nx])
                    + c4 * (current[i + 4*nx] + current[i - 4*nx])) *inv_dz2;

        double px = (a1*(current[i+1] - current[i-1]) +
                    a2*(current[i+2] - current[i-2]) +
                    a3*(current[i+3] - current[i-3]) +
                    a4*(current[i+4] - current[i-4])) * inv_dx;

        double pz = (a1 * (current[i + nx] - current[i - nx]) +
                    a2 * (current[i + 2*nx] - current[i - 2*nx]) +
                    a3 * (current[i + 3*nx] - current[i - 3*nx]) +
                    a4 * (current[i + 4*nx] - current[i - 4*nx])) * inv_dz;
        
        double num = -2.0f*(epsilon[i]-delta[i])*(px*px)*(pz*pz);
        double den = (1.0f + 2.0f*epsilon[i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0f*(1.0f + delta[i])*(px*px)*(pz*pz);

        double dnum_deps = -2.0f*px*px*pz*pz;
        double dnum_ddelta = 2.0f*px*px*pz*pz;
        double dden_deps = 2.0f*px*px*px*px;
        double dden_ddelta = 2.0f*px*px*pz*pz;

        dSd_deps = 0.0f;
        dSd_ddelta = 0.0f;
        if (fabs(den) > 1.0e-150){
            dSd_deps = (dnum_deps*den - num*dden_deps)/(den*den);
            dSd_ddelta = (dnum_ddelta*den - num*dden_ddelta)/(den*den);
        }

        dP_deps = ((-2.0f - dSd_deps)*pxx - dSd_deps*pzz);
        dP_ddelta = (-dSd_ddelta*(pxx + pzz));

        epsilon_partial[i] += adj[i]*dP_deps;
        delta_partial[i] += adj[i]*dP_ddelta;
    
    }
}
   

__global__ void calculateGradientTTICuda(const float* current,const float* adj,float* epsilon_partial,float* delta_partial,float* theta_partial,const float dx,const float dz,const int nx,const int nz,const float* epsilon,const float* delta,const float* theta)
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

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i / nx;
    int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4)
    {
        double pxx = (c0 * current[i]
            + c1 * (current[i + 1] + current[i - 1])
            + c2 * (current[i + 2] + current[i - 2])
            + c3 * (current[i + 3] + current[i - 3])
            + c4 * (current[i + 4] + current[i - 4])) * inv_dx2;

        double pzz = (c0 * current[i]
            + c1 * (current[i + nx] + current[i - nx])
            + c2 * (current[i + 2 * nx] + current[i - 2 * nx])
            + c3 * (current[i + 3 * nx] + current[i - 3 * nx])
            + c4 * (current[i + 4 * nx] + current[i - 4 * nx])) *inv_dz2;

        double pxz = (a1*a1*(current[i + nx + 1] - current[i - nx + 1] + current[i - nx - 1] - current[i + nx - 1])
        + a1*a2*(current[i + 2*nx + 1] - current[i - 2*nx + 1] + current[i - 2*nx - 1] - current[i + 2*nx - 1])
        + a1*a3*(current[i + 3*nx + 1] - current[i - 3*nx + 1] + current[i - 3*nx - 1] - current[i + 3*nx - 1])
        + a1*a4*(current[i + 4*nx + 1] - current[i - 4*nx + 1] + current[i - 4*nx - 1] - current[i + 4*nx - 1])

        + a2*a1*(current[i + nx + 2] - current[i - nx + 2] + current[i - nx - 2] - current[i + nx - 2])
        + a2*a2*(current[i + 2*nx + 2] - current[i - 2*nx + 2] + current[i - 2*nx - 2] - current[i + 2*nx - 2])
        + a2*a3*(current[i + 3*nx + 2] - current[i - 3*nx + 2] + current[i - 3*nx - 2] - current[i + 3*nx - 2])
        + a2*a4*(current[i + 4*nx + 2] - current[i - 4*nx + 2] + current[i - 4*nx - 2] - current[i + 4*nx - 2])

        + a3*a1*(current[i + nx + 3] - current[i - nx + 3] + current[i - nx - 3] - current[i + nx - 3])
        + a3*a2*(current[i + 2*nx + 3] - current[i - 2*nx + 3] + current[i - 2*nx - 3] - current[i + 2*nx - 3])
        + a3*a3*(current[i + 3*nx + 3] - current[i - 3*nx + 3] + current[i - 3*nx - 3] - current[i + 3*nx - 3])
        + a3*a4*(current[i + 4*nx + 3] - current[i - 4*nx + 3] + current[i - 4*nx - 3] - current[i + 4*nx - 3])

        + a4*a1*(current[i + nx + 4] - current[i - nx + 4] + current[i - nx - 4] - current[i + nx - 4])
        + a4*a2*(current[i + 2*nx + 4] - current[i - 2*nx + 4] + current[i - 2*nx - 4] - current[i + 2*nx - 4])
        + a4*a3*(current[i + 3*nx + 4] - current[i - 3*nx + 4] + current[i - 3*nx - 4] - current[i + 3*nx - 4])
        + a4*a4*(current[i + 4*nx + 4] - current[i - 4*nx + 4] + current[i - 4*nx - 4] - current[i + 4*nx - 4])) * inv_dxdz;

        double px = (a1 * (current[i + 1] - current[i - 1])
            + a2 * (current[i + 2] - current[i - 2])
            + a3 * (current[i + 3] - current[i - 3])
            + a4 * (current[i + 4] - current[i - 4])) * inv_dx;

        double pz = (a1 * (current[i + nx] - current[i - nx])
            + a2 * (current[i + 2 * nx] - current[i - 2 * nx])
            + a3 * (current[i + 3 * nx] - current[i - 3 * nx])
            + a4 * (current[i + 4 * nx] - current[i - 4 * nx])) * inv_dz;

        double eps = epsilon[i];
        double del = delta[i];
        double th  = theta[i];

        double costh = cosf(th);
        double sinth = sinf(th);

        double h = px * costh - pz * sinth;
        double q = px * sinth + pz * costh;

        double h2 = h * h;
        double q2 = q * q;

        double num = -2.0f * (eps - del) * h2 * q2;

        double den = (1.0f + 2.0f * eps) * h2 * h2 + q2 * q2 + 2.0f * (1.0f + del) * h2 * q2;

        double dSd_deps = 0.0f;
        double dSd_ddelta = 0.0f;
        double dSd_dtheta = 0.0f;

        if (fabs(den) >= 1.0e-150)
        {
            double dnum_deps = -2.0f * h2 * q2;
            double dnum_ddelta = 2.0f * h2 * q2;

            double dden_deps = 2.0f * h2 * h2;
            double dden_ddelta = 2.0f * h2 * q2;

            double dnum_dtheta = -2.0f * (eps - del) * (-2.0f * h * q * q * q+ 2.0f * h * h * h * q);

            double dden_dtheta = -4.0f * (1.0f + 2.0f * eps) * h * h * h * q + 4.0f * h * q * q * q + 2.0f * (1.0f + del) * (-2.0f * h * q * q * q+ 2.0f * h * h * h * q);

            dSd_deps = (dnum_deps * den - num * dden_deps) / (den * den);

            dSd_ddelta = (dnum_ddelta * den - num * dden_ddelta) / (den * den);

            dSd_dtheta = (dnum_dtheta * den - num * dden_dtheta) / (den * den);
        }

        double sin2th = sinf(2.0f * th);
        double cos2th = cosf(2.0f * th);

        double dA_dtheta = 2.0f * eps * sin2th - dSd_dtheta;

        double dB_dtheta = -2.0f * eps * sin2th - dSd_dtheta;

        double dC_dtheta = 4.0f * eps * cos2th;

        double dP_deps = (-(2.0f * costh * costh + dSd_deps)) * pxx - (2.0f * sinth * sinth + dSd_deps) * pzz + 2.0f * sin2th * pxz;

        double dP_ddelta = -dSd_ddelta * (pxx + pzz);

        double dP_dtheta = dA_dtheta * pxx + dB_dtheta * pzz + dC_dtheta * pxz;

        epsilon_partial[i] += adj[i] * dP_deps;
        delta_partial[i]   += adj[i] * dP_ddelta;
        theta_partial[i]   += adj[i] * dP_dtheta;    
    }
}

