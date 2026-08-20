#include "Migration.cuh"
#include <algorithm>
#include <cmath>

Migration::Migration(Survey* parameters, Modeling* modeling)
{
    pmt = parameters;
    mdl = modeling;
}

void Migration::initializeMigrationFields()
{
    const int n_model = pmt->nx * pmt->nz;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;

    expBlocks=(n_model_exp+nThreads-1)/nThreads;
    seisBlocks=(pmt->Nrec+nThreads-1)/nThreads;
    nBlocks=(n_model+nThreads-1)/nThreads;
    
    cudaStreamCreate(&mdl->copy_stream);
    cudaStreamCreate(&mdl->compute_stream);

    cudaMalloc((void**)&image, n_model * sizeof(float));
    cudaMalloc((void**)&ilum, n_model * sizeof(float));
    cudaMemset(image, 0, n_model * sizeof(float));
    cudaMemset(ilum, 0, n_model * sizeof(float));
    cudaMalloc((void**)&currentbck, n_model_exp * sizeof(float));
    cudaMalloc((void**)&futurebck, n_model_exp * sizeof(float));
    
    if (pmt->migration == "onthefly"){
        cudaMalloc((void**)&savefield,pmt->nt*n_model_exp*sizeof(float));
    }
    if (pmt->migration == "checkpoint"){
        cudaMalloc((void**)&d_current, n_model_exp * sizeof(float));
        cudaMallocHost((void**)&h_current,n_model_exp*sizeof(float));
        cudaMallocHost((void**)&h_current_next ,n_model_exp*sizeof(float));
        cudaMallocHost((void**)&h_future_next,n_model_exp*sizeof(float));
        cudaMallocHost((void**)&h_future,n_model_exp*sizeof(float));
        cudaMalloc((void**)&d_future, n_model_exp * sizeof(float));
    }

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaMalloc((void**)&AUc, n_model_exp * sizeof(float));
        cudaMalloc((void**)&BUc, n_model_exp * sizeof(float));
        cudaMalloc((void**)&QCxUc, n_model_exp * sizeof(float));
        cudaMalloc((void**)&QCzUc, n_model_exp * sizeof(float));
        if (pmt->approximation == "TTI"){
            cudaMalloc((void**)&HUc, n_model_exp * sizeof(float));   
        }
    }
}

void Migration::freeMemory(){
    cudaFree(image);
    cudaFree(ilum);
    cudaFree(currentbck);
    cudaFree(futurebck);
    if (pmt->migration == "checkpoint"){
        cudaFree(d_current);
        cudaFreeHost(h_current);
        cudaFreeHost(h_current_next);
        cudaFreeHost(h_future);
        cudaFreeHost(h_future_next);
        cudaFree(d_future);
    } 
    if (pmt->migration == "onthefly"){
        cudaFree(savefield);
    }
    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaFree(AUc);
        cudaFree(BUc);
        cudaFree(QCxUc);
        cudaFree(QCzUc);
        if (pmt->approximation == "TTI"){
            cudaFree(HUc);   
        }
    }
}

void Migration::resetFields(){
    int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    cudaMemset(currentbck, 0, n_model_exp * sizeof(float));
    cudaMemset(futurebck, 0, n_model_exp * sizeof(float));
}

void Migration::Mute(float* seismogram, const int shot)
{   
    #pragma omp parallel for
    for (int irec = 0; irec < pmt->Nrec; irec++)
    {
        float dz = pmt->rec_z[irec] - pmt->shot_z[shot];
        float dx = pmt->rec_x[irec] - pmt->shot_x[shot];

        float dist = sqrtf(dz * dz + dx * dx);

        float traveltime = (dist / pmt->v0) + pmt->shift;

        float t1 = traveltime;
        float t2 = t1 + pmt->window;
        for (int it = 0; it < pmt->nt_data; it++)
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
    int n_seis = pmt->nt_data * pmt->Nrec;
    float* seismogram_h = new float[n_seis];

    std::ostringstream fcut_stream;
    fcut_stream<<std::fixed<<std::setprecision(1)<<pmt->fcut;
    std::string seismogramFile = pmt->seismogramFolder+"seismogram_shot_"+std::to_string(shot+1)+"_Nt"+std::to_string(pmt->nt_data)+"_Nrec"+std::to_string(pmt->Nrec)+"_fcut"+fcut_stream.str()+".bin";
    mdl->importBin(seismogramFile, seismogram_h, n_seis);
    Mute(seismogram_h, shot);
    
    cudaMemcpy(mdl->seismogram,seismogram_h,n_seis * sizeof(float), cudaMemcpyHostToDevice);
    delete[] seismogram_h;
}

bool* Migration::createMask(const float* f)
{
    const int n_model = pmt->nx * pmt->nz;
    const float f_min = *std::min_element(f, f + n_model);

    bool* mask = new bool[n_model];

    #pragma omp parallel for
    for (int i = 0; i < n_model; i++)
    {
        mask[i] = std::fabs(f[i] - f_min) < 1e-3f;
    }

    return mask;
}

float Migration::gaussianKernel(const int x){
    return std::exp(-(x*x)/(2.0f*pmt->sigma*pmt->sigma));
}


std::vector<float> Migration::gaussianFilter1D(){
    int kernelSize = std::ceil(6.0f*pmt->sigma+1.0f);
    if(kernelSize%2==0){
        kernelSize++;
    }  

    int half = kernelSize/2;
    std::vector<float> kernel(kernelSize);
    float total = 0.0f;
    for(int i=0;i<kernelSize;i++){
        kernel[i]=gaussianKernel(i-half);
        total+=kernel[i];
    }
    for(int i=0;i<kernelSize;i++){
        kernel[i]/=total;
    } 
    return kernel;
}

void Migration::smoothModel(float* f,const bool* mask,const bool parameter){
    const int n=pmt->nz*pmt->nx;
    std::vector<float> s(n);
    std::vector<float> sOld(n);
    std::vector<float> temp(n);

    #pragma omp parallel for
    for(int i=0;i<n;i++){
        if(parameter) sOld[i]=f[i];
        else sOld[i]=1.0f/f[i];
    }

    std::vector<float> kernel = gaussianFilter1D();
    int kernelSize = kernel.size();
    int half = kernelSize/2;

    #pragma omp parallel for
    for(int z = 0; z < pmt->nz; z++){
        for(int x = 0; x < pmt->nx ; x++){
            float newValue = 0.0f;
            for(int i = -half; i<= half; i++){
                int xx = x + i;
                if(xx<0){
                    xx=0;
                } 
                if(xx>=pmt->nx){
                    xx=pmt->nx-1;
                } 
                newValue += kernel[i+half] * sOld[z*pmt->nx+xx];
            }

            temp[z*pmt->nx+x] = newValue;
        }
    }

    #pragma omp parallel for
    for(int z = 0; z < pmt->nz; z++){
        for(int x = 0; x< pmt->nx; x++){
            float newValue = 0.0f;
            for(int i= -half; i<= half; i++){
                int zz = z + i;
                if(zz<0){
                    zz=0;
                } 
                if(zz>=pmt->nz){
                    zz=pmt->nz-1;
                } 
                newValue += kernel[i+half]*temp[zz*pmt->nx+x];
            }
            s[z*pmt->nx+x] = newValue;
        }
    }

    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        if(mask[i]){
            s[i] = sOld[i];
        } 
        if(parameter){
            f[i] = s[i];
        } 
        else{
            f[i] = 1.0f/s[i];
        } 
    }
}

void Migration::backward_step(const int k, float* P){
    if (pmt->approximation == "acoustic"){
        updateAdjointWaveEquation<<<expBlocks, nThreads,0,mdl->compute_stream>>>(futurebck, currentbck, P, image, ilum, mdl->vp, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt, mdl->A, pmt->N_abc);
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
    bool* water_mask = createMask(vp_h);
    smoothModel(vp_h, water_mask, false);
    mdl->expandModel(vp_h, vp_exp_h);
    cudaMemcpy(mdl->vp, vp_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        epsilon_h = new float[n_model]();
        epsilon_exp_h = new float[n_model_exp]();
        mdl->importBin(pmt->epsilonFile, epsilon_h, n_model);
        smoothModel(epsilon_h, water_mask, true);
        mdl->expandModel(epsilon_h, epsilon_exp_h);
        cudaMemcpy(mdl->epsilon, epsilon_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);

        delta_h = new float[n_model]();
        delta_exp_h = new float[n_model_exp]();
        mdl->importBin(pmt->deltaFile, delta_h, n_model);
        smoothModel(delta_h, water_mask, true);
        mdl->expandModel(delta_h, delta_exp_h);
        cudaMemcpy(mdl->delta, delta_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (pmt->approximation == "TTI"){
        theta_h = new float[n_model]();
        theta_exp_h = new float[n_model_exp]();
        mdl->importBin(pmt->thetaFile, theta_h, n_model);
        smoothModel(theta_h, water_mask, true);
        #pragma omp parallel for
        for (int i = 0; i < n_model; i++){
            theta_h[i] = theta_h[i] * pi / 180.0f;
        }
        mdl->expandModel(theta_h, theta_exp_h);
        cudaMemcpy(mdl->theta, theta_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    }

    delete[] vp_h;
    delete[] vp_exp_h;
    delete[] epsilon_h;
    delete[] epsilon_exp_h;
    delete[] delta_h;
    delete[] delta_exp_h;
    delete[] theta_h;
    delete[] theta_exp_h;
    delete[] water_mask;
}

void Migration::saveCheckpoint(const int k){
    const int n_model_exp=pmt->nx_abc*pmt->nz_abc;
    
    std::string checkpointFile=pmt->checkpointFolder+"checkpoint_Nx"+std::to_string(pmt->nx_abc)+"_Nz"+std::to_string(pmt->nz_abc)+"_frame"+std::to_string(k)+".bin";
    std::ofstream file(checkpointFile,std::ios::binary);
    if(!file.is_open()) throw std::invalid_argument("Info: Could not open checkpoint file.");

    file.write((char*)h_current,n_model_exp*sizeof(float));
    file.write((char*)h_future,n_model_exp*sizeof(float));
    
    file.close();
}

void Migration::importCheckpoint(const int k, float* current, float* future){
    const int n_model_exp = pmt->nx_abc*pmt->nz_abc;
    std::string checkpointFile = pmt->checkpointFolder+"checkpoint_Nx"+std::to_string(pmt->nx_abc)+"_Nz"+std::to_string(pmt->nz_abc)+"_frame"+std::to_string(k)+".bin";

    std::ifstream file(checkpointFile,std::ios::binary);
    if(!file.is_open()) throw std::invalid_argument("Info: Could not open checkpoint file.");

    file.read((char*)current,n_model_exp*sizeof(float));
    file.read((char*)future,n_model_exp*sizeof(float));

    file.close();
}

void Migration::saveImage(){
    std::string imageFile=pmt->imageFolder+"image_"+pmt->approximation+"_Nx"+std::to_string(pmt->nx)+"_Nz"+std::to_string(pmt->nz)+".bin";

    const int n_model = pmt->nx*pmt->nz;
    float* image_h = new float[n_model];

    cudaMemcpy(image_h,image,n_model*sizeof(float),cudaMemcpyDeviceToHost);

    std::ofstream file(imageFile,std::ios::binary);
    if(!file.is_open()){
        delete[] image_h;
        throw std::invalid_argument("Info: Could not open file. Please verify the file path.");
    }

    file.write((char*)image_h,n_model*sizeof(float));
    file.close();

    delete[] image_h;

    std::cout<<"Info: File saved to " + imageFile<<std::endl;
}

void Migration::solveReverseTimeMigrationOntheFly(){
    std::cout << "info: Solving " + pmt->approximation + " Reverse Time Migration by " + pmt->migration + " method." << std::endl;

    initializeMigrationFields();
    mdl->createWavelet();
    if (pmt->ABC == "cerjan"){
        mdl->createCerjanVector();
    }
    setModel();
    for (int shot = 0; shot < pmt->Nshot; shot++){
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];
        resetFields();
        mdl->resetFields();
        loadSeismogram(shot);
        const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
        for (int k = 0; k < pmt->nt; k++){
            injectSource <<<1, 1, 0, mdl->compute_stream>>>(mdl->current, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz);
            mdl->forward_step(k);
            cudaMemcpyAsync(savefield + k * n_model_exp,mdl->current,n_model_exp * sizeof(float),cudaMemcpyDeviceToDevice,mdl->copy_stream);
            std::swap(mdl->current, mdl->future);
        }
        cudaStreamSynchronize(mdl->compute_stream);
        for (int t = pmt->nt - 1; t >= 0; t--){
            if (t >= pmt->itlag){
                int it = t - pmt->itlag;
                injectAdjointSource<<<seisBlocks, nThreads,0,mdl->compute_stream>>>(currentbck, mdl->seismogram, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz);
            }
            backward_step(t,savefield + t * n_model_exp);
            std::swap(currentbck, futurebck);
        }
        cudaStreamSynchronize(mdl->compute_stream);
    }

    normalizeImage<<<nBlocks, nThreads>>>(image, ilum, pmt->nx, pmt->nz);   
    saveImage();
    std::cout << "info: Reverse Time Migration" << std::endl;
}

void Migration::solveReverseTimeMigrationCheckpoint(){
    std::cout << "info: Solving " + pmt->approximation + " Reverse Time Migration by " + pmt->migration + " method." << std::endl;
    initializeMigrationFields();
    mdl->createWavelet();
    if (pmt->ABC == "cerjan"){
        mdl->createCerjanVector();
    }
    setModel();
    for (int shot = 0; shot < pmt->Nshot; shot++){
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        mdl->sx = pmt->sx[shot];
        mdl->sz = pmt->sz[shot];
        resetFields();
        mdl->resetFields();
        loadSeismogram(shot);
        const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
        for (int k = 0; k < pmt->nt; k++){
            injectSource <<<1, 1, 0, mdl->compute_stream>>>(mdl->current, mdl->source, k, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz);
            mdl->forward_step(k);
            if (k % pmt->step == 0){
                if (k >= pmt->step){
                    cudaStreamSynchronize(mdl->copy_stream);
                    saveCheckpoint(k - pmt->step);
                }

                cudaStreamSynchronize(mdl->compute_stream);
                cudaMemcpyAsync(d_current,mdl->current,n_model_exp * sizeof(float),cudaMemcpyDeviceToDevice,mdl->copy_stream);
                cudaMemcpyAsync(d_future,mdl->future,n_model_exp * sizeof(float),cudaMemcpyDeviceToDevice,mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                cudaMemcpyAsync(h_current,d_current,n_model_exp * sizeof(float),cudaMemcpyDeviceToHost,mdl->copy_stream);
                cudaMemcpyAsync(h_future,d_future,n_model_exp * sizeof(float),cudaMemcpyDeviceToHost,mdl->copy_stream);
            }
            std::swap(mdl->current, mdl->future);
        }
        cudaStreamSynchronize(mdl->copy_stream);
        cudaStreamSynchronize(mdl->compute_stream);
        for (int t = pmt->nt - 1; t >= 0; t--){
            if (t%pmt->step == 0){
                cudaStreamSynchronize(mdl->compute_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                cudaMemcpyAsync(mdl->current,d_current,n_model_exp * sizeof(float),cudaMemcpyDeviceToDevice,mdl->copy_stream);
                cudaMemcpyAsync(mdl->future,d_future,n_model_exp * sizeof(float),cudaMemcpyDeviceToDevice,mdl->copy_stream);
                cudaStreamSynchronize(mdl->copy_stream);
                }
            if (t >= pmt->itlag){
                int it = t - pmt->itlag;
                injectAdjointSource<<<seisBlocks, nThreads, 0, mdl->compute_stream>>>(currentbck, mdl->seismogram, mdl->rx, mdl->rz, it, pmt->Nrec, pmt->nx_abc, pmt->dx, pmt->dz);
            }
            removeSource<<<1, 1, 0, mdl->compute_stream>>>(mdl->current, mdl->source, t, pmt->nt, pmt->nx_abc, mdl->sx, mdl->sz);
            backward_step(t,mdl->current);
            mdl->forward_step(t);
            std::swap(mdl->current, mdl->future);
            std::swap(currentbck, futurebck);

            if (t%pmt->step == 0){
                int next_t = t - pmt->step;
                if(next_t >=0){
                    importCheckpoint(next_t, h_current_next, h_future_next);
                    cudaMemcpyAsync(d_current, h_current_next, n_model_exp * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
                    cudaMemcpyAsync(d_future, h_future_next, n_model_exp * sizeof(float), cudaMemcpyHostToDevice, mdl->copy_stream);
                }
            }
        }
        cudaStreamSynchronize(mdl->compute_stream);
        cudaStreamSynchronize(mdl->copy_stream);
    }
    normalizeImage<<<nBlocks, nThreads>>>(image, ilum, pmt->nx, pmt->nz);   
    saveImage();
    std::cout << "info: Reverse Time Migration" << std::endl;
}

__global__ void normalizeImage(float* __restrict__ image, const float* __restrict__ ilum, int nx, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_model = nx * nz;

    float eps = 1e-12f;

    if(i<n_model){
        if(ilum[i]>eps){
            image[i]/=ilum[i];
        } 
        else{
            image[i]=0.0f;
        } 
    }       
}

__global__ void removeSource(float* __restrict__ current, const float* __restrict__ source, int k, const int nt, const int nx_abc, const int sx, const int sz){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ((index == 0) && (k < nt)){
        current[sz * nx_abc + sx] -= source[k];
    }
}

__global__ void injectAdjointSource(float* __restrict__ currentbck, const float* __restrict__ seismogram, const int* rx, const int* rz, int t, int Nrec, int nx_abc, float dx, float dz){
    int irec = blockIdx.x * blockDim.x + threadIdx.x;

    if (irec >= Nrec){
        return;
    }

    float inv_dxdz = 1.0f / (dx * dz);  
    currentbck[rz[irec] * nx_abc + rx[irec]] += seismogram[t * Nrec + irec] * inv_dxdz;;
}

__global__ void updateAdjointWaveEquation(float* __restrict__ Uf, float* __restrict__ Uc, float* __restrict__ P, float* __restrict__ image, float* __restrict__ ilum, const float* __restrict__ vp,const int nz,const int nx,const float dz,const float dx,const float dt, float* __restrict__ A, int N_abc){
    const float c0 = -2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);
    const float dt2 = dt * dt;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i/nx;
    int ix = i%nx;

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


        if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc)
        {
            int xf = ix - N_abc;
            int zf = iz - N_abc;

            int nxf = nx - 2 * N_abc;

            int idx = zf * nxf + xf;

            ilum[idx] += P[i] * P[i];
            image[idx] += Uc[i] * P[i];
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

__global__ void calculateAdjointVTIProducts(const float* __restrict__ Uc, const float* __restrict__ P, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc, const int nx, const int nz, const float dx, const float dz, const float* __restrict__ epsilon, const float* __restrict__ delta)
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

__global__ void updateAdjointWaveEquationVTI(float* __restrict__ Uf, float* __restrict__ Uc, float* __restrict__ P, float* __restrict__ image, float* __restrict__ ilum, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp, float* __restrict__ A, int N_abc)
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

        if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc)
        {
            int xf = ix - N_abc;
            int zf = iz - N_abc;

            int nxf = nx - 2 * N_abc;

            int idx = zf * nxf + xf;

            ilum[idx] += P[i] * P[i];
            image[idx] += Uc[i] * P[i];
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

__global__ void calculateAdjointTTIProducts(const float* __restrict__ Uc, const float* __restrict__ P, float* __restrict__ AUc, float* __restrict__ BUc, float* __restrict__ HUc, float* __restrict__ QCxUc, float* __restrict__ QCzUc, const int nx, const int nz, const float dx, const float dz, const float* __restrict__ epsilon, const float* __restrict__ delta, const float* __restrict__ theta)
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
        const float c = cosf(theta[i]);
        const float s = sinf(theta[i]);

        const float xi = px * c - pz * s;
        const float eta = px * s + pz * c;

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

        const float cos2 = c * c;
        const float sin2 = s * s;

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

__global__ void updateAdjointWaveEquationTTI(float* __restrict__ Uf, float* __restrict__ Uc, float* __restrict__ P, float* __restrict__ image, float* __restrict__ ilum, const float* __restrict__ AUc, const float* __restrict__ BUc, const float* __restrict__ HUc, const float* __restrict__ QCxUc, const float* __restrict__ QCzUc, const int nx, const int nz, const float dt, const float dx, const float dz, const float* __restrict__ vp, float* __restrict__ A, int N_abc)
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

        if (ix >= N_abc && ix < nx - N_abc && iz >= N_abc && iz < nz - N_abc)
        {
            int xf = ix - N_abc;
            int zf = iz - N_abc;

            int nxf = nx - 2 * N_abc;

            int idx = zf * nxf + xf;

            ilum[idx] += P[i] * P[i];
            image[idx] += Uc[i] * P[i];
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
