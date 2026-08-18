#include "Modeling.cuh"
#include <omp.h>
#include <iomanip>
#include <sstream>

const float pi = 3.14159265358979323846f;

Modeling::Modeling(Survey* parameters)
{
    pmt = parameters;
}

void Modeling::freeMemory()
{   
    cudaFree(rx);
    cudaFree(rz);
    cudaFree(source);
    cudaFree(vp);
    cudaFree(current);
    cudaFree(future);
    cudaFree(seismogram);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(compute_stream);
    if (pmt->ABC == "cerjan"){
        cudaFree(A);
    }
    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaFree(epsilon);
        cudaFree(delta);
    }
    if (pmt->approximation == "TTI"){
        cudaFree(theta);
    }
    if (pmt->snap == true){
        cudaFreeHost(snapshot);
        cudaFree(d_snapshot);
    }
}

void Modeling::initializeFields()
{
    const int n_model = pmt->nx * pmt->nz;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec * pmt->nt_data;

    expBlocks = (n_model_exp + nThreads - 1) / nThreads;
    BlocksSeis = (pmt->Nrec + nThreads - 1) / nThreads;

    cudaStreamCreate(&copy_stream);
    cudaStreamCreate(&compute_stream);

    cudaMalloc((void**)&source, pmt->nt * sizeof(float));

    if (pmt->ABC == "cerjan"){
        cudaMalloc((void**)&A, pmt->N_abc * sizeof(float));
    }

    cudaMalloc((void**)&vp, n_model_exp * sizeof(float));

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        cudaMalloc((void**)&epsilon,n_model_exp * sizeof(float));
        cudaMalloc((void**)&delta,n_model_exp * sizeof(float));
    }

    if (pmt->approximation == "TTI"){
        cudaMalloc((void**)&theta,n_model_exp * sizeof(float));
    }

    cudaMalloc((void**)&current,n_model_exp * sizeof(float));
    cudaMalloc((void**)&future,n_model_exp * sizeof(float));
    cudaMalloc((void**)&seismogram,n_seis * sizeof(float));

    if (pmt->snap == true){
        cudaMallocHost((void**)&snapshot,n_model * sizeof(float));
        cudaMalloc((void**)&d_snapshot,n_model * sizeof(float));
    }

    cudaMalloc((void**)&rx,pmt->Nrec * sizeof(int));
    cudaMalloc((void**)&rz,pmt->Nrec * sizeof(int));
    cudaMemcpy(rx,pmt->rx.data(),pmt->Nrec * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(rz,pmt->rz.data(),pmt->Nrec * sizeof(int),cudaMemcpyHostToDevice);
}

void Modeling::createWavelet(){
    float tlag = pmt->tlag;
    float dt = pmt->dt;
    float fcut = pmt->fcut;
    float* source_h = new float[pmt->nt]();
    float scale = 1.0f / (pmt->dx * pmt->dz);
    float fc = fcut / (3.0f * sqrtf(pi));
    for (int n = 0; n < pmt->nt; n++){
        float td = n*dt - tlag;

        float arg = pi*pi*pi*fc*fc*td*td;

        source_h[n] = (1.0f - 2.0f*arg)*expf(-arg)*scale;
    }
    cudaMemcpy(source, source_h, pmt->nt * sizeof(float), cudaMemcpyHostToDevice);
    delete[] source_h;
}

void Modeling::importBin(std::string path, float* array, int n){
    std::ifstream file(path, std::ios::in);
    if (!file.is_open()){
        throw std::invalid_argument("Info: Could not open file. Please verify the file path.");
    }
    file.read((char *) array, n * sizeof(float));
    file.close();
}

void Modeling::createCerjanVector(){
    const float sb = 3.0f * pmt->N_abc;
    float* A_h = new float[pmt->N_abc]();
    for (int i = 0; i < pmt->N_abc; i++){
        float fb = (pmt->N_abc - i) / (1.4142f * sb);
        A_h[i] = expf(-fb * fb);
    }
    cudaMemcpy(A, A_h, pmt->N_abc * sizeof(float), cudaMemcpyHostToDevice);
    delete[] A_h;
}

void Modeling::resetFields(){
    int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seis = pmt->Nrec * pmt->nt_data;
    cudaMemset(current, 0, n_model_exp * sizeof(float));
    cudaMemset(future, 0, n_model_exp * sizeof(float));
    cudaMemset(seismogram, 0, n_seis * sizeof(float));
}

void Modeling::expandModel(float* __restrict__ model, float* __restrict__ output){
    int N_abc = pmt->N_abc;
    int nx = pmt->nx;
    int nz = pmt->nz;
    int nx_abc = pmt->nx_abc;

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nz; j++){
        // Centro
        #pragma omp simd
        for (int i = 0; i < nx; i++){
            const int index = (j + N_abc)*nx_abc + (i + N_abc);
            output[index] = model[j*nx + i];
        }
        // Esquerda e Direita
        #pragma omp simd
        for (int i = 0; i < N_abc; i++){
            const int index_left = (j + N_abc)*nx_abc + i;
            output[index_left] = model[j*nx];

            const int index_right = (j + N_abc)*nx_abc + (nx + N_abc + i);
            output[index_right] = model[j*nx + nx - 1];
        }
    }
    
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N_abc; j++){
        // Superior e Inferior
        #pragma omp simd
        for (int i = 0; i < nx; i++){
            const int index_top = j*nx_abc + (i + N_abc);
            output[index_top] = model[i];

            const int index_bottom = (nz + N_abc + j)*nx_abc + (i + N_abc);
            output[index_bottom] = model[(nz - 1)*nx + i];
        }
        // Quinas
        #pragma omp simd
        for (int i = 0; i < N_abc; i++){
            const int index_topleft = j*nx_abc + i;
            output[index_topleft] = model[0];

            const int index_topright = j*nx_abc + (nx + N_abc + i);
            output[index_topright] = model[nx - 1];

            const int index_bottomleft = (nz + N_abc + j)*nx_abc + i;
            output[index_bottomleft] = model[(nz - 1)*nx];
 
            const int index_bottomright = (nz + N_abc + j)*nx_abc + (nx + N_abc + i);
            output[index_bottomright] = model[(nz - 1)*nx + nx - 1];
        }
    }
}

void Modeling::checkDispersionAndStability(const float* __restrict__ vp_h, const float* __restrict__ epsilon_h, const float* __restrict__ delta_h, const float* __restrict__ theta_h){
    int n_model = pmt->nx * pmt->nz;

    float vp_min = std::numeric_limits<float>::infinity();
    float vp_max = 0.0f;

    float dx_lim = 0.0f;
    float dz_lim = 0.0f;
    float dt_lim = 0.0f;

    float epsilon_max = 0.0f;
    float delta_max = 0.0f;

    float max_stability = 0.0f;

    float vpfase_min = std::numeric_limits<float>::infinity();
    float vpfase_max = 0.0f;

    float coeffs_8th[4] = {8.0f/5.0f, -1.0f/5.0f, 8.0f/315.0f, -1.0f/560.0f};

    float soma = 0.0f;
    for (int m = 1; m <= 4; m++){
        soma += coeffs_8th[m - 1] * (1.0f - powf(-1.0f, m));
    }

    int nangles = 181;
    float angle_max = 0.5f * pi;

    if (pmt->approximation == "TTI"){
        nangles = 361;
        angle_max = pi;
    }

    float inv_dx = 1.0f / pmt->dx;
    float inv_dz = 1.0f / pmt->dz;
    float inv_dx2 = inv_dx * inv_dx;
    float inv_dz2 = inv_dz * inv_dz;
    float inv_dx4 = inv_dx2 * inv_dx2;
    float inv_dz4 = inv_dz2 * inv_dz2;

    #pragma omp parallel
    {
        #pragma omp for simd reduction(min:vp_min) reduction(max:vp_max) schedule(static)
        for (int i = 0; i < n_model; i++){
            vp_min = std::min(vp_min, vp_h[i]);
            vp_max = std::max(vp_max, vp_h[i]);
        }

        if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
            #pragma omp for simd reduction(max:epsilon_max) reduction(max:delta_max) reduction(max:max_stability) schedule(static)
            for (int i = 0; i < n_model; i++){
                epsilon_max = std::max(epsilon_max, epsilon_h[i]);
                delta_max = std::max(delta_max, delta_h[i]);

                float num = -2.0f * (epsilon_h[i] - delta_h[i]) * inv_dx2 * inv_dz2;
                float den = (1.0f + 2.0f * epsilon_h[i]) * inv_dx4 + inv_dz4 + 2.0f * (1.0f + delta_h[i]) * inv_dx2 * inv_dz2;

                float Sk = 0.0f;
                if (fabsf(den) > 1e-12f){
                    Sk = num / den;
                }

                float factor = ((1.0f + 2.0f * epsilon_h[i]) + Sk) * inv_dx2 + (1.0f + Sk) * inv_dz2;
                max_stability = std::max(max_stability, vp_h[i] * sqrtf(factor));  
            }

            #pragma omp for reduction(min:vpfase_min) reduction(max:vpfase_max) schedule(static)
            for (int ia = 0; ia < nangles; ia++){
                float phi = angle_max * ia / (nangles - 1);

                for (int i = 0; i < n_model; i++){
                    float beta = phi;

                    if (pmt->approximation == "TTI"){
                        beta = phi - theta_h[i];
                    }

                    float seno = sinf(beta);
                    float cosseno = cosf(beta);

                    float sin2 = seno * seno;
                    float cos2 = cosseno * cosseno;

                    float sin4 = sin2 * sin2;
                    float cos4 = cos2 * cos2;

                    float num = -2.0f * (epsilon_h[i] - delta_h[i]) * sin2 * cos2;
                    float den = (1.0f + 2.0f * epsilon_h[i]) * sin4 + cos4 + 2.0f * (1.0f + delta_h[i]) * sin2 * cos2;

                    float Sk = 0.0f;
                    if (fabsf(den) > 1e-12f){
                        Sk = num / den;
                    }

                    float factor = (1.0f + 2.0f * epsilon_h[i]) * sin2 + cos2 + Sk;
                    float vpfase = vp_h[i] * sqrtf(factor);
                    vpfase_min = std::min(vpfase_min, vpfase);
                    vpfase_max = std::max(vpfase_max, vpfase);    
                }
            }
        }
    }

    if (pmt->approximation == "acoustic"){
        float lambda_min = vp_min / pmt->fcut;
        dx_lim = lambda_min / 4.28f;
        dz_lim = lambda_min / 4.28f;
        dt_lim = dx_lim / (4.0f * vp_max);
    }
    else if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        dt_lim = sqrtf(2.0f) / (sqrtf(soma) * max_stability);

        vp_min = vpfase_min;
        vp_max = vpfase_max;

        float lambda_min = vp_min / pmt->fcut;
        dx_lim = lambda_min / 4.28f;
        dz_lim = lambda_min / 4.28f;

        std::cout<<"info: Maximum epsilon: "<<epsilon_max<<std::endl;
        std::cout<<"info: Maximum delta: "<<delta_max<<std::endl;
    }

    bool dispersion_flag = pmt->dx <= dx_lim && pmt->dz <= dz_lim;
    bool stability_flag = pmt->dt <= dt_lim;

    std::cout<<"info: Dispersion and stability check"<<std::endl;
    std::cout<<"info: Minimum velocity: "<<vp_min<<" m/s"<<std::endl;
    std::cout<<"info: Maximum velocity: "<<vp_max<<" m/s"<<std::endl;
    std::cout<<"info: Maximum frequency: "<<pmt->fcut<<" Hz"<<std::endl;
    std::cout<<"info: Current dx: "<<pmt->dx<<" m"<<std::endl;
    std::cout<<"info: Current dz: "<<pmt->dz<<" m"<<std::endl;
    std::cout<<"info: Current dt: "<<pmt->dt<<" s"<<std::endl;
    std::cout<<"info: Critical dx for dispersion: "<<dx_lim<<" m"<<std::endl;
    std::cout<<"info: Critical dz for dispersion: "<<dz_lim<<" m"<<std::endl;
    std::cout<<"info: Critical dt for stability: "<<dt_lim<<" s"<<std::endl;

    if (dispersion_flag && stability_flag){
        std::cout<<"info: Dispersion and stability conditions satisfied."<<std::endl;
    }
    else{
        std::cout<<"WARNING: Dispersion or stability conditions not satisfied."<<std::endl;
    }
}

void Modeling::setModel(){
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

    importBin(pmt->vpFile, vp_h, n_model);
    expandModel(vp_h, vp_exp_h);
    cudaMemcpy(vp, vp_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        epsilon_h = new float[n_model]();
        epsilon_exp_h = new float[n_model_exp]();
        importBin(pmt->epsilonFile, epsilon_h, n_model);
        expandModel(epsilon_h, epsilon_exp_h);
        cudaMemcpy(epsilon, epsilon_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);

        delta_h = new float[n_model]();
        delta_exp_h = new float[n_model_exp]();
        importBin(pmt->deltaFile, delta_h, n_model);
        expandModel(delta_h, delta_exp_h);
        cudaMemcpy(delta, delta_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (pmt->approximation == "TTI"){
        theta_h = new float[n_model]();
        theta_exp_h = new float[n_model_exp]();
        importBin(pmt->thetaFile, theta_h, n_model);
        #pragma omp parallel for
        for (int i = 0; i < n_model; i++){
            theta_h[i] = theta_h[i] * pi / 180.0f;
        }
        expandModel(theta_h, theta_exp_h);
        cudaMemcpy(theta, theta_exp_h, n_model_exp * sizeof(float), cudaMemcpyHostToDevice);
    }

    checkDispersionAndStability(vp_h, epsilon_h, delta_h, theta_h);

    delete[] vp_h;
    delete[] vp_exp_h;
    delete[] epsilon_h;
    delete[] epsilon_exp_h;
    delete[] delta_h;
    delete[] delta_exp_h;
    delete[] theta_h;
    delete[] theta_exp_h;
}

void Modeling::saveSnapshot(const int shot,const int k){
    const int n_model = pmt->nx*pmt->nz;
    std::string snapshotFile = pmt->snapshotFolder + pmt->approximation + "forward_shot_" + std::to_string(shot + 1) + "Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_Nt" + std::to_string(pmt->nt) + "_frame" + std::to_string(k) + ".bin";
    
    std::ofstream file(snapshotFile,std::ios::binary);
    if (!file.is_open()){
        throw std::invalid_argument("Info: Could not open file. Please verify the file path.");
    }
    file.write((char*)snapshot,n_model*sizeof(float));
    std::cout<<"Info: File saved to " + snapshotFile <<std::endl;
    file.close();
}

void Modeling::saveSeismogram(const int shot){
    std::ostringstream fcut_stream;
    fcut_stream<<std::fixed<<std::setprecision(1)<<pmt->fcut;
    std::string seismogramFile=pmt->seismogramFolder+"seismogram_shot_"+std::to_string(shot+1)+"_Nt"+std::to_string(pmt->nt_data)+"_Nrec"+std::to_string(pmt->Nrec)+"_fcut"+fcut_stream.str()+".bin";

    const int n_seis = pmt->Nrec*pmt->nt_data;
    float* seismogram_h = new float[n_seis];

    cudaMemcpy(seismogram_h,seismogram,n_seis*sizeof(float),cudaMemcpyDeviceToHost);

    std::ofstream file(seismogramFile,std::ios::binary);
    if(!file.is_open()){
        delete[] seismogram_h;
        throw std::invalid_argument("Info: Could not open file. Please verify the file path.");
    }

    file.write((char*)seismogram_h,n_seis*sizeof(float));
    file.close();

    delete[] seismogram_h;

    std::cout<<"Info: File saved to " + seismogramFile<<std::endl;
}

void Modeling::forward_step(const int k){
    if (pmt->approximation == "acoustic"){
        updateWaveEquation<<<expBlocks, nThreads, 0, compute_stream>>>(future, current, vp, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt, A, pmt->N_abc);
    }
    else if (pmt->approximation == "VTI"){
        updateWaveEquationVTI<<<expBlocks, nThreads, 0, compute_stream>>>(future, current, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, vp, epsilon, delta, A, pmt->N_abc);
    }
    else if (pmt->approximation == "TTI"){
        updateWaveEquationTTI<<<expBlocks, nThreads, 0, compute_stream>>>(future, current, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, vp, epsilon, delta, theta, A, pmt->N_abc);
    }
}

void Modeling::solveWaveEquation(){
    std::cout << "info: Solving " + pmt->approximation + " wave equation" << std::endl;
    initializeFields();
    createWavelet();
    if (pmt->ABC == "cerjan"){
        createCerjanVector();
    }
    setModel();
    for (int shot = 0; shot < pmt->Nshot; shot++){
        std::cout << "info: Shot " << shot + 1 << " of " << pmt->Nshot << std::endl;
        sx = pmt->sx[shot];
        sz = pmt->sz[shot];
        resetFields();
        for (int k = 0; k < pmt->nt; k++){
            injectSource <<<1, 1, 0, compute_stream>>>(current, source, k, pmt->nt, pmt->nx_abc, sx, sz);
            forward_step(k);
            if(k>=pmt->itlag){
                storeSeismogram<<<BlocksSeis, nThreads, 0,compute_stream>>>(current, seismogram, rx, rz, k, pmt->itlag, pmt->Nrec, pmt->nx_abc);
            }
            if (pmt->snap && k <= pmt->last_save && k % pmt->step == 0)
            {
                if (k >= pmt->step)
                {
                    cudaStreamSynchronize(copy_stream);
                    saveSnapshot(shot, k - pmt->step);
                }

                cudaStreamSynchronize(compute_stream);

                const float* current_interior = current + pmt->N_abc * pmt->nx_abc + pmt->N_abc;
                cudaMemcpy2DAsync(d_snapshot, pmt->nx * sizeof(float), current_interior, pmt->nx_abc * sizeof(float), pmt->nx * sizeof(float), pmt->nz, cudaMemcpyDeviceToDevice, copy_stream);
                cudaStreamSynchronize(copy_stream);
                cudaMemcpyAsync(snapshot,d_snapshot,pmt->nx * pmt->nz * sizeof(float),cudaMemcpyDeviceToHost,copy_stream);
            }
            std::swap(current, future);
        }
        if (pmt->snap)
        {
            cudaStreamSynchronize(copy_stream);
            saveSnapshot(shot, pmt->last_save);
        }

        cudaStreamSynchronize(compute_stream);
        saveSeismogram(shot);
        std::cout << "info: Wave equation solved" << std::endl;
    }
}

__global__ void injectSource(float* current, const float* source, int k, const int nt, const int nx_abc, const int sx, const int sz){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ((index == 0) && (k < nt)){
        current[sz * nx_abc + sx] += source[k];
    }
}

__global__ void storeSeismogram(const float* current, float* seismogram, const int* rx, const int* rz, int k, int itlag, int Nrec, int nx_abc){
    int irec = blockIdx.x * blockDim.x + threadIdx.x;

    if (irec >= Nrec){
        return;
    }

    int it = k - itlag;
    seismogram[it * Nrec + irec] = current[rz[irec] * nx_abc + rx[irec]];
}

__global__ void updateWaveEquation(float* __restrict__ Uf, float* __restrict__ Uc,const float* __restrict__ vp,const int nz,const int nx,const float dz,const float dx,const float dt, float* __restrict__ A, int N_abc){
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

__global__ void updateWaveEquationVTI(float* __restrict__ Uf, float* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta, float* __restrict__ A, int N_abc){
    const float c0 = -2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;
    const float a1 =  0.8f;
    const float a2 = -0.2f;
    const float a3 =  0.03809523809f;
    const float a4 = -0.00357142857f;
    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
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

        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) * inv_dx;

        float pz = (a1 * (Uc[i + nx] - Uc[i - nx]) +
                    a2 * (Uc[i + 2*nx] - Uc[i - 2*nx]) +
                    a3 * (Uc[i + 3*nx] - Uc[i - 3*nx]) +
                    a4 * (Uc[i + 4*nx] - Uc[i - 4*nx])) * inv_dz;
        
        float num = -2.0f*(epsilon[i]-delta[i])*(px*px)*(pz*pz);
        float den = (1.0f + 2.0f*epsilon[i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0f*(1.0f + delta[i])*(px*px)*(pz*pz);

        float Sd = 0.0f;
        if (fabsf(den)>1e-12f){
            Sd = num / den;
        }
        Uf[i] = 2.0f * Uc[i] - Uf[i] + vp2 * dt2 * ((1.0f+ 2.0f*epsilon[i]) + Sd) * pxx + vp2 * dt2 *(1.0f + Sd) * pzz;

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

__global__ void updateWaveEquationTTI(float* __restrict__ Uf, float* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta,const float* __restrict__ theta, float* __restrict__ A, int N_abc){
    const float c0 = -2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;
    const float a1 =  0.8f;
    const float a2 = -0.2f;
    const float a3 =  0.03809523809f;
    const float a4 = -0.00357142857f;
    const float inv_dx = 1.0f / dx;
    const float inv_dz = 1.0f / dz;
    const float inv_dx2 = 1.0f / (dx * dx);
    const float inv_dz2 = 1.0f / (dz * dz);
    const float inv_dxdz = 1.0f / (dx * dz);
    const float dt2 = dt * dt;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i / nx;
    int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4)
    {
        const float vp2 = vp[i]*vp[i];

        float pxx = (c0 * Uc[i]
                   + c1 * (Uc[i + 1] + Uc[i - 1])
                   + c2 * (Uc[i + 2] + Uc[i - 2])
                   + c3 * (Uc[i + 3] + Uc[i - 3])
                   + c4 * (Uc[i + 4] + Uc[i - 4])) * inv_dx2;

        float pzz = (c0 * Uc[i]
                   + c1 * (Uc[i + nx]     + Uc[i - nx])
                   + c2 * (Uc[i + 2*nx] + Uc[i - 2*nx])
                   + c3 * (Uc[i + 3*nx] + Uc[i - 3*nx])
                   + c4 * (Uc[i + 4*nx] + Uc[i - 4*nx])) * inv_dz2;

        float pxz = (
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

        float px = (a1 * (Uc[i + 1] - Uc[i - 1]) +
                    a2 * (Uc[i + 2] - Uc[i - 2]) +
                    a3 * (Uc[i + 3] - Uc[i - 3]) +
                    a4 * (Uc[i + 4] - Uc[i - 4])) * inv_dx;

        float pz = (a1 * (Uc[i + nx]     - Uc[i - nx]) +
                    a2 * (Uc[i + 2 * nx] - Uc[i - 2 * nx]) +
                    a3 * (Uc[i + 3 * nx] - Uc[i - 3 * nx]) +
                    a4 * (Uc[i + 4 * nx] - Uc[i - 4 * nx])) * inv_dz;

        float th = theta[i];

        float c = cosf(th);
        float s = sinf(th);

        float px_rot = px*c - pz*s;
        float pz_rot = px*s + pz*c;

        float px_rot2 = px_rot*px_rot;
        float pz_rot2 = pz_rot*pz_rot;

        float c2 = c*c;
        float s2 = s*s;
        float sin2th = 2.0f*s*c;

        float num = -2.0f*(epsilon[i] - delta[i])*px_rot2*pz_rot2;

        float den = (1.0f + 2.0f*epsilon[i])*px_rot2*px_rot2 + pz_rot2*pz_rot2 + 2.0f*(1.0f + delta[i])*px_rot2*pz_rot2;

        float Sd = 0.0f;

        if (fabsf(den) > 1e-12f){
            Sd = num/den;
        }

        Uf[i] = 2.0f*Uc[i] - Uf[i] + vp2*dt2*((1.0f + 2.0f*epsilon[i])*c2 + s2 + Sd)*pxx + vp2*dt2*((1.0f + 2.0f*epsilon[i])*s2 + c2 + Sd)*pzz - 2.0f*epsilon[i]*vp2*dt2*sin2th*pxz;
        
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
