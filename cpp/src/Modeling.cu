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
        cudaFree(snapshots);
    }
}

void Modeling::initializeFields()
{
    const int n_model = pmt->nx * pmt->nz;
    const int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    const int n_seismogram = pmt->Nrec * pmt->nt_data;

    nBlocks = (n_model_exp + nThreads - 1) / nThreads;
    nBlocksSeis = (pmt->Nrec + nThreads - 1) / nThreads;

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
    cudaMalloc((void**)&seismogram,n_seismogram * sizeof(float));

    if (pmt->snap == true){
        const int n_snaps = pmt->last_save / pmt->step + 1;
        cudaMalloc((void**)&snapshots, n_snaps * n_model * sizeof(float));
        snap_idx = 0;
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

void Modeling::exportBin(std::string path, float* array, int n){
    float* array_h = new float[n];
    cudaMemcpy(array_h,array,n * sizeof(float),cudaMemcpyDeviceToHost);

    std::ofstream file(path, std::ios::out);
    if (!file.is_open()){
        throw std::invalid_argument("Info: Could not open file. Please verify the file path.");
    }
    file.write((char*) array_h, n * sizeof(float));
    std::cout<<"Info: File saved to " + path <<std::endl;
    file.close();
    delete[] array_h;
}

void Modeling::createCerjanVector(){
    const float sb = 6.0f * pmt->N_abc;
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
    cudaMemset(current, 0, n_model_exp * sizeof(float));
    cudaMemset(future, 0, n_model_exp * sizeof(float));
    if (pmt->snap == true){
        snap_idx = 0;
    }
}

void Modeling::expandModel(float* model, float* output){
    int N_abc = pmt->N_abc;
    int nx = pmt->nx;
    int nz = pmt->nz;
    int nx_abc = pmt->nx_abc;
    int index;

    // Centro
    # pragma omp parallel for
    for (int j = 0; j < nz; j++){
        for (int i = 0; i < nx; i++){
            index = (j + N_abc)*nx_abc + (i + N_abc);
            output[index] = model[j*nx + i];
        }
    }

    // Esquerda
    # pragma omp parallel for
    for (int j = 0; j < nz; j++){
        for (int i = 0; i < N_abc; i++){
            index = (j + N_abc)*nx_abc + i;
            output[index] = model[j*nx];
        }
    }

    // Direita
    # pragma omp parallel for
    for (int j = 0; j < nz; j++){
        for (int i = 0; i < N_abc; i++){
            index = (j + N_abc)*nx_abc + (nx + N_abc + i);
            output[index] = model[j*nx + nx - 1];
        }
    }

    // Superior
    # pragma omp parallel for
    for (int j = 0; j < N_abc; j++){
        for (int i = 0; i < nx; i++){
            index = j*nx_abc + (i + N_abc);
            output[index] = model[i];
        }
    }

    // Inferior
    # pragma omp parallel for
    for (int j = 0; j < N_abc; j++){
        for (int i = 0; i < nx; i++){
            index = (nz + N_abc + j)*nx_abc + (i + N_abc);
            output[index] = model[(nz - 1)*nx + i];
        }
    }

    // Canto superior esquerdo
    # pragma omp parallel for
    for (int j = 0; j < N_abc; j++){
        for (int i = 0; i < N_abc; i++){
            index = j*nx_abc + i;
            output[index] = model[0];
        }
    }

    // Canto superior direito
    # pragma omp parallel for
    for (int j = 0; j < N_abc; j++){
        for (int i = 0; i < N_abc; i++){
            index = j*nx_abc + (nx + N_abc + i);
            output[index] = model[nx - 1];
        }
    }

    // Canto inferior esquerdo
    # pragma omp parallel for
    for (int j = 0; j < N_abc; j++){
        for (int i = 0; i < N_abc; i++){
            index = (nz + N_abc + j)*nx_abc + i;
            output[index] = model[(nz - 1)*nx];
        }
    }

    // Canto inferior direito
    # pragma omp parallel for
    for (int j = 0; j < N_abc; j++){
        for (int i = 0; i < N_abc; i++){
            index = (nz + N_abc + j)*nx_abc + (nx + N_abc + i);
            output[index] = model[(nz - 1)*nx + nx - 1];
        }
    }
}

void Modeling::checkDispersionAndStability(const float* vp_h, const float* epsilon_h, const float* delta_h, const float* theta_h){
    int n_model = pmt->nx * pmt->nz;

    float vp_min = vp_h[0];
    float vp_max = vp_h[0];

    # pragma omp parallel for
    for (int i = 1; i < n_model; i++){
        vp_min = std::min(vp_min, vp_h[i]);
        vp_max = std::max(vp_max, vp_h[i]);
    }

    float dx_lim = 0.0f;
    float dz_lim = 0.0f;
    float dt_lim = 0.0f;

    if (pmt->approximation == "acoustic"){
        float lambda_min = vp_min / pmt->fcut;
        dx_lim = lambda_min / 4.28f;
        dz_lim = lambda_min / 4.28f;
        dt_lim = dx_lim / (4.0f * vp_max);
    }
    else if (pmt->approximation == "VTI" || pmt->approximation == "TTI"){
        float epsilon_max = epsilon_h[0];
        float delta_max = delta_h[0];

        float inv_dx = 1.0f / pmt->dx;
        float inv_dz = 1.0f / pmt->dz;
        float inv_dx2 = inv_dx * inv_dx;
        float inv_dz2 = inv_dz * inv_dz;
        float inv_dx4 = inv_dx2 * inv_dx2;
        float inv_dz4 = inv_dz2 * inv_dz2;

        float coeffs_8th[4] = {8.0f/5.0f, -1.0f/5.0f, 8.0f/315.0f, -1.0f/560.0f};

        float soma = 0.0f;
        for (int m = 1; m <= 4; m++){
            soma += coeffs_8th[m - 1] * (1.0f - powf(-1.0f, m));
        }

        float max_stability = 0.0f;

        # pragma omp parallel for
        for (int i = 0; i < n_model; i++){
            epsilon_max = std::max(epsilon_max, epsilon_h[i]);
            delta_max = std::max(delta_max, delta_h[i]);

            float num = -2.0f * (epsilon_h[i] - delta_h[i]) * inv_dx2 * inv_dz2;
            float den = (1.0f + 2.0f * epsilon_h[i]) * inv_dx4 + inv_dz4 + 2.0f * (1.0f + delta_h[i]) * inv_dx2 * inv_dz2;

            float Sk = 0.0f;
            if (fabsf(den) > 1e-12f){
                Sk = num / den;
            }

            float factor = ((1.0f + 2.0f * epsilon_h[i]) + Sk) / (pmt->dx * pmt->dx) + (1.0f + Sk) / (pmt->dz * pmt->dz);

            if (factor > 0.0f){
                max_stability = std::max(max_stability, vp_h[i] * sqrtf(factor));
            }
        }

        if (max_stability > 0.0f && soma > 0.0f){
            dt_lim = sqrtf(2.0f) / (sqrtf(soma) * max_stability);
        }

        float vqp_min = std::numeric_limits<float>::infinity();
        float vqp_max = 0.0f;

        int nangles = 181;
        float angle_max = 0.5f * pi;

        if (pmt->approximation == "TTI"){
            nangles = 361;
            angle_max = pi;
        }

        # pragma omp parallel for
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

                if (factor >= 0.0f){
                    float vqp = vp_h[i] * sqrtf(factor);
                    vqp_min = std::min(vqp_min, vqp);
                    vqp_max = std::max(vqp_max, vqp);
                }
            }
        }

        vp_min = vqp_min;
        vp_max = vqp_max;

        float lambda_min = vp_min / pmt->fcut;
        dx_lim = lambda_min / 4.28f;
        dz_lim = lambda_min / 4.28f;

        std::cout<<"info: Maximum epsilon: "<<epsilon_max<<std::endl;
        std::cout<<"info: Maximum delta: "<<delta_max<<std::endl;
    }
    else{
        throw std::invalid_argument("ERROR: Unknown approximation. Choose 'acoustic', 'VTI' or 'TTI'.");
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

void Modeling::storeSnapshotGPU(const int k){
    if (!pmt->snap){
        return;
    }
    if (k > pmt->last_save){
        return;
    }
    if (k % pmt->step != 0){
        return;
    }

    int n_model = pmt->nx * pmt->nz;
    int n_snaps = pmt->last_save / pmt->step + 1;

    if (snap_idx >= n_snaps){
        return;
    }

    int nBlocksSnap = (n_model + nThreads - 1) / nThreads;
    storeSnapshot<<<nBlocksSnap, nThreads>>>(current, snapshots, snap_idx, pmt->nx, pmt->nz, pmt->nx_abc, pmt->N_abc);
    snap_idx += 1;
}

void Modeling::saveSnapshot(const int shot){
    if (!pmt->snap){
        return;
    }

    if (snap_idx == 0){
        return;
    }

    int n_model = pmt->nx * pmt->nz;

    for (int i = 0; i < snap_idx; i++){
        int k = i * pmt->step;

        std::string snapshotFile = pmt->snapshotFolder + pmt->approximation + "forward_shot_" + std::to_string(shot + 1) + "_Nx" + std::to_string(pmt->nx) + "_Nz" + std::to_string(pmt->nz) + "_Nt" + std::to_string(pmt->nt) + "_frame_" + std::to_string(k) + ".bin";
        exportBin(snapshotFile,snapshots + i * n_model,n_model);
    }
}

void Modeling::saveSeismogram(const int shot){
    std::ostringstream fcut_stream;
    fcut_stream << std::fixed << std::setprecision(1) << pmt->fcut;
    std::string seismogramFile = pmt->seismogramFolder + "seismogram_shot_" + std::to_string(shot + 1) + "_Nt" + std::to_string(pmt->nt_data) + "_Nrec" + std::to_string(pmt->Nrec) + "_fcut" + fcut_stream.str() + ".bin";
    exportBin(seismogramFile, seismogram, pmt->Nrec * pmt->nt_data);
}

void Modeling::foward_step(const int k){
    injectSource <<<1, 1>>>(current, source, k, pmt->nt, pmt->nx_abc, sx, sz);
    if (pmt->approximation == "acoustic"){
        updateWaveEquation<<<nBlocks, nThreads>>>(future, current, vp, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt);
        AbsorbingBoundary<<<nBlocks, nThreads>>>(future, current, pmt->N_abc, pmt->nz_abc, pmt->nx_abc, A);
    }
    else if (pmt->approximation == "VTI"){
        updateWaveEquationVTI<<<nBlocks, nThreads>>>(future, current, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, vp, epsilon, delta);
        AbsorbingBoundary<<<nBlocks, nThreads>>>(future, current, pmt->N_abc, pmt->nz_abc, pmt->nx_abc, A);
    }
    else if (pmt->approximation == "TTI"){
        updateWaveEquationTTI<<<nBlocks, nThreads>>>(future, current, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, vp, epsilon, delta, theta);
        AbsorbingBoundary<<<nBlocks, nThreads>>>(future, current, pmt->N_abc, pmt->nz_abc, pmt->nx_abc, A);
    }
}

__global__ void injectSource(float* current, const float* source, int k, const int nt, const int nx_abc, const int sx, const int sz){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ((index == 0) && (k < nt)){
        current[sz * nx_abc + sx] += source[k];
    }
}

__global__ void storeSeismogram(const float* current, float* seismogram, const int* rx, const int* rz, int k, int itlag, int nt, int Nrec, int nx_abc){
    int irec = blockIdx.x * blockDim.x + threadIdx.x;
    if (irec >= Nrec){
        return;
    }

    int it = k - itlag;

    seismogram[it * Nrec + irec] = current[rz[irec] * nx_abc + rx[irec]];
}

__global__ void storeSnapshot(const float* current, float* snapshots,int snap_idx, const int nx, const int nz, const int nx_abc, const int N_abc){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int n_model = nx * nz;

    if (i >= n_model){
        return;
    }

    int iz = i / nx;
    int ix = i % nx;

    snapshots[snap_idx * n_model + i] = current[(iz + N_abc) * nx_abc + (ix + N_abc)];
}

__global__ void updateWaveEquation(float* __restrict__ Uf,const float* __restrict__ Uc,const float* __restrict__ vp,const int nz,const int nx,const float dz,const float dx,const float dt){
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
    }
}

__global__ void updateWaveEquationVTI(float* __restrict__ Uf,const float* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta ){
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
    }
}

__global__ void updateWaveEquationTTI(float* __restrict__ Uf,const float* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta,const float* __restrict__ theta){
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
            a4*a4*(Uc[i + 4*nx + 4]   - Uc[i - 4*nx + 4]   + Uc[i - 4*nx - 4]   - Uc[i + 4*nx - 4])
        ) * inv_dxdz;

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
}
}

__global__ void AbsorbingBoundary(float* __restrict__ Uf , float* __restrict__ Uc, int N_abc, int nz, int nx, float* __restrict__ A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;
    if (i >= total_size) return;

    int iz = i / nx;
    int ix = i % nx;

    if (ix < N_abc){
        Uf[i] =  Uf[i] * A[ix];
        Uc[i] =  Uc[i] * A[ix];
    }
    if (ix >=  nx - N_abc){
        Uf[i] =  Uf[i] * A[nx - 1 - ix];
        Uc[i] =  Uc[i] * A[nx - 1 - ix];
    }
    if (iz < N_abc){
        Uf[i] =  Uf[i] * A[iz];
        Uc[i] =  Uc[i] * A[iz];
    }
    if (iz >= nz - N_abc){
        Uf[i] =  Uf[i] * A[nz - 1 - iz];
        Uc[i] =  Uc[i] * A[nz - 1 - iz];
    }
}

void Modeling::solveWaveEquation(){
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
            foward_step(k);
            if (k >= pmt->itlag){
                storeSeismogram<<<nBlocksSeis, nThreads>>>(current, seismogram, rx, rz, k, pmt->itlag, pmt->nt_data, pmt->Nrec, pmt->nx_abc);
            }
            storeSnapshotGPU(k);
            std::swap(current, future);
        }

        saveSeismogram(shot);
        saveSnapshot(shot);
        std::cout << "info: Wave equation solved" << std::endl;
    }
}