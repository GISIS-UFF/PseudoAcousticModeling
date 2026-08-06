#include "Modeling.cuh"

Modeling::Modeling(Survey* parameters)
{
    pmt = parameters;
}

Modeling::deleteCPU()
{
    delete[] source;
    delete[] A;
    delete[] vp;
    delete[] epsilon;
    delete[] delta;
    delete[] theta;
    delete[] vp_exp;
    delete[] epsilon_exp;
    delete[] delta_exp;
    delete[] theta_exp;
    delete[] current;
    delete[] future;
    delete[] seismogram;
}

Modeling::deleteGPU()
{
    cudaFree(source);
    cudaFree(A);
    cudaFree(vp);
    cudaFree(epsilon);
    cudaFree(delta);
    cudaFree(theta);
    cudaFree(vp_exp);
    cudaFree(epsilon_exp);
    cudaFree(delta_exp);
    cudaFree(theta_exp);
    cudaFree(current);
    cudaFree(future);
    cudaFree(seismograms);
}

Modeling::AllocateGPU()
{
    int n_model = pmt->nx * pmt->nz;
    int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    int n_seismogram = pmt->Nrec * pmt->nt;

    cudaMalloc((void**)&source, pmt->nt * sizeof(float));
    cudaMalloc((void**)&A, pmt->N_abc * sizeof(float));

    cudaMalloc((void**)&vp, n_model * sizeof(float));
    cudaMalloc((void**)&epsilon, n_model * sizeof(float));
    cudaMalloc((void**)&delta, n_model * sizeof(float));
    cudaMalloc((void**)&theta, n_model * sizeof(float));

    cudaMalloc((void**)&vp_exp, n_model_exp * sizeof(float));
    cudaMalloc((void**)&epsilon_exp, n_model_exp * sizeof(float));
    cudaMalloc((void**)&delta_exp, n_model_exp * sizeof(float));
    cudaMalloc((void**)&theta_exp, n_model_exp * sizeof(float));

    cudaMalloc((void**)&current, n_model_exp * sizeof(float));
    cudaMalloc((void**)&future, n_model_exp * sizeof(float));

    cudaMalloc((void**)&seismograms, n_seismogram * pmt->Nrec * sizeof(float));
}

void Modeling::initializeFields(){
    int n_model = pmt->nx * pmt->nz;
    int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    int n_seismogram = pmt->Nrec * pmt->nt;
    
    source = new float[pmt->nt]();

    vp = new float[n_model]();
    vp_exp = new float[n_model_exp]();

    current = new float[n_model_exp]();
    future = new float[n_model_exp]();

    seismogram = new float[n_seismogram]();
    seismograms = new float[n_seismogram * pmt->Nrec]();

    nBlocks = (int)((n_model_exp + NTHREADS - 1) / NTHREADS);

    if (pmt->approximation == "VTI" || pmt->approximation == "TTI") {
        
        epsilon = new float[n_model]();
        delta = new float[n_model]();
        theta = new float[n_model]();

        epsilon_exp = new float[n_model_exp]();
        delta_exp = new float[n_model_exp]();
        theta_exp = new float[n_model_exp]();

    }

    if (pmt->approximation == "TTI") {

        theta = new float[n_model]();
        theta_exp = new float[n_model_exp]();
    }
}

void Modeling::createWavelet(){
    float tlag = pmt->tlag;
    float dt = pmt->dt;
    float fcut = pmt->fcut;

    float scale = 1.0f / (pmt->dx * pmt->dz);
    float fc = fcut / (3.0f * sqrtf(pi));
    for (int n = 0; n < pmt->nt; n++){
        float td = n*dt - tlag;

        float arg = pi*pi*pi*fc*fc*td*td;

        source[n] = (1.0f - 2.0f*arg)*expf(-arg)*scale;
    }
}

void Modeling::importBin(std::string path, float* array, int n){
    std::ifstream file(path, std::ios::in);
    if (!file.is_open()){
        throw std::invalid_argument("Could not open file. Please verify the file path.");
    }

    file.read((char *) array, n * sizeof(float));
    file.close();
}

void Modeling::exportBin(std::string path, float* array, int n){
    std::ofstream file(path, std::ios::out);
    if (!file.is_open()){
        throw std::invalid_argument("Could not open file. Please verify the file path.");
    }
    file.write((char*) array, n * sizeof(float));
    std::cout<<"File saved to the path" + path <<std::endl;
    file.close();
}

void Modeling::expandModel(float* model, float* output){

    int N_abc = pmt->N_abc;
    int nx = pmt->nx;
    int nz = pmt->nz;
    int nx_abc = pmt->nx_abc;
    int nz_abc = pmt->nz_abc;
    int index;

    // Centro
    for (int j = 0; j < nz; j++){
        for (int i = 0; i < nx; i++){
            index = (j + N_abc)*nx_abc + (i + N_abc);
            output[index] = model[j*nx + i];
        }
    }
    // Esquerda
    for (int j = 0; j < nz; j++){
        for (int i = 0; i < N_abc; i++){
            index = (j + N_abc)*nx_abc + i;
            output[index] = model[j*nx + i];
        }
    }
    // Direita
    for (int j = 0; j < nz; j++){
        for (int i = nx - N_abc; i < nx; i++){
            index = (j + N_abc)*nx_abc + (i + 2*N_abc);
            output[index] = model[j*nx + i];
        }
    }
    // Superior
    for (int j = 0; j < N_abc; j++){
        for (int i = 0; i < nx; i++){
            index = j*nx_abc + (i + N_abc);
            output[index] = model[j*nx + i];
        }
    }
    // Inferior
    for (int j = nz - N_abc; j < nz; j++){
        for (int i = 0; i < nx; i++){
            index = (j + 2*N_abc)*nx_abc + (i + N_abc);
            output[index] = model[j*nx + i];
        }
    }
    // Canto superior esquerdo
    for (int j = 0; j < N_abc; j++){
        for (int i = 0; i < N_abc; i++){
            index = j *nx_abc + i;
            output[index] = model[j*nx + i];
        }
    }
    // Canto superior direito
    for (int j = 0; j < N_abc; j++){
        for (int i = nx - N_abc; i < nx; i++){
            index = j*nx_abc + (i + 2*N_abc);
            output[index] = model[j*nx + i];
        }
    }
    // Canto inferior esquerdo
    for (int j = nz - N_abc; j < nz; j++){
        for (int i = 0; i < N_abc; i++){
            index = (j + 2*N_abc)*nx_abc + i;
            output[index] = model[j*nx + i];
        }
    }
    // Canto inferior direito
    for (int j = nz - N_abc; j < nz; j++){
        for (int i = nx - N_abc; i < nx; i++){
            index = (j + 2*N_abc)*nx_abc + (i + 2*N_abc);
            output[index] = model[j*nx + i];
        }
    }
}

void Modeling::reduceModel(const float* model_exp, float* output)
{
    int N_abc = pmt->N_abc;
    int nx = pmt->nx;
    int nz = pmt->nz;
    int nx_abc = pmt->nx_abc;

    for (int j = 0; j < nz; j++){
        for (int i = 0; i < nx; i++){
            int index = (j + N_abc) * nx_abc + (i + N_abc);
            output[j * nx + i] = model_exp[index];
        }
    }
}

void Modeling::createCerjanVector(){
    const float sb = 6.0f * pmt->N_abc;
    float* A = new float[pmt->N_abc]();
    for (int i = 0; i < pmt->N_abc; i++){
        float fb = (pmt->N_abc - i) / (1.4142f * sb);
        A[i] = expf(-fb * fb);
    }
}

void Modeling::resetFields(){
    int n_model_exp = pmt->nx_abc * pmt->nz_abc;
    int n_seismogram = pmt->Nrec * pmt->nt;

    cudaMemset(current, 0, n_model_exp * sizeof(float));
    cudaMemset(future, 0, n_model_exp * sizeof(float));
    cudaMemset(seismograms, 0, n_seismogram * pmt->Nrec * sizeof(float));
}

void Modeling::fowardstep(const int k){
    if (pmt->approximation == "acoustic"){
        current[isz * nx_abc + isx] += source[k];
        updateWaveEquation<<<nBlocks, nThreads>>>(future, current, vp, pmt->nz_abc, pmt->nx_abc, pmt->dz, pmt->dx, pmt->dt);
        AbsorbingBoundary<<<nBlocks, nThreads>>>(future, current, pmt->N_abc, pmt->nz_abc, pmt->nx_abc, A);
    }
    if (pmt->approximation == "VTI"){
        current[isz * nx_abc + isx] += source[k];
        updateWaveEquationVTICuda<<<nBlocks, nThreads>>>(future, current, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, vp_exp, epsilon_exp, delta_exp);
        AbsorbingBoundary<<<nBlocks, nThreads>>>(future, current, pmt->N_abc, pmt->nz_abc, pmt->nx_abc, A);
    }
    if (pmt->approximation == "TTI"){
        current[isz * nx_abc + isx] += source[k];
        updateWaveEquationTTICuda<<<nBlocks, nThreads>>>(future, current, pmt->nx_abc, pmt->nz_abc, pmt->dt, pmt->dx, pmt->dz, vp_exp, epsilon_exp, delta_exp, theta_exp);
        AbsorbingBoundary<<<nBlocks, nThreads>>>(future, current, pmt->N_abc, pmt->nz_abc, pmt->nx_abc, A);
    }
}

void Modeling::storeSeismogram(const int k){
    int it = k - pmt->itlag;

    if (it < 0){
        return;
    }

    seismograms[it * pmt->Nrec] = current[rx * pmt->Nrec + rz];
}



__global__ void updateWaveEquation(float* __restrict__ Uf,const float* __restrict__ Uc,const float* __restrict__ vp,const int nz,const int nx,const float dz,const float dx,const float dt)
{
    const float c0 = 2.847222222222f;
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

__global__ void updateWaveEquationVTICuda(float* __restrict__ Uf,const float* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta )
{
    const float c0 = 2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;
    const float a1 =  0.8f;
    const float a2 = -0.2f;
    const float a3 =  0.03809523809f;
    const float a4 = -0.00357142857f;
    const float inv_dx2 = 1.0f / dx;
    const float inv_dz2 = 1.0f / dz;
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

__global__ void updateWaveEquationTTICuda(float* __restrict__ Uf,const float* __restrict__ Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* __restrict__ vp,const float* __restrict__ epsilon,const float* __restrict__ delta,const float* __restrict__ theta)
{
    const float c0 = 2.847222222222f;
    const float c1 =  1.6f;
    const float c2 = -0.2f;
    const float c3 =  0.02539682539f;
    const float c4 = -0.00178571428f;
    const float a1 =  0.8f;
    const float a2 = -0.2f;
    const float a3 =  0.03809523809f;
    const float a4 = -0.00357142857f;
    const float inv_dx2 = 1.0f / dx;
    const float inv_dz2 = 1.0f / dz;
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

        float num = -2.0f * (epsilon[i] - delta[i]) * ((px * cosf(th) - pz * sinf(th)) * (px * cosf(th) - pz * sinf(th))) * ((px * sinf(th) + pz * cosf(th)) * (px * sinf(th) + pz * cosf(th)));
        float den = (1.0f + 2.0f * epsilon[i]) * ((px * cosf(th) - pz * sinf(th)) * (px * cosf(th) - pz * sinf(th)) * (px * cosf(th) - pz * sinf(th)) * (px * cosf(th) - pz * sinf(th))) + ((px * sinf(th) + pz * cosf(th)) * (px * sinf(th) + pz * cosf(th)) *(px * sinf(th) + pz * cosf(th)) * (px * sinf(th) + pz * cosf(th))) + 2.0f * (1.0f + delta[i]) * ((px * cosf(th) - pz * sinf(th)) * (px * cosf(th) - pz * sinf(th))) * ((px * sinf(th) + pz * cosf(th)) * (px * sinf(th) + pz * cosf(th)));

        float Sd = 0.0f;
        if (fabsf(den) > 1e-12f) {
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i]+ vp2 * dt2 * (((1.0f + 2.0f * epsilon[i]) * (cosf(th) * cosf(th)) + (sinf(th) * sinf(th)) + Sd) * pxx) + vp2 * dt2 * (((1.0f + 2.0f * epsilon[i]) * (sinf(th) * sinf(th)) + (cosf(th) * cosf(th)) + Sd) * pzz) - 2.0f * epsilon[i] * vp2 * dt2 * sinf(2.0f * th) * pxz;
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