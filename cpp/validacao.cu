#include "Survey.hpp"
#include "Modeling.cuh"
#include "Migration.cuh"
#include "Inversion.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

const int NX = 31;
const int NZ = 31;
const int ABC = 20;
const int NT_DATA = 1601;
const float DX = 10.0f;
const float DZ = 10.0f;
const float DT = 0.0005f;
const float FCUT = 20.0f;
const float VP0 = 2000.0f;
const float EPS0 = 0.10f;
const float DEL0 = 0.05f;
const double H = 0.01;

const std::string folder = "../outputs/validacao/";
const std::string parameters[3] = {"q", "epsilon", "delta"};
const double scale[3] = {1.0 / (1500.0 * 1500.0), 0.3, 0.2};


void saveBin(const std::string& filename, const std::vector<float>& data, const int nz, const int nx){
    if(data.size() != size_t(nz)*nx){
        throw std::runtime_error("Invalid binary data size.");
    }

    std::ofstream file(folder + filename + ".bin", std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("Could not open binary file.");
    }

    int dimensions[2] = {nz, nx};
    file.write((char*)dimensions, 2*sizeof(int));
    file.write((char*)data.data(), data.size()*sizeof(float));
    if(!file){
        throw std::runtime_error("Error writing binary file.");
    }
    file.close();
}

double calculateNorm(const std::vector<float>& data){
    double sum = 0.0;
    for(int i = 0; i < int(data.size()); i++){
        if(!std::isfinite(data[i])){
            throw std::runtime_error("NaN or Inf found.");
        }
        sum += double(data[i])*data[i];
    }
    return std::sqrt(sum);
}

double relativeError(const std::vector<float>& a, const std::vector<float>& b){
    if(a.size() != b.size()){
        throw std::runtime_error("Arrays with different sizes.");
    }

    double sum = 0.0;
    for(int i = 0; i < int(a.size()); i++){
        double diff = double(a[i]) - b[i];
        sum += diff*diff;
    }

    double denominator = std::max(calculateNorm(a), calculateNorm(b));
    if(denominator == 0.0){
        return 0.0;
    }
    return std::sqrt(sum)/denominator;
}


std::vector<float> getParameter(const std::vector<float>& model, const int parameter){
    const int n_model = NX*NZ;
    std::vector<float> result(n_model);
    for(int i = 0; i < n_model; i++){
        result[i] = model[parameter*n_model + i];
    }
    return result;
}

void saveMetrics(std::ofstream& csv, const std::string& test, const int input, const int output, const double h, const std::vector<float>& result, const std::vector<float>& fd){
    double error = relativeError(result, fd);
    csv << test << "," << input << "," << output << "," << h << "," << error << "," << calculateNorm(result) << "," << calculateNorm(fd) << "\n";
    csv.flush();
    std::cout << "info: " << test << " " << parameters[input] << " -> " << parameters[output] << " h=" << h << " error=" << error << std::endl;
}

void setupSurvey(Survey& pmt, const bool receiversOnEdges){
    pmt.approximation = "VTI";
    pmt.multiparameter = true;
    pmt.migration = "onthefly";
    pmt.snap = false;

    pmt.nx = NX;
    pmt.nz = NZ;
    pmt.N_abc = ABC;
    pmt.nx_abc = NX + 2*ABC;
    pmt.nz_abc = NZ + 2*ABC;
    pmt.dx = DX;
    pmt.dz = DZ;
    pmt.dt = DT;
    pmt.fcut = FCUT;
    pmt.nt_data = NT_DATA;

    const float pi = 3.14159265358979323846f;
    pmt.tlag = 2.0f*std::sqrt(pi)/FCUT;
    pmt.itlag = int(std::round(pmt.tlag/DT));
    pmt.nt = pmt.nt_data + pmt.itlag;

    pmt.vmin = 1500.0f;
    pmt.vmax = 3000.0f;
    pmt.epsmin = 0.0f;
    pmt.epsmax = 0.3f;
    pmt.deltamin = 0.0f;
    pmt.deltamax = 0.2f;

    pmt.Nshot = 1;
    pmt.sx = {ABC + NX/2};
    pmt.sz = {ABC + NZ/2};
    pmt.rx.clear();
    pmt.rz.clear();

    if(receiversOnEdges){
        for(int x = 0; x < NX; x++){
            pmt.rx.push_back(ABC+x);
            pmt.rz.push_back(ABC);
            pmt.rx.push_back(ABC+x);
            pmt.rz.push_back(ABC+NZ-1);
        }
        for(int z = 1; z < NZ-1; z++){
            pmt.rx.push_back(ABC);
            pmt.rz.push_back(ABC+z);
            pmt.rx.push_back(ABC+NX-1);
            pmt.rz.push_back(ABC+z);
        }
    }
    else{
        pmt.rx = {ABC, ABC+NX-1};
        pmt.rz = {ABC, ABC+NZ-1};
    }

    pmt.Nrec = int(pmt.rx.size());
    pmt.seismogramFolder = folder;
}

void initializeFields(Modeling& mdl, Migration& mgt, Inversion& inv){
    mdl.initializeFields();
    mgt.initializeMigrationFields();
    cudaFree(mgt.ilum);
    mgt.ilum = nullptr;
    inv.InitializeInversionFields();
    inv.water_mask = nullptr;
    mdl.createCerjanVector();
    mdl.createWavelet();
    
}

std::vector<float> createInitialModel(){
    const int n_model = NX*NZ;
    std::vector<float> model(3*n_model);
    for(int i = 0; i < n_model; i++){
        model[i] = 1.0f/(VP0*VP0);
        model[n_model+i] = EPS0;
        model[2*n_model+i] = DEL0;
    }
    return model;
}

void setModel(Inversion& inv, const std::vector<float>& model){
    const int n_model = NX*NZ;
    inv.ExpandModelDevice("vp", model.data());
    inv.ExpandModelDevice("epsilon", model.data()+n_model);
    inv.ExpandModelDevice("delta", model.data()+2*n_model);
    
}

std::vector<float> forwardModeling(Survey& pmt, Modeling& mdl, Inversion& inv, const std::vector<float>& model){
    setModel(inv, model);
    mdl.resetFields();
    mdl.sx = pmt.sx[0];
    mdl.sz = pmt.sz[0];

    slowness2ToVp<<<mdl.expBlocks, nThreads, 0, mdl.compute_stream>>>(inv.slowness2, mdl.vp, pmt.nx_abc, pmt.nz_abc);
    for(int k = 0; k < pmt.nt; k++){
        mdl.forward_step();
        injectSource<<<1, 1, 0, mdl.compute_stream>>>(mdl.future, mdl.source, k, pmt.nt, pmt.nx_abc, mdl.sx, mdl.sz, pmt.dx, pmt.dz, pmt.dt);
        if(k >= pmt.itlag){
            storeSeismogram<<<mdl.seisBlocks, nThreads, 0, mdl.compute_stream>>>(mdl.current, mdl.seismogram, mdl.rx, mdl.rz, k, pmt.itlag, pmt.Nrec, pmt.nx_abc);
        }
        std::swap(mdl.current, mdl.future);
    }

    
    const int n_seis = pmt.nt_data*pmt.Nrec;
    std::vector<float> seismogram_h(n_seis);
    cudaMemcpy(seismogram_h.data(), mdl.seismogram, n_seis*sizeof(float), cudaMemcpyDeviceToHost);
    
    return seismogram_h;
}

double objectiveFunction(const std::vector<float>& calculated, const std::vector<float>& observed){
    if(calculated.size() != observed.size()){
        throw std::runtime_error("Seismograms with different sizes.");
    }

    double X = 0.0;
    for(int i = 0; i < int(calculated.size()); i++){
        double residual = double(calculated[i]) - observed[i];
        X += 0.5*residual*residual;
    }
    return X;
}


std::string saveObservedData(const Survey& pmt, const std::vector<float>& observed){
    std::ostringstream fcut;
    fcut << std::fixed << std::setprecision(1) << pmt.fcut;
    std::string filename = pmt.seismogramFolder + "seismogram_shot_1_Nt" + std::to_string(pmt.nt_data) + "_Nrec" + std::to_string(pmt.Nrec) + "_fcut" + fcut.str() + ".bin";

    std::ofstream file(filename, std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("Could not save observed seismogram.");
    }
    file.write((char*)observed.data(), observed.size()*sizeof(float));
    if(!file){
        throw std::runtime_error("Error writing observed seismogram.");
    }
    file.close();
    return filename;
}

std::vector<float> calculateGradient(Inversion& inv, const std::vector<float>& model){
    setModel(inv, model);
    std::vector<float> gradient;
    inv.calculateMultiparameterGradient(true, true, gradient, false);
    return gradient;
}

std::vector<float> bornModeling(Survey& pmt, Modeling& mdl, Migration& mgt, Inversion& inv, const std::vector<float>& model, const std::vector<float>& direction, const bool saveFields){
    const int n_model_exp = pmt.nx_abc*pmt.nz_abc;
    const int n_seis = pmt.nt_data*pmt.Nrec;
    setModel(inv, model);
    inv.setBornPerturbation(direction);
    mdl.resetFields();
    cudaMemset(inv.current_born, 0, n_model_exp*sizeof(float));
    cudaMemset(inv.future_born, 0, n_model_exp*sizeof(float));
    mdl.sx = pmt.sx[0];
    mdl.sz = pmt.sz[0];

    slowness2ToVp<<<mdl.expBlocks, nThreads, 0, mdl.compute_stream>>>(inv.slowness2, mdl.vp, pmt.nx_abc, pmt.nz_abc);
    for(int k = 0; k < pmt.nt; k++){
        updateWaveEquationVTIBornTGN<<<mdl.expBlocks, nThreads, 0, mdl.compute_stream>>>(inv.future_born, inv.current_born, mdl.future, mdl.current, mdl.vp, mdl.epsilon, mdl.delta, inv.dvp, inv.deps, inv.ddelta, pmt.nz_abc, pmt.nx_abc, pmt.dz, pmt.dx, pmt.dt, mdl.A, pmt.N_abc);
        injectSource<<<1, 1, 0, mdl.compute_stream>>>(mdl.future, mdl.source, k, pmt.nt, pmt.nx_abc, mdl.sx, mdl.sz, pmt.dx, pmt.dz, pmt.dt);
        if(k >= pmt.itlag){
            storeSeismogram<<<mdl.seisBlocks, nThreads, 0, mdl.compute_stream>>>(inv.current_born, mdl.seismogram, mdl.rx, mdl.rz, k, pmt.itlag, pmt.Nrec, pmt.nx_abc);
        }
        if(saveFields){
            cudaMemcpyAsync(mgt.savefield + size_t(k)*n_model_exp, mdl.current, n_model_exp*sizeof(float), cudaMemcpyDeviceToDevice, mdl.compute_stream);
        }
        std::swap(mdl.current, mdl.future);
        std::swap(inv.current_born, inv.future_born);
    }

    std::vector<float> seismogram_h(n_seis);
    cudaMemcpy(seismogram_h.data(), mdl.seismogram, n_seis*sizeof(float), cudaMemcpyDeviceToHost);
    
    return seismogram_h;
}

std::vector<float> hessianVectorProduct(Survey& pmt, Modeling& mdl, Migration& mgt, Inversion& inv, const std::vector<float>& model, const std::vector<float>& direction){
    bornModeling(pmt, mdl, mgt, inv, model, direction, true);
    inv.resetGradients();
    mgt.resetFields();

    const int n_model_exp = pmt.nx_abc*pmt.nz_abc;
    const int n_model = NX*NZ;
    for(int t = pmt.nt-1; t >= 0; t--){
        float* current = mgt.savefield + size_t(t)*n_model_exp;
        float* past = mgt.savefield + size_t(std::max(t-1,0))*n_model_exp;
        float* future = mgt.savefield + size_t(std::min(t+1,pmt.nt-1))*n_model_exp;
        inv.backward_step(current, past, future, false);
        if(t >= pmt.itlag){
            injectAdjointSource<<<mdl.seisBlocks, nThreads, 0, mdl.compute_stream>>>(mgt.futurebck, mdl.seismogram, mdl.rx, mdl.rz, t-pmt.itlag, pmt.Nrec, pmt.nx_abc, pmt.dx, pmt.dz, pmt.dt, mdl.vp);
        }
        std::swap(mgt.currentbck, mgt.futurebck);
    }

    std::vector<float> result(3*n_model);
    cudaMemcpy(result.data(), mgt.image, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.data()+n_model, inv.eps_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.data()+2*n_model, inv.delta_grad, n_model*sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int j = 0; j < 3; j++){
        for(int i = 0; i < n_model; i++){
            result[j*n_model+i] *= -scale[j];
        }
    }
    return result;
}

void validateGradient(std::ofstream& csv){
    std::cout << "\ninfo: Gradient validation." << std::endl;
    Survey pmt;
    setupSurvey(pmt, false);
    Modeling mdl(&pmt);
    Migration mgt(&pmt, &mdl);
    Inversion inv(&pmt, &mdl, &mgt);
    initializeFields(mdl, mgt, inv);

    const int n_model = NX*NZ;
    std::vector<float> model = createInitialModel();
    std::vector<float> d0 = forwardModeling(pmt, mdl, inv, model);
    std::vector<float> d1 = forwardModeling(pmt, mdl, inv, model);
    csv << "repeat,-1,-1,0," << relativeError(d0,d1) << "," << calculateNorm(d0) << "," << calculateNorm(d1) << "\n";

    std::vector<float> trueModel = model;
    for(int z = 0; z < NZ; z++){
        for(int x = 0; x < NX; x++){
            double r2 = (x-10.0)*(x-10.0) + (z-15.0)*(z-15.0);
            float value = float(std::exp(-r2/15.0));
            int i = z*NX+x;
            for(int j = 0; j < 3; j++){
                trueModel[j*n_model+i] += float(0.01*scale[j]*value);
            }
        }
    }

    std::vector<float> observed = forwardModeling(pmt, mdl, inv, trueModel);
    std::string observedFile = saveObservedData(pmt, observed);
    std::vector<float> gradient = calculateGradient(inv, model);

    for(int j = 0; j < 3; j++){
        std::vector<float> fd(n_model);
        std::vector<float> plus = model;
        std::vector<float> minus = model;
        for(int i = 0; i < n_model; i++){
            int k = j*n_model+i;
            plus[k] = float(double(model[k]) + H*scale[j]);
            minus[k] = float(double(model[k]) - H*scale[j]);
            double realStep = double(plus[k])-minus[k];
            double fp = objectiveFunction(forwardModeling(pmt,mdl,inv,plus), observed);
            double fm = objectiveFunction(forwardModeling(pmt,mdl,inv,minus), observed);
            fd[i] = float((fp-fm)/realStep);
            plus[k] = model[k];
            minus[k] = model[k];
            if(i%100 == 0){
                std::cout << "info: Gradient " << parameters[j] << " " << i << "/" << n_model << std::endl;
            }
        }

        std::vector<float> adj = getParameter(gradient,j);
        saveBin("gradiente_" + parameters[j] + "_operator", adj, NZ, NX);
        saveBin("gradiente_" + parameters[j] + "_fd", fd, NZ, NX);
        saveMetrics(csv, "gradient", j, j, H, adj, fd);
    }

    std::remove(observedFile.c_str());
    inv.freeMemory();
    mgt.freeMemory();
    mdl.freeMemory();
}

void validateBornAndHessian(std::ofstream& csv){
    std::cout << "\ninfo: Born and Hessian validation." << std::endl;
    Survey pmt;
    setupSurvey(pmt, true);
    Modeling mdl(&pmt);
    Migration mgt(&pmt, &mdl);
    Inversion inv(&pmt, &mdl, &mgt);
    initializeFields(mdl, mgt, inv);

    const int n_model = NX*NZ;
    const int point = 10*NX+10;
    std::vector<float> model = createInitialModel();

    for(int j = 0; j < 3; j++){
        std::vector<float> direction(3*n_model,0.0f);
        direction[j*n_model+point] = 1.0f;
        std::vector<float> bornData = bornModeling(pmt,mdl,mgt,inv,model,direction,false);
        std::vector<float> plus = model;
        std::vector<float> minus = model;
        plus[j*n_model+point] = float(double(model[j*n_model+point]) + H*scale[j]);
        minus[j*n_model+point] = float(double(model[j*n_model+point]) - H*scale[j]);
        double realStep = (double(plus[j*n_model+point])-minus[j*n_model+point])/scale[j];
        std::vector<float> dp = forwardModeling(pmt,mdl,inv,plus);
        std::vector<float> dm = forwardModeling(pmt,mdl,inv,minus);
        for(int i = 0; i < int(dp.size()); i++){
            dp[i] = float((double(dp[i])-dm[i])/realStep);
        }
        saveBin("born_" + parameters[j] + "_operator", bornData, pmt.nt_data, pmt.Nrec);
        saveBin("born_" + parameters[j] + "_fd", dp, pmt.nt_data, pmt.Nrec);
        saveMetrics(csv, "born", j, j, H, bornData, dp);
    }

    std::vector<float> d0 = forwardModeling(pmt,mdl,inv,model);
    std::string observedFile = saveObservedData(pmt,d0);
    std::vector<float> g0 = calculateGradient(inv,model);
    csv << "gradient_zero,-1,-1,0,0," << calculateNorm(g0) << ",0\n";

    std::vector<float> columns[3];
    for(int j = 0; j < 3; j++){
        std::vector<float> direction(3*n_model,0.0f);
        direction[j*n_model+point] = 1.0f;
        std::vector<float> hx = hessianVectorProduct(pmt,mdl,mgt,inv,model,direction);
        columns[j] = hx;
        csv << "curvature," << j << "," << j << ",0,0," << hx[j*n_model+point] << ",0\n";

        double h = H;
        for(int s = 0; s < 4; s++){
            std::vector<float> plus = model;
            std::vector<float> minus = model;
            plus[j*n_model+point] = float(double(model[j*n_model+point]) + h*scale[j]);
            minus[j*n_model+point] = float(double(model[j*n_model+point]) - h*scale[j]);
            double realStep = (double(plus[j*n_model+point])-minus[j*n_model+point])/scale[j];
            if(realStep == 0.0){
                throw std::runtime_error("Hessian step too small.");
            }
            std::vector<float> gp = calculateGradient(inv,plus);
            std::vector<float> gm = calculateGradient(inv,minus);

            for(int b = 0; b < 3; b++){
                std::vector<float> fd(n_model);
                for(int i = 0; i < n_model; i++){
                    fd[i] = float(scale[b]*(double(gp[b*n_model+i])-gm[b*n_model+i])/realStep);
                }
                std::vector<float> result = getParameter(hx,b);
                std::ostringstream filename;
                filename << "hessiana_" << parameters[j] << "_para_" << parameters[b] << "_h" << std::fixed << std::setprecision(5) << h;
                saveBin(filename.str()+"_operator", result, NZ, NX);
                saveBin(filename.str()+"_fd", fd, NZ, NX);
                saveMetrics(csv, "hessian", j, b, h, result, fd);
            }
            h /= 2.0;
        }
    }

    for(int a = 0; a < 3; a++){
        for(int b = a+1; b < 3; b++){
            double hab = columns[b][a*n_model+point];
            double hba = columns[a][b*n_model+point];
            double denominator = std::max(std::abs(hab),std::abs(hba));
            double error = 0.0;
            if(denominator != 0.0){
                error = std::abs(hab-hba)/denominator;
            }
            csv << "symmetry," << a << "," << b << ",0," << error << "," << hab << "," << hba << "\n";
            std::cout << "info: Symmetry " << parameters[a] << "/" << parameters[b] << " error=" << error << std::endl;
        }
    }

    std::remove(observedFile.c_str());
    inv.freeMemory();
    mgt.freeMemory();
    mdl.freeMemory();
}

int main(){
    mkdir("../outputs", 0755);
    mkdir(folder.c_str(), 0755);
    std::ofstream csv(folder + "metrics.csv");
    if(!csv.is_open()){
        throw std::runtime_error("Could not open metrics.csv.");
    }
    csv << std::setprecision(17);
    csv << "test,input,output,h,relative_error,operator_norm,fd_norm\n";
    validateGradient(csv);
    validateBornAndHessian(csv);
    csv.close();
    std::cout << "\ninfo: Results saved to " << folder << std::endl;
    return 0;
}
