#include "Survey.hpp"
#include "Modeling.cuh"
#include "Migration.cuh"
#include "Inversion.cuh"
#include <chrono>

int main(int argc, char* argv[])
{
auto ti = std::chrono::system_clock::now();
    Survey pmt;
    Modeling mdl(&pmt);
    Migration mgt(&pmt, &mdl);
    Inversion inv(&pmt, &mdl, &mgt);

    mdl.initializeFields();
    mgt.initializeMigrationFields();
    cudaFree(mgt.ilum);
    mgt.ilum = nullptr;
    inv.InitializeInversionFields();

    mdl.createWavelet();
    mdl.createCerjanVector();
    
    // if(pmt.multiparameter){
    //     inv.solveFullWaveformInversionMultiparameterHierarchical();
    // }
    // else{
    //     inv.solveFullWaveformInversionMonoparameter();  
    // }

    const int n_model = pmt.nx * pmt.nz;

    const std::string background_files[3] = {"../inputs/models/layer2vp_Nz31_Nx31.bin", "../inputs/models/layer2epsilon_Nz31_Nx31.bin", "../inputs/models/layer2delta_Nz31_Nx31.bin"};
    const std::string diffractor_files[3] = {"../inputs/models/diffractorvp_Nz31_Nx31.bin", "../inputs/models/layer2epsilon_Nz31_Nx31.bin", "../inputs/models/layer2delta_Nz31_Nx31.bin"};

    inv.models.resize(3 * n_model);

    std::vector<float> models_diffractor(3 * n_model);
    std::vector<float> direction(3 * n_model);

    for(int j = 0; j < 3; j++){
        mdl.importBin(background_files[j],inv.models.data() + j * n_model,n_model);
        mdl.importBin(diffractor_files[j],models_diffractor.data() + j * n_model,n_model);
    }

    for(int i = 0; i < n_model; i++){
        const float vp_background = inv.models[i];
        const float vp_diffractor = models_diffractor[i];

        inv.models[i] = 1.0f / (vp_background * vp_background);
        models_diffractor[i] = 1.0f / (vp_diffractor * vp_diffractor);
    }

    inv.convertNormalizedAndPhysical(false, inv.models);
    inv.convertNormalizedAndPhysical(false, models_diffractor);

    for(int k = 0; k < 3 * n_model; k++){
        direction[k] = models_diffractor[k] - inv.models[k];

    }

    inv.validategradient(0.1f);
    inv.validateborn(direction, 0.1f);
    inv.validateHessian(direction, 0.1f);

    inv.freeMemory();
    mgt.freeMemory();
    mdl.freeMemory();

    auto tf = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;

    return 0; 
}