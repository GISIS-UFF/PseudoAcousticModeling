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
    inv.InitializeInversionFields();
    if(pmt.multiparameter){
        inv.solveFullWaveformInversionMultiparameterHierarchical();
    }
    else{
        inv.solveFullWaveformInversionMonoparameter();  
    }

    inv.freeMemory();
    mgt.freeMemory();
    mdl.freeMemory();

    auto tf = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;

    return 0; 
}