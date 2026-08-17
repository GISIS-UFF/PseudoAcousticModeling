#include "Survey.hpp"
#include "Modeling.cuh"
#include "Migration.cuh"
#include <chrono>

int main(){
    auto ti = std::chrono::system_clock::now();
    Survey pmt;

    Modeling mdl(&pmt);
    mdl.initializeFields();

    Migration mig(&pmt,&mdl);

    if(pmt.migration=="checkpoint"){
        mig.solveReverseTimeMigrationCheckpoint();
    }
    else if(pmt.migration=="onthefly"){
        mig.solveReverseTimeMigrationOntheFly();
    }
    else{
        std::cerr<<"Error: Invalid migration method: "<<pmt.migration<<std::endl;
        return 1;
    }

    mig.freeMemory();
    mdl.freeMemory();

    auto tf = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;

    return 0;
}