#include "Survey.hpp"
#include "Modeling.cuh"
#include <chrono>

int main(){
    auto ti = std::chrono::system_clock::now();

    Survey pmt;

    Modeling modeling(&pmt);

    modeling.solveWaveEquation();

    modeling.freeMemory();

    auto tf = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;

    return 0;
}