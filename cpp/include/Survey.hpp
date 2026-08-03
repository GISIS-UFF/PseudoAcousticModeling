#pragma once

#include <string>
#include <vector>

class Survey
{
public:
    Survey();

    // Tipo de processamento
    std::string unit;
    std::string approximation;
    std::string migration;
    std::string ABC;

    // Discretização
    float dx;
    float dz;
    float dt;

    // Dimensões físicas
    float L;
    float D;
    float T;

    // Camada absorvente
    int N_abc;

    // Fonte
    float fcut;
    float tlag;

    // Dimensões discretas
    int nx;
    int nz;
    int nt;

    int nx_abc;
    int nz_abc;

    // Eixos
    std::vector<float> x;
    std::vector<float> z;
    std::vector<float> t;

    // Migração
    float sigma;
    float dvel;
    float ratio;
    float shift;
    float window;
    float v0;

    bool reciprocity = false;
    bool mirror = false;

    // Snapshots
    int step;
    int last_save;
    bool snap = false;

    // FWI
    bool fwi = false;
    int niter;

    std::vector<float> freqs;

    float vmin;
    float vmax;

    float epsmin;
    float epsmax;

    float deltamin;
    float deltamax;

    float thetamin;
    float thetamax;

    bool multiparameter = false;

    // Modelos sintéticos
    bool layer2 = false;
    bool layer3 = false;
    bool gradientmodel = false;
    bool diffractor = false;
    bool modelfromvp = false;
    bool waterlayer = false;

    int idx_water;

    // Arquivos da aquisição
    std::string rec_file;
    std::string src_file;

    // Arquivos dos modelos
    std::string vpFile;
    std::string epsilonFile;
    std::string deltaFile;
    std::string thetaFile;

    // Pastas
    std::string snapshotFolder;
    std::string seismogramFolder;
    std::string checkpointFolder;
    std::string migratedimageFolder;
    std::string modelFolder;
    std::string estimatedmodelsFolder;
    std::string gradientsFolder;

    // Geometria
    std::vector<float> rec_x;
    std::vector<float> rec_z;
    std::vector<float> shot_x;
    std::vector<float> shot_z;

    std::vector<int> rx;
    std::vector<int> rz;
    std::vector<int> sx;
    std::vector<int> sz;

    int Nrec;
    int Nshot;

private:
    void readParameters()
    void readGeometry()
};
