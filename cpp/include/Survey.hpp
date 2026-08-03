#pragma once

#include <string>
#include <vector>

class Survey
{
public:
    Survey();
    
    void readParameters();
    void readGeometry();

    // Tipo de processamento
    std::string unit;
    std::string approximation;
    std::string migration;
    std::string ABC;

    // Discretização
    float dx = 0.0f;
    float dz = 0.0f;
    float dt = 0.0f;

    // Dimensões físicas
    float L = 0.0f;
    float D = 0.0f;
    float T = 0.0f;

    // Camada absorvente
    int N_abc = 0;

    // Fonte
    float fcut = 0.0f;
    float tlag = 0.0f;

    // Dimensões discretas
    int nx = 0;
    int nz = 0;
    int nt = 0;

    int nx_abc = 0;
    int nz_abc = 0;

    // Eixos
    std::vector<float> x;
    std::vector<float> z;
    std::vector<float> t;

    // Migração
    float sigma = 0.0f;
    float dvel = 0.0f;
    float ratio = 0.0f;
    float shift = 0.0f;
    float window = 0.0f;
    float v0 = 0.0f;

    bool reciprocity = false;
    bool mirror = false;

    // Snapshots
    int step = 0;
    int last_save = 0;
    bool snap = false;

    // FWI
    bool fwi = false;
    int niter = 0;

    std::vector<float> freqs;

    float vmin = 0.0f;
    float vmax = 0.0f;

    float epsmin = 0.0f;
    float epsmax = 0.0f;

    float deltamin = 0.0f;
    float deltamax = 0.0f;

    float thetamin = 0.0f;
    float thetamax = 0.0f;

    bool multiparameter = false;

    // Modelos sintéticos
    bool layer2 = false;
    bool layer3 = false;
    bool gradientmodel = false;
    bool diffractor = false;
    bool modelfromvp = false;
    bool waterlayer = false;

    int idx_water = 0;

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

    int Nrec = 0;
    int Nshot = 0;

};
