#include <iostream> 
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>
#include <cmath>
#include "Survey.hpp"

using json = nlohmann::json;

const float pi = 3.14159265358979323846f;

Survey::Survey()
{
    readParameters();
    readGeometry();
}

void read_csv(std::string path, std::vector<float>& x,std::vector<float>& z)
{
    std::ifstream file(path);
    if (file.is_open()){
        std::string line;
        getline(file, line);
        while (getline(file, line)){
            std::stringstream ss(line);
            std::string index;
            std::string coordx;
            std::string coordz;
            getline(ss, index, ',');
            getline(ss, coordx, ',');
            getline(ss, coordz, ',');
            x.push_back(std::stof(coordx));
            z.push_back(std::stof(coordz));
        }
    }
}

void Survey::readParameters() 
{
    std::ifstream jsonFile("Parameters.json");
    json parameters = json::parse(jsonFile);

    unit = parameters["unit"].get<std::string>();
    approximation = parameters["approximation"].get<std::string>();
    migration = parameters["migration"].get<std::string>();
    ABC = parameters["ABC"].get<std::string>();
    dx = parameters["dx"].get<float>();
    dz = parameters["dz"].get<float>();
    dt = parameters["dt"].get<float>();
    L = parameters["L"].get<float>();
    D = parameters["D"].get<float>();
    T = parameters["T"].get<float>();
    N_abc = parameters["N_abc"].get<int>();
    fcut = parameters["fcut"].get<float>();
    sigma = parameters["sigma"].get<float>();
    dvel = parameters["dvel"].get<float>();
    ratio = parameters["ratio"].get<float>();
    shift = parameters["shift"].get<float>();
    window = parameters["window"].get<float>();
    v0 = parameters["v0"].get<float>();
    step = parameters["step"].get<int>();
    last_save = parameters["last_save"].get<int>();
    fwi = parameters["fwi"].get<bool>();
    niter = parameters["niter"].get<int>();
    freqs = parameters["freqs"].get<std::vector<float>>();
    vmin = parameters["vmin"].get<float>();
    vmax = parameters["vmax"].get<float>();
    epsmin = parameters["epsmin"].get<float>();
    epsmax = parameters["epsmax"].get<float>();
    deltamin = parameters["deltamin"].get<float>();
    deltamax = parameters["deltamax"].get<float>();
    thetamin = parameters["thetamin"].get<float>();
    thetamax = parameters["thetamax"].get<float>();
    multiparameter = parameters["multiparameter"].get<bool>();
    reciprocity = parameters["reciprocity"].get<bool>();
    mirror = parameters["mirror"].get<bool>();
    layer2 = parameters["layer2"].get<bool>();
    layer3 = parameters["layer3"].get<bool>();
    gradientmodel = parameters["gradientmodel"].get<bool>();
    diffractor = parameters["diffractor"].get<bool>();
    modelfromvp = parameters["modelfromvp"].get<bool>();
    waterlayer = parameters["waterlayer"].get<bool>();
    idx_water = parameters["idx_water"].get<int>();
    snap = parameters["snap"].get<bool>();
    rec_file = parameters["rec_file"].get<std::string>();
    src_file = parameters["src_file"].get<std::string>();
    vpFile = parameters["vpFile"].get<std::string>();
    epsilonFile = parameters["epsilonFile"].get<std::string>();
    deltaFile = parameters["deltaFile"].get<std::string>();
    thetaFile = parameters["thetaFile"].get<std::string>();
    snapshotFolder = parameters["snapshotFolder"].get<std::string>();
    seismogramFolder = parameters["seismogramFolder"].get<std::string>();
    checkpointFolder = parameters["checkpointFolder"].get<std::string>();
    migratedimageFolder = parameters["migratedimageFolder"].get<std::string>();
    modelFolder = parameters["modelFolder"].get<std::string>();
    estimatedmodelsFolder = parameters["estimatedmodelsFolder"].get<std::string>();
    gradientsFolder = parameters["gradientsFolder"].get<std::string>();

    tlag = 2.0f * std::sqrt(pi)/fcut;
    itlag = std::round(tlag/dt);
    nt_data = std::round(T/dt) + 1;
    nx = std::round(L/dx) + 1;
    nz = std::round(D/dz) + 1;
    nt = itlag + nt_data;
    nx_abc = nx + 2*N_abc;
    nz_abc = nz + 2*N_abc;

    x.resize(nx);
    z.resize(nz);
    t.resize(nt);

    for (int i = 0; i < nx; i++) {
        x[i] = i * dx;
    }

    for (int i = 0; i < nz; i++) {
        z[i] = i * dz;
    }

    for (int i = 0; i < nt; i++) {
        t[i] = i * dt;
    }

}

void Survey::readGeometry()
{
    read_csv(rec_file, rec_x, rec_z);
    read_csv(src_file, shot_x, shot_z);

    if (reciprocity) {
        std::swap(shot_x, rec_x);
        std::swap(shot_z, rec_z);
    }

    Nrec = (rec_x.size());
    Nshot = (shot_x.size());

    rx.resize(Nrec);
    rz.resize(Nrec);

    sx.resize(Nshot);
    sz.resize(Nshot);

    for (int irec = 0; irec < Nrec; ++irec)
    {
        rx[irec] = std::round(rec_x[irec] / dx) + N_abc;
        rz[irec] = std::round(rec_z[irec] / dz) + N_abc;
    }

    for (int ishot = 0; ishot < Nshot; ++ishot)
    {
        sx[ishot] = std::round(shot_x[ishot] / dx) + N_abc;
        sz[ishot] = std::round(shot_z[ishot] / dz) + N_abc;
    }

    if (mirror)
    {
        for (int irec = 0; irec < Nrec; ++irec)
        {
            rz[irec] = std::round(rec_z[irec] / dz) + N_abc - idx_water;
        }

        for (int ishot = 0; ishot < Nshot; ++ishot)
        {
            sz[ishot] = std::round(shot_z[ishot] / dz) + N_abc - idx_water;
        }
    }    
    
}

