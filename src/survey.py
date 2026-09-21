import pandas as pd
import json
import numpy as np

class parameters: 
    def __init__(self, parameters_path):
        self.parameters_path = parameters_path
        self.readParameters()
        self.readAcquisitionGeometry()

    def readParameters(self):
        with open(self.parameters_path) as f:
            self.parameters = json.load(f)

        # Approximation type
        self.approximation = self.parameters["approximation"]

        
        # Discretization self.parameters
        self.dx   = self.parameters["dx"]
        self.dz   = self.parameters["dz"]
        self.dt   = self.parameters["dt"]
        
        # Model size
        self.L    = self.parameters["L"]
        self.D    = self.parameters["D"]
        self.T    = self.parameters["T"]

        # Number of point for absorbing boundary condition
        self.N_abc = self.parameters["N_abc"]

        # Max frequency
        self.fcut = self.parameters["fcut"]

        # Source delay
        self.tlag = 2.0*np.sqrt(np.pi)/self.fcut

        # Number of points in each direction
        self.nx = np.round(self.L/self.dx).astype(np.int32)+1
        self.nz = np.round(self.D/self.dz).astype(np.int32)+1
        self.itlag = np.round(self.tlag / self.dt).astype(np.int32)
        self.nt_data = np.round(self.T / self.dt).astype(np.int32) + 1
        self.nt = self.itlag + self.nt_data

        self.nx_abc = self.nx + 2*self.N_abc
        self.nz_abc = self.nz + 2*self.N_abc

        # Define arrays for space and time
        self.x = np.arange(self.nx) * self.dx
        self.z = np.arange(self.nz) * self.dz
        self.t = np.arange(self.nt) * self.dt

        # Folders
        self.seismogramFolder = self.parameters["seismogramFolder"]
        self.imageFolder = self.parameters["imageFolder"]
        self.snapshotFolder = self.parameters["snapshotFolder"]
        self.modelFolder = self.parameters["modelFolder"]
        self.checkpointFolder = self.parameters["checkpointFolder"]
        self.estimatedmodelsFolder = self.parameters["estimatedmodelsFolder"]
        self.gradientsFolder = self.parameters["gradientsFolder"]

        # Source and receiver files
        self.rec_file = self.parameters["rec_file"]
        self.src_file = self.parameters["src_file"]

        # Velocity model file
        self.vpFile = self.parameters["vpFile"]
        self.thetaFile = self.parameters["thetaFile"]

        # Snapshot flag
        self.snap = self.parameters["snap"]
        self.step = self.parameters["step"]
        self.last_save = self.parameters["last_save"]

        # Anisotropy parameters files
        self.epsilonFile = self.parameters["epsilonFile"]  
        self.deltaFile   = self.parameters["deltaFile"]  

        # Synthetic models 
        self.layer2 =  self.parameters['layer2']
        self.layer3 =  self.parameters['layer3']
        self.gradientmodel =  self.parameters['gradientmodel']
        self.diffractor =  self.parameters['diffractor']
        self.modelfromvp =  self.parameters['modelfromvp']
        self.waterlayer = self.parameters['waterlayer']
        
        #migration parameters
        self.mirror = self.parameters['mirror']
        self.reciprocity = self.parameters['reciprocity']
        self.idx_water = self.parameters['idx_water']

    def readAcquisitionGeometry(self):        
        # Read receiver and source coordinates from CSV files
        receiverTable = pd.read_csv(self.rec_file)
        print(f"info: Imported: {self.rec_file}")     
        sourceTable = pd.read_csv(self.src_file)
        print(f"info: Imported: {self.src_file}")

        # Read receiver and source coordinates
        self.rec_x = receiverTable['coordx'].to_numpy()
        self.rec_z = receiverTable['coordz'].to_numpy()
        self.shot_x = sourceTable['coordx'].to_numpy()
        self.shot_z = sourceTable['coordz'].to_numpy()

        if self.reciprocity == True:
            self.shot_x, self.rec_x = self.rec_x.copy(), self.shot_x.copy()
            self.shot_z, self.rec_z = self.rec_z.copy(), self.shot_z.copy()

        self.rx = np.round(self.rec_x / self.dx).astype(np.int32) + self.N_abc 
        self.rz = np.round(self.rec_z / self.dz).astype(np.int32) + self.N_abc 
        self.sx = np.round(self.shot_x / self.dx).astype(np.int32) + self.N_abc
        self.sz = np.round(self.shot_z / self.dz).astype(np.int32) + self.N_abc 
            
        if self.mirror == True:
            self.rz = np.round(self.rec_z/self.dz).astype(np.int32) + self.N_abc - self.idx_water
            self.sz = np.round(self.shot_z/self.dz).astype(np.int32) + self.N_abc - self.idx_water

        self.Nrec = len(self.rec_x)
        self.Nshot = len(self.shot_x) 



    
  