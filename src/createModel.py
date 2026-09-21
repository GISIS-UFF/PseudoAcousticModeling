import numpy as np

class model:
    def __init__(self, parameters):
        self.pmt = parameters

        self.vp = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
            self.delta = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
            if self.pmt.approximation == "TTI":
                self.theta = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)

        self.vp1 = 1500.0 
        self.vp2 = 1500.0
        self.vp3 = 00.0

        self.epsilon1 = 0.1
        self.epsilon2 = 0.1
        self.epsilon3 = 0.0

        self.delta1 = 0.05
        self.delta2 = 0.05
        self.delta3 = 0.0

        self.theta1 = 30.0
        self.theta2 = 30.0
        self.theta3 = 0.0

    def ImportModel(self, filename):
        data = np.fromfile(filename, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        print(f"info: Imported: {filename}")
        return data
    
    def loadModels(self):
        self.vp = self.ImportModel(self.pmt.vpFile)
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon = self.ImportModel(self.pmt.epsilonFile)
            self.delta = self.ImportModel(self.pmt.deltaFile)
        if self.pmt.approximation == "TTI":
            self.theta = self.ImportModel(self.pmt.thetaFile)
            self.theta = np.radians(self.theta)
        
        print(f"info: Models loaded: {self.pmt.nx}x{self.pmt.nz}")

    def create2LayerModel(self,v1,v2,e1,e2,d1,d2,t1,t2):
        self.vp[:self.pmt.nz//2, :] = v1
        self.vp[self.pmt.nz//2:self.pmt.nz, :] = v2
        self.modelFile = f"{self.pmt.modelFolder}layer2vp_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
        self.vp.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon[:self.pmt.nz//2, :] = e1
            self.epsilon[self.pmt.nz//2:self.pmt.nz, :] = e2
            self.modelFile = f"{self.pmt.modelFolder}layer2epsilon_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.epsilon.tofile(self.modelFile)
            print(f"info: Epsilon saved to {self.modelFile}")

            self.delta[:self.pmt.nz//2, :] = d1
            self.delta[self.pmt.nz//2:self.pmt.nz, :] = d2
            self.modelFile = f"{self.pmt.modelFolder}layer2delta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.delta.tofile(self.modelFile)
            print(f"info: Delta saved to {self.modelFile}")
        
        if self.pmt.approximation == "TTI":
            self.theta[:self.pmt.nz//2, :] = t1
            self.theta[self.pmt.nz//2:self.pmt.nz, :] = t2
            self.modelFile = f"{self.pmt.modelFolder}layer2theta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.theta.tofile(self.modelFile)
            print(f"info: Theta saved to {self.modelFile}")
        
    
    def create3LayerModel(self,v1, v2, v3,e1,e2,e3,d1,d2,d3,t1,t2,t3):
        self.vp[:self.pmt.nz//3, :] = v1
        self.vp[self.pmt.nz//3:2*self.pmt.nz//3, :] = v2
        self.vp[2*self.pmt.nz//3:, :] = v3
        self.modelFile = f"{self.pmt.modelFolder}layer3vp_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
        self.vp.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon[:self.pmt.nz//3, :] = e1
            self.epsilon[self.pmt.nz//3:2*self.pmt.nz//3, :] = e2
            self.epsilon[2*self.pmt.nz//3:, :] = e3
            self.modelFile = f"{self.pmt.modelFolder}layer3epsilon_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.epsilon.tofile(self.modelFile)
            print(f"info: Epsilon saved to {self.modelFile}")

            self.delta[:self.pmt.nz//3, :] = d1
            self.delta[self.pmt.nz//3:2*self.pmt.nz//3, :] = d2
            self.delta[2*self.pmt.nz//3:, :] = d3
            self.modelFile = f"{self.pmt.modelFolder}layer3delta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.delta.tofile(self.modelFile)
            print(f"info: Delta saved to {self.modelFile}")
        
        if self.pmt.approximation == "TTI":
            self.theta[:self.pmt.nz//3, :] = t1
            self.theta[self.pmt.nz//3:2*self.pmt.nz//3, :] = t2
            self.theta[2*self.pmt.nz//3:, :] = t3
            self.modelFile = f"{self.pmt.modelFolder}layer3theta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.theta.tofile(self.modelFile)
            print(f"info: Theta saved to {self.modelFile}")
    
    def createDiffractorModel(self,v1,v2,e1,e2,d1,d2,t1,t2):
        self.vp[:, :] = v1
        # self.vp[self.pmt.nz//2,self.pmt.nx//2] = v2
        self.vp[(self.pmt.nz // 2)-5:(self.pmt.nz // 2)+5, (self.pmt.nx // 2)-5:(self.pmt.nx // 2)+5] = v2
        self.modelFile = f"{self.pmt.modelFolder}diffractorvp_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
        self.vp.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon[:, :] = e1
            # self.epsilon[self.pmt.nz // 2, self.pmt.nx // 2] = e2
            self.epsilon[(self.pmt.nz // 2)-5:(self.pmt.nz // 2)+5, (self.pmt.nx // 2)-5:(self.pmt.nx // 2)+5] = e2
            self.modelFile = f"{self.pmt.modelFolder}diffractorepsilon_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.epsilon.tofile(self.modelFile)
            print(f"info: Epsilon saved to {self.modelFile}")

            self.delta[:, :] = d1
            # self.delta[self.pmt.nz // 2, self.pmt.nx // 2] = d2
            self.delta[(self.pmt.nz // 2)-5:(self.pmt.nz // 2)+5, (self.pmt.nx // 2)-5:(self.pmt.nx // 2)+5] = d2
            self.modelFile = f"{self.pmt.modelFolder}diffractordelta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.delta.tofile(self.modelFile)
            print(f"info: Delta saved to {self.modelFile}")
            
        if self.pmt.approximation == "TTI":
            self.theta[:, :] = t1
            self.theta[self.pmt.nz // 2, self.pmt.nx // 2] = t2
            self.modelFile = f"{self.pmt.modelFolder}diffractortheta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.theta.tofile(self.modelFile)
            print(f"info: Theta saved to {self.modelFile}")
    
    def createGradientModel(self,v1,e1,d1,t1):
        self.vp[:, :] = v1
        alpha = 0.7
        for iz in range(self.pmt.nz):
            self.vp[iz,:] = v1 + alpha*self.pmt.z[iz]

        self.modelFile = f"{self.pmt.modelFolder}gradientvp_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
        self.vp.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon[:, :] = e1
            for iz in range(self.pmt.nz):
                self.epsilon[iz,:] = e1 + alpha*self.pmt.z[iz]
            self.modelFile = f"{self.pmt.modelFolder}gradientepsilon_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.epsilon.tofile(self.modelFile)
            print(f"info: Epsilon saved to {self.modelFile}")

            self.delta[:, :] = d1
            for iz in range(self.pmt.nz):
                self.delta[iz,:] = d1 + alpha*self.pmt.z[iz]
            self.modelFile = f"{self.pmt.modelFolder}gradientdelta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.delta.tofile(self.modelFile)
            print(f"info: Delta saved to {self.modelFile}")
            
        if self.pmt.approximation == "TTI":
            self.theta[:, :] = t1
            for iz in range(self.pmt.nz):
                self.theta[iz,:] = t1 + alpha*self.pmt.z[iz]
            self.modelFile = f"{self.pmt.modelFolder}gradienttheta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.theta.tofile(self.modelFile)
            print(f"info: Theta saved to {self.modelFile}")

    def createModelFromVp(self):  
        if self.pmt.vpFile == None:
            raise ValueError("ERROR: Import or create a velocity model first.")

        self.vp = self.ImportModel(self.pmt.vpFile)

        self.idx_water = np.where(self.vp <= 1500)
        # create density model with Gardner's equation
        self.rho = np.zeros_like(self.vp,dtype=np.float32)
        a, b = 0.23, 0.25
        self.rho = a * np.power(self.vp/0.3048,b)*1000 # Gardner relation - Rosa (2010) apud Gardner et al. (1974) pag. 496 rho = a * v^b
        self.rho[self.idx_water] = 1000.0 # water density

        # create epsilon model epsilon = 0.25 rho - 0.3 - Petrov et al. (2021) 
        self.epsilon = np.zeros_like(self.vp,dtype=np.float32)
        self.epsilon = (0.25 * self.rho/1000) - 0.3 # rho in g/cm3
        self.epsilon[self.idx_water] = 0.0 # water epsilon
        self.epsilon.tofile(self.pmt.vpFile.replace(".bin","_epsilon.bin"))	
        print(f"info: Epsilon model saved to {self.pmt.vpFile.replace('.bin','_epsilon.bin')}")

        # create delta model delta = 0.125 rho - 0.1 - Petrov et al. (2021)
        self.delta = np.zeros_like(self.vp,dtype=np.float32)
        self.delta = (0.125 * self.rho/1000) - 0.1 # rho in g/cm3
        self.delta[self.idx_water] = 0.0 # water delta
        self.delta.tofile(self.pmt.vpFile.replace(".bin","_delta.bin"))
        print(f"info: Delta model saved to {self.pmt.vpFile.replace('.bin','_delta.bin')}")
    
    def createWaterLayer(self):
        self.loadModels()
        vp_exp = np.zeros((self.pmt.nz + self.pmt.idx_water, self.pmt.nx), dtype=self.vp.dtype)
        vp_exp[:self.pmt.idx_water, :] = 1500.0
        vp_exp[self.pmt.idx_water:self.pmt.nz + self.pmt.idx_water, :] = self.vp
        self.modelFile = f"{self.pmt.modelFolder}ExpandWatervp_Nz{self.pmt.nz + self.pmt.idx_water}_Nx{self.pmt.nx}.bin"
        vp_exp.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")
    
    def buildModel(self):
        if self.pmt.layer2 == True:
            self.create2LayerModel(self.vp1,self.vp2,self.epsilon1,self.epsilon2,self.delta1,self.delta2,self.theta1,self.theta2)
        elif self.pmt.layer3 == True:
            self.create3LayerModel(self.vp1,self.vp2,self.vp3,self.epsilon1,self.epsilon2,self.epsilon3,self.delta1,self.delta2,self.delta3,self.theta1,self.theta2,self.theta3)
        elif self.pmt.diffractor == True:
            self.createDiffractorModel(self.vp1,self.vp2,self.epsilon1,self.epsilon2,self.delta1,self.delta2,self.theta1,self.theta2)
        elif self.pmt.gradientmodel == True:
            self.createGradientModel(self.vp1,self.epsilon1,self.delta1,self.theta1)
        elif self.pmt.modelfromvp == True:
            self.createModelFromVp()
        elif self.pmt.waterlayer == True:
            self.createWaterLayer()
        else:
            raise ValueError(f"ERROR: Unknwon synthetic model.")
        
if __name__ == "__main__":
    from survey import parameters

    pmt = parameters("../inputs/Parameters.json")
    
    mdl = model(pmt)
    mdl.buildModel()

        

        