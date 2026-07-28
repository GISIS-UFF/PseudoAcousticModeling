import numpy as np
import time
import cupy as cp
from utils import smooth_model
from utils import ricker

class fwi:
    def __init__(self,parameters,wavefield,migration):
        self.pmt = parameters
        self.wf = wavefield
        self.mig = migration

    def objective_function(self, vp, epsilon, delta, theta, save_residual):
        X = 0.0
        self.vp = 1.0 / np.sqrt(vp)   
        self.wf.source = cp.asarray(self.wf.source, dtype=cp.float32)
        self.wf.vp_exp = self.wf.ExpandModel(self.vp)
        self.wf.vp_exp = cp.asarray(self.wf.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
            self.wf.A = cp.asarray(self.wf.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(delta)
            self.wf.epsilon_exp  = cp.asarray(self.wf.epsilon_exp, dtype=cp.float32)
            self.wf.delta_exp  = cp.asarray(self.wf.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(theta)
                self.wf.theta_exp  = cp.asarray(self.wf.theta_exp, dtype=cp.float32)
        
        self.pmt.rx = cp.asarray(self.pmt.rx)
        self.pmt.rz = cp.asarray(self.pmt.rz)
        for shot in range(self.pmt.Nshot):
            dobs = self.loadObsSeismogram(shot)
            self.wf.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]            
            for k in range(self.pmt.nt): 
                self.wf.forward_stepGPU(k)
                # Register seismogram and snapshot
                self.wf.store_seismogram(k,self.pmt.rz,self.pmt.rx)      
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.seismogram = cp.asnumpy(self.wf.seismogram_gpu)
            residual = dobs - self.seismogram
            if save_residual==True:
                self.save_residual(shot,residual) 
            X += 0.5 * np.sum(residual * residual)
        return X

    def calculate_gradient(self, vp, epsilon, delta, theta):
        self.mig.vp = 1.0 / np.sqrt(vp)
        self.mig.ilum.fill(0)
        self.mig.migrated_image.fill(0)
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.mig.epsilon = epsilon
            self.mig.delta = delta
            if self.pmt.multiparameter == True:
                self.mig.epsilon_grad.fill(0)
                self.mig.delta_grad.fill(0)
        if self.pmt.approximation == "TTI":
            self.mig.theta = theta
            if self.pmt.multiparameter == True:
                self.mig.theta_grad.fill(0)
        self.mig.SolveBackwardWaveEquation()
        water_mask = np.abs(self.wf.vp - 1500) < 1e-3
        g_vp = self.loadGradientVp()
        g_vp[water_mask] = 0.0
        g_eps = None
        g_delta = None
        g_theta = None
        if self.pmt.multiparameter == True:
            if self.pmt.approximation in ["VTI", "TTI"]:
                g_eps = self.loadGradientEps()
                g_delta = self.loadGradientDelta()
                g_eps[water_mask] = 0.0
                g_delta[water_mask] = 0.0
            if self.pmt.approximation == "TTI":
                g_theta = self.loadGradientTheta()
                g_theta[water_mask] = 0.0

        return g_vp, g_eps, g_delta, g_theta
    
    def two_loop_recursion(self,g,s_store,y_store):
        q = g.copy()
        alpha = np.zeros(len(s_store))
        beta = np.zeros(len(s_store))

        for i in reversed(range(len(s_store))):
            s = s_store[i]
            y = y_store[i]
            sy = np.sum(s * y)
            rho = 1.0 / sy
            alpha[i] = rho * np.sum(s * q)
            q = q - alpha[i] * y

        if len(s_store) > 0:
            s_last = s_store[-1]
            y_last = y_store[-1]

            sy = np.sum(s_last * y_last)
            yy = np.sum(y_last * y_last)

            gamma = sy / yy

        else:
            gamma = 1.0
 
        r = gamma * q

        for i in range(len(s_store)):
            s = s_store[i]
            y = y_store[i]
            sy = np.sum(s * y)
            rho = 1.0 / sy
            beta[i] = rho * np.sum(y * r)
            r = r + s * (alpha[i] - beta[i])

        return r
    
    def step_length(self, parametro, vp, epsilon, delta, theta,p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr):
        c1 = 1e-4

        if parametro == "vp":
            m_min = 1.0 / (self.pmt.vmax * self.pmt.vmax)
            m_max = 1.0 / (self.pmt.vmin * self.pmt.vmin)
            gTp0 = np.sum(g_vp * p_vp)

        elif parametro == "epsilon":
            m_min = self.pmt.epsmin
            m_max = self.pmt.epsmax
            gTp0 = np.sum(g_eps * p_eps)

        elif parametro == "delta":
            m_min = self.pmt.deltamin
            m_max = self.pmt.deltamax
            gTp0 = np.sum(g_delta * p_delta)

        elif parametro == "theta":
            m_min = self.pmt.thetamin
            m_max = self.pmt.thetamax
            gTp0 = np.sum(g_theta * p_theta)

        alpha = 0.01 * (m_max - m_min)
        X_check = None
        for _ in range(10):
            vp_new = vp
            epsilon_new = epsilon
            delta_new = delta
            theta_new = theta
            if parametro == "vp":
                vp_new = vp + alpha * p_vp
                vp_new = np.clip(vp_new, m_min, m_max)

            elif parametro == "epsilon":
                epsilon_new = epsilon + alpha * p_eps
                epsilon_new = np.clip(epsilon_new, m_min, m_max)

            elif parametro == "delta":
                delta_new = delta + alpha * p_delta
                delta_new = np.clip(delta_new, m_min, m_max)

            elif parametro == "theta":
                theta_new = theta + alpha * p_theta
                theta_new = np.clip(theta_new, m_min, m_max)

            X_new = self.objective_function(vp_new, epsilon_new, delta_new, theta_new, save_residual=False)

            armijo = X_new <= X + c1 * alpha * gTp0

            print("parametro =", parametro)
            print("alpha =", alpha)
            print("X =", X)
            print("X_new =", X_new)
            print("gTp0 =", gTp0)
            print("Armijo =", armijo)

            if armijo:
                return alpha

            if X_check is not None and X_new >= X_check:
                print("info: Armijo condition not reducing objective function. Stopping line search.")
                break
            
            X_check = X_new

            alpha *= 0.5

        return alpha

    def step_length_multiparameter(self, vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, beta_vp, beta_eps, beta_delta, beta_theta, X, itr):
        c1 = 1e-4
        alpha = 1.0

        vp_min = 1.0 / (self.pmt.vmax * self.pmt.vmax)
        vp_max = 1.0 / (self.pmt.vmin * self.pmt.vmin)

        gTp0 = np.sum(beta_vp * g_vp * p_vp)
        gTp0 += np.sum(beta_eps * g_eps * p_eps)
        gTp0 += np.sum(beta_delta * g_delta * p_delta)

        if self.pmt.approximation == "TTI":
            gTp0 += np.sum(beta_theta * g_theta * p_theta)

        X_check = None
        for _ in range(10):
            vp_new = vp + alpha * beta_vp * p_vp
            vp_new = np.clip(vp_new, vp_min, vp_max)

            epsilon_new = epsilon + alpha * beta_eps * p_eps
            delta_new = delta + alpha * beta_delta * p_delta

            epsilon_new = np.clip(epsilon_new, self.pmt.epsmin, self.pmt.epsmax)
            delta_new = np.clip(delta_new, self.pmt.deltamin, self.pmt.deltamax)

            theta_new = theta

            if self.pmt.approximation == "TTI":
                theta_new = theta + alpha * beta_theta * p_theta
                theta_new = np.clip(theta_new, self.pmt.thetamin, self.pmt.thetamax)

            X_new = self.objective_function(vp_new, epsilon_new, delta_new, theta_new, save_residual=False)

            armijo = X_new <= X + c1 * alpha * gTp0

            print("alpha =", alpha)
            print("X =", X)
            print("X_new =", X_new)
            print("gTp0 =", gTp0)
            print("Armijo =", armijo)

            if armijo:
                return alpha
            
            if X_check is not None and X_new >= X_check:
                print("info: Armijo condition not reducing objective function. Stopping line search.")
                break
            
            X_check = X_new

            alpha *= 0.5

        return alpha

    def loadGradientVp(self):
        gradientFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad

    def loadGradientEps(self):
        gradientFile = f"{self.pmt.gradientsFolder}epsilon_gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad

    def loadGradientDelta(self):
        gradientFile = f"{self.pmt.gradientsFolder}delta_gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad

    def loadGradientTheta(self):
        gradientFile = f"{self.pmt.gradientsFolder}theta_gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad
    
    def loadObsSeismogram(self,shot):
        seismogramFile = f"{self.pmt.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}_fcut{self.pmt.fcut}.bin"
        seismogram = np.fromfile(seismogramFile, dtype=np.float32).reshape(self.pmt.nt,self.pmt.Nrec) 
        return seismogram

    def save_residual(self,shot,residual):        
        self.seismogramFile = f"{self.pmt.seismogramFolder}residual_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        residual.tofile(self.seismogramFile)
        print(f"info: Residuo saved to {self.seismogramFile}")

    def solveFullWaveformInversionMonoparameter(self):
        start_time = time.time()
        print("info: Solving Full Waveform Inversion")
        
        # Modelo inicial de vp
        mask_vp = np.abs(self.wf.vp - np.min(self.wf.vp)) < 1e-3 
        self.vp0 = smooth_model(self.wf.vp,self.pmt.sigma,mask_vp)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_vp_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.vp0.astype(np.float32).tofile(smooth_model_file)

        # Vagarosidade ao quadrado
        vp = 1.0 / (self.vp0 * self.vp0)
        epsilon = None
        delta = None
        theta = None

        if self.pmt.approximation in ["VTI","TTI"]:
            # Modelo inicial de epsilon
            mask_eps = np.abs(self.wf.epsilon - np.min(self.wf.epsilon)) < 1e-3 
            self.eps0 = smooth_model(self.wf.epsilon,self.pmt.sigma,mask_eps, parameter = True)
            smooth_model_file = (f"{self.pmt.modelFolder}fwi_epsilon_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
            self.eps0.astype(np.float32).tofile(smooth_model_file)
            
            # Modelo inicial de delta
            mask_delta = np.abs(self.wf.delta - np.min(self.wf.delta)) < 1e-3 
            self.delta0 = smooth_model(self.wf.delta,self.pmt.sigma,mask_delta, parameter = True)
            smooth_model_file = (f"{self.pmt.modelFolder}fwi_delta_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
            self.delta0.astype(np.float32).tofile(smooth_model_file)
            
            epsilon = self.eps0
            delta = self.delta0

        if self.pmt.approximation == "TTI":
            # Modelo inicial de theta
            mask_theta = np.abs(self.wf.theta - np.min(self.wf.theta)) < 1e-3 
            self.theta0 = smooth_model(self.wf.theta,self.pmt.sigma,mask_theta, parameter = True)
            smooth_model_file = (f"{self.pmt.modelFolder}fwi_theta_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
            self.theta0.astype(np.float32).tofile(smooth_model_file)

            theta = self.theta0

        self.history = []

        for fmax in self.pmt.freqs:
            print(f"\033[31minfo: FWI frequency {fmax} of {self.pmt.freqs}\033[0m")

            self.pmt.fcut = fmax
            self.pmt.tlag = 2.0*np.sqrt(np.pi)/self.pmt.fcut
            self.pmt.nt = int((self.pmt.T + self.pmt.tlag)/self.pmt.dt)+1
            self.pmt.t = np.arange(self.pmt.nt) * self.pmt.dt
            self.wf.initializeWavefields()
            self.wf.createSourceWavelet()

            s_vp_store = []
            y_vp_store = []

            # Gradiente e função objetivo no modelo atual
            X = self.objective_function(vp, epsilon, delta, theta, save_residual = True)
            g_vp, g_eps, g_delta, g_theta = self.calculate_gradient(vp, epsilon, delta, theta)
            
            X0 = X

            self.history.append([X/X0, fmax])

            for itr in range(self.pmt.niter):
                print(f"\033[31minfo: FWI iteration {itr + 1}/{self.pmt.niter} for frequency {fmax}\033[0m")

                # Salvar gradiente da iteração atual
                gradient_file = (f"{self.pmt.gradientsFolder}vp_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                (g_vp).astype(np.float32).tofile(gradient_file)
                print(f"info: Vp gradient saved to {gradient_file}")

                # Direção de busca: LBFGS
                p_vp = -self.two_loop_recursion(g_vp,s_vp_store,y_vp_store)
                p_vp = p_vp/np.max(np.abs(p_vp))

                p_eps = None
                p_delta = None
                p_theta = None

                # Line search
                alpha_vp = self.step_length("vp", vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr)

                # Atualização do modelo
                vp_min = 1.0/(self.pmt.vmax*self.pmt.vmax)
                vp_max = 1.0/(self.pmt.vmin*self.pmt.vmin)

                vp_new = vp + alpha_vp * p_vp
                vp_new = np.clip(vp_new, vp_min, vp_max)

                epsilon_new = epsilon
                delta_new = delta
                theta_new = theta

                X_new = self.objective_function(vp_new, epsilon_new, delta_new, theta_new, save_residual = True)
                g_vp_new, g_eps_new, g_delta_new, g_theta_new = self.calculate_gradient(vp_new, epsilon_new, delta_new, theta_new)

                self.history.append([X_new/X0, fmax])

                s_vp = (vp_new - vp)
                y_vp = (g_vp_new - g_vp)
                sy_vp = np.sum(s_vp * y_vp)
                if sy_vp > 0:
                    s_vp_store.append(s_vp)
                    y_vp_store.append(y_vp)

                if len(s_vp_store) > 8:
                    s_vp_store.pop(0)
                    y_vp_store.pop(0)

                vp = vp_new.copy()
                g_vp = g_vp_new.copy()
                X = X_new

                m_it = 1.0 / np.sqrt(vp)
                model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_vp_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                m_it.astype(np.float32).tofile(model_file)
                print(f"info: Model of {itr+1} iteration saved to {model_file}")
                       
        history = np.array(self.history, dtype=np.float32)
        history_file = (f"../outputs/history.txt")
        np.savetxt(history_file,history)

        print(f"info: FWI history saved to {history_file}")

        end_time = time.time()
        print(f"\ninfo: FWI finished in {end_time - start_time:.2f} s")
    
    def solveFullWaveformInversionMultiparameterSimultaneos(self):
        start_time = time.time()
        print("info: Solving Full Waveform Inversion - Simultaneous update")
        
        # Modelo inicial de vp
        mask_vp = np.abs(self.wf.vp - np.min(self.wf.vp)) < 1e-3 
        self.vp0 = self.wf.vp#smooth_model(self.wf.vp,self.pmt.sigma,mask_vp)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_vp_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.vp0.astype(np.float32).tofile(smooth_model_file)

        # Vagarosidade ao quadrado
        vp = 1.0 / (self.vp0 * self.vp0)
        theta = None

        # Escalas das direções
        vp_min = 1.0 / (self.pmt.vmax * self.pmt.vmax)
        vp_max = 1.0 / (self.pmt.vmin * self.pmt.vmin)

        self.scale_vp = 0.01 * (vp_max - vp_min)

        # Modelo inicial de epsilon
        mask_eps = self.wf.epsilon#np.abs(self.wf.epsilon - np.min(self.wf.epsilon)) < 1e-3 
        self.eps0 = smooth_model(self.wf.epsilon,self.pmt.sigma,mask_eps, parameter = True)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_epsilon_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.eps0.astype(np.float32).tofile(smooth_model_file)
        
        # Modelo inicial de delta
        mask_delta = self.wf.delta#np.abs(self.wf.delta - np.min(self.wf.delta)) < 1e-3 
        self.delta0 = smooth_model(self.wf.delta,self.pmt.sigma,mask_delta, parameter = True)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_delta_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.delta0.astype(np.float32).tofile(smooth_model_file)
        
        epsilon = self.eps0
        delta = self.delta0

        # Escalas das direções
        self.scale_eps = 0.01 * (self.pmt.epsmax - self.pmt.epsmin)
        self.scale_delta = 0.01 * (self.pmt.deltamax - self.pmt.deltamin)
        
        self.scale_theta = 0.0
        if self.pmt.approximation == "TTI":
            # Modelo inicial de theta
            mask_theta = np.abs(self.wf.theta - np.min(self.wf.theta)) < 1e-3 
            self.theta0 = smooth_model(self.wf.theta,self.pmt.sigma,mask_theta, parameter = True)
            smooth_model_file = (f"{self.pmt.modelFolder}fwi_theta_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
            self.theta0.astype(np.float32).tofile(smooth_model_file)

            theta = self.theta0

            # Escalas das direções
            if self.pmt.multiparameter == True:
                self.scale_theta = 0.01 * (self.pmt.thetamax - self.pmt.thetamin)

        self.history = []

        for fmax in self.pmt.freqs:
            print(f"\033[31minfo: FWI frequency {fmax} of {self.pmt.freqs}\033[0m")

            self.pmt.fcut = fmax
            self.pmt.tlag = 2.0*np.sqrt(np.pi)/self.pmt.fcut
            self.pmt.nt = int((self.pmt.T + self.pmt.tlag)/self.pmt.dt)+1
            self.pmt.t = np.arange(self.pmt.nt) * self.pmt.dt
            self.wf.initializeWavefields()
            self.wf.createSourceWavelet()
            self.mig.initializeMigrationfields()

            s_vp_store = []
            y_vp_store = []

            s_eps_store = []
            y_eps_store = []
            s_delta_store = []
            y_delta_store = []

            if self.pmt.approximation == "TTI":
                s_theta_store = []
                y_theta_store = []

            # Gradiente e função objetivo no modelo atual
            X = self.objective_function(vp, epsilon, delta, theta, save_residual=True)
            g_vp, g_eps, g_delta, g_theta = self.calculate_gradient(vp, epsilon, delta, theta)
            
            X0 = X
            self.history.append([X/X0, fmax])

            for itr in range(self.pmt.niter):
                print(f"\033[31minfo: FWI iteration {itr + 1}/{self.pmt.niter} for frequency {fmax}\033[0m")

                # Salvar gradiente da iteração atual
                gradient_file = (f"{self.pmt.gradientsFolder}vp_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                (g_vp).astype(np.float32).tofile(gradient_file)
                print(f"info: Vp gradient saved to {gradient_file}")

                gradient_file = (f"{self.pmt.gradientsFolder}epsilon_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                (g_eps).astype(np.float32).tofile(gradient_file)
                print(f"info: Epsilon gradient saved to {gradient_file}")

                gradient_file = (f"{self.pmt.gradientsFolder}delta_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                (g_delta).astype(np.float32).tofile(gradient_file)
                print(f"info: Delta gradient saved to {gradient_file}")

                if self.pmt.approximation == "TTI":
                    gradient_file = (f"{self.pmt.gradientsFolder}theta_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                    (g_theta).astype(np.float32).tofile(gradient_file)
                    print(f"info: Theta gradient saved to {gradient_file}")

                # Direção de busca: LBFGS
                p_vp = -self.two_loop_recursion(g_vp,s_vp_store,y_vp_store)
                p_vp = p_vp / np.max(np.abs(p_vp))

                p_eps = -self.two_loop_recursion(g_eps,s_eps_store,y_eps_store)
                p_eps = p_eps / np.max(np.abs(p_eps)) 

                p_delta = -self.two_loop_recursion(g_delta,s_delta_store,y_delta_store)
                p_delta = p_delta / np.max(np.abs(p_delta)) 

                p_theta = None

                if self.pmt.approximation == "TTI":
                    p_theta = -self.two_loop_recursion(g_theta,s_theta_store,y_theta_store)
                    p_theta = p_theta / np.max(np.abs(p_theta)) 

                # Line search monoparameter
                beta_vp = self.step_length("vp", vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr)
                beta_eps = self.step_length("epsilon", vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr)
                beta_delta = self.step_length("delta", vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr)
                beta_theta = 0.0
                if self.pmt.approximation == "TTI":   
                    beta_theta = self.step_length("theta", vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr)

                # Line search multiparameter
                alpha = self.step_length_multiparameter(vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, beta_vp, beta_eps, beta_delta, beta_theta, X, itr)    
        
                # Atualização do modelo
                vp_new = vp + alpha * beta_vp * p_vp
                vp_new = np.clip(vp_new, vp_min, vp_max)

                epsilon_new = epsilon + alpha * beta_eps * p_eps
                delta_new = delta + alpha * beta_delta * p_delta

                epsilon_new = np.clip(epsilon_new, self.pmt.epsmin, self.pmt.epsmax)
                delta_new = np.clip(delta_new, self.pmt.deltamin, self.pmt.deltamax)

                theta_new = theta
                if self.pmt.approximation == "TTI":
                    theta_new = theta + alpha * beta_theta * p_theta
                    theta_new = np.clip(theta_new, self.pmt.thetamin, self.pmt.thetamax)

                X_new = self.objective_function(vp_new, epsilon_new, delta_new, theta_new,save_residual=True)
                g_vp_new, g_eps_new, g_delta_new, g_theta_new = self.calculate_gradient(vp_new, epsilon_new, delta_new, theta_new)

                self.history.append([X_new/X0, fmax])

                s_vp = (vp_new - vp)
                y_vp = (g_vp_new - g_vp)
                sy_vp = np.sum(s_vp * y_vp)

                s_eps = (epsilon_new - epsilon)
                y_eps = (g_eps_new - g_eps)
                sy_eps = np.sum(s_eps * y_eps)

                s_delta = (delta_new - delta)
                y_delta = (g_delta_new - g_delta)
                sy_delta = np.sum(s_delta * y_delta)

                if sy_vp > 0:
                    s_vp_store.append(s_vp)
                    y_vp_store.append(y_vp)

                if sy_eps > 0:
                    s_eps_store.append(s_eps)
                    y_eps_store.append(y_eps)

                if sy_delta > 0:
                    s_delta_store.append(s_delta)
                    y_delta_store.append(y_delta)

                if self.pmt.approximation == "TTI":
                    s_theta = (theta_new - theta)
                    y_theta = (g_theta_new - g_theta)
                    sy_theta = np.sum(s_theta * y_theta)

                    if sy_theta > 0:
                        s_theta_store.append(s_theta)
                        y_theta_store.append(y_theta)

                if len(s_vp_store) > 8:
                    s_vp_store.pop(0)
                    y_vp_store.pop(0)

                if len(s_eps_store) > 8:
                    s_eps_store.pop(0)
                    y_eps_store.pop(0)

                if len(s_delta_store) > 8:
                    s_delta_store.pop(0)
                    y_delta_store.pop(0)

                if self.pmt.approximation == "TTI":
                    if len(s_theta_store) > 8:
                        s_theta_store.pop(0)
                        y_theta_store.pop(0)

                vp = vp_new.copy()
                g_vp = g_vp_new.copy()
                epsilon = epsilon_new.copy()
                delta = delta_new.copy()
                g_eps = g_eps_new.copy()
                g_delta = g_delta_new.copy()
                X = X_new

                if self.pmt.approximation == "TTI":
                    theta = theta_new.copy()
                    g_theta = g_theta_new.copy()

                m_it = 1.0 / np.sqrt(vp)
                model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_vp_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                m_it.astype(np.float32).tofile(model_file)
                print(f"info: Model of {itr+1} iteration saved to {model_file}")

                model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_epsilon_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                epsilon.astype(np.float32).tofile(model_file)
                print(f"info: Epsilon model of {itr+1} iteration saved to {model_file}")

                model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_delta_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                delta.astype(np.float32).tofile(model_file)
                print(f"info: Delta model of {itr+1} iteration saved to {model_file}")

                if self.pmt.approximation == "TTI":
                    model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_theta_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                    theta.astype(np.float32).tofile(model_file)
                    print(f"info: Theta model of {itr+1} iteration saved to {model_file}")
            
        history = np.array(self.history, dtype=np.float32)
        history_file = (f"../outputs/history.txt")
        np.savetxt(history_file,history)

        print(f"info: FWI history saved to {history_file}")

        end_time = time.time()
        print(f"\ninfo: FWI finished in {end_time - start_time:.2f} s")


    def solveFullWaveformInversionMultiparameterHierarchical(self):
        start_time = time.time()
        print("info: Solving Full Waveform Inversion - Hierarchical cumulative update")
        
        # Modelo inicial de vp
        mask_vp = np.abs(self.wf.vp - np.min(self.wf.vp)) < 1e-3 
        self.vp0 = smooth_model(self.wf.vp,self.pmt.sigma,mask_vp)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_vp_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.vp0.astype(np.float32).tofile(smooth_model_file)

        # Vagarosidade ao quadrado
        vp = 1.0 / (self.vp0 * self.vp0)
        theta = None

        # Escalas das direções
        vp_min = 1.0 / (self.pmt.vmax * self.pmt.vmax)
        vp_max = 1.0 / (self.pmt.vmin * self.pmt.vmin)

        self.scale_vp = 0.01 * (vp_max - vp_min)

        # Modelo inicial de epsilon
        mask_eps = np.abs(self.wf.epsilon - np.min(self.wf.epsilon)) < 1e-3 
        self.eps0 = smooth_model(self.wf.epsilon,self.pmt.sigma,mask_eps, parameter = True)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_epsilon_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.eps0.astype(np.float32).tofile(smooth_model_file)
        
        # Modelo inicial de delta
        mask_delta = np.abs(self.wf.delta - np.min(self.wf.delta)) < 1e-3 
        self.delta0 = smooth_model(self.wf.delta,self.pmt.sigma,mask_delta, parameter = True)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_delta_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.delta0.astype(np.float32).tofile(smooth_model_file)
        
        epsilon = self.eps0
        delta = self.delta0

        # Escalas das direções
        self.scale_eps = 0.01 * (self.pmt.epsmax - self.pmt.epsmin)
        self.scale_delta = 0.01 * (self.pmt.deltamax - self.pmt.deltamin)
        
        self.scale_theta = 0.0
        if self.pmt.approximation == "TTI":
            # Modelo inicial de theta
            mask_theta = np.abs(self.wf.theta - np.min(self.wf.theta)) < 1e-3 
            self.theta0 = smooth_model(self.wf.theta,self.pmt.sigma,mask_theta, parameter = True)
            smooth_model_file = (f"{self.pmt.modelFolder}fwi_theta_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
            self.theta0.astype(np.float32).tofile(smooth_model_file)

            theta = self.theta0

            # Escalas das direções
            if self.pmt.multiparameter == True:
                self.scale_theta = 0.01 * (self.pmt.thetamax - self.pmt.thetamin)

        # Porcentagens para liberar os parâmetros
        eps_start = 0.50
        delta_start = 0.70
        theta_start = 0.85

        self.history = []

        for fmax in self.pmt.freqs:
            print(f"\033[31minfo: FWI frequency {fmax} of {self.pmt.freqs}\033[0m")

            self.pmt.fcut = fmax
            self.pmt.tlag = 2.0*np.sqrt(np.pi)/self.pmt.fcut
            self.pmt.nt = int((self.pmt.T + self.pmt.tlag)/self.pmt.dt)+1
            self.pmt.t = np.arange(self.pmt.nt) * self.pmt.dt
            self.wf.initializeWavefields()
            self.wf.createSourceWavelet()

            s_vp_store = []
            y_vp_store = []

            s_eps_store = []
            y_eps_store = []
            s_delta_store = []
            y_delta_store = []

            if self.pmt.approximation == "TTI":
                s_theta_store = []
                y_theta_store = []

            # Gradiente e função objetivo no modelo atual
            X = self.objective_function(vp, epsilon, delta, theta, save_residual=True)
            g_vp, g_eps, g_delta, g_theta = self.calculate_gradient(vp, epsilon, delta, theta)
            
            X0 = X
            self.history.append([X/X0, fmax])

            for itr in range(self.pmt.niter):
                print(f"\033[31minfo: FWI iteration {itr + 1}/{self.pmt.niter} for frequency {fmax}\033[0m")

                progress = (itr + 1) / self.pmt.niter

                print("info: Hierarchical stage")
                print("Vp update: True")
                if progress >= eps_start:
                    print("Epsilon update: True")
                else:
                    print("Epsilon update: False")
                if progress >= delta_start:
                    print("Delta update: True")
                else:
                    print("Delta update: False")
                if self.pmt.approximation == "TTI":
                    if progress >= theta_start:
                        print("Theta update: True")
                    else:
                        print("Theta update: False")

                # Salvar gradiente da iteração atual
                gradient_file = (f"{self.pmt.gradientsFolder}vp_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                (g_vp).astype(np.float32).tofile(gradient_file)
                print(f"info: Vp gradient saved to {gradient_file}")

                if progress >= eps_start:
                    gradient_file = (f"{self.pmt.gradientsFolder}epsilon_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                    (g_eps).astype(np.float32).tofile(gradient_file)
                    print(f"info: Epsilon gradient saved to {gradient_file}")

                if progress >= delta_start:
                    gradient_file = (f"{self.pmt.gradientsFolder}delta_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                    (g_delta).astype(np.float32).tofile(gradient_file)
                    print(f"info: Delta gradient saved to {gradient_file}")

                if self.pmt.approximation == "TTI" and progress >= theta_start:
                    gradient_file = (f"{self.pmt.gradientsFolder}theta_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                    (g_theta).astype(np.float32).tofile(gradient_file)
                    print(f"info: Theta gradient saved to {gradient_file}")

                # Direção de busca: LBFGS
                p_vp = -self.two_loop_recursion(g_vp, s_vp_store, y_vp_store)
                p_vp = p_vp / np.max(np.abs(p_vp))

                p_eps = np.zeros_like(epsilon)
                p_delta = np.zeros_like(delta)
                p_theta = np.zeros_like(theta)

                if progress >= eps_start:
                    p_eps = -self.two_loop_recursion(g_eps, s_eps_store, y_eps_store)
                    p_eps = p_eps / np.max(np.abs(p_eps))

                if progress >= delta_start:
                    p_delta = -self.two_loop_recursion(g_delta, s_delta_store, y_delta_store)
                    p_delta = p_delta / np.max(np.abs(p_delta))

                if self.pmt.approximation == "TTI" and progress >= theta_start:
                    p_theta = -self.two_loop_recursion(g_theta, s_theta_store, y_theta_store)
                    p_theta = p_theta / np.max(np.abs(p_theta))

                # Line search monoparameter
                beta_vp = self.step_length("vp", vp, epsilon, delta, theta,p_vp, p_eps, p_delta, p_theta,g_vp, g_eps, g_delta, g_theta,X, itr)
                
                beta_eps = 0.0
                beta_delta = 0.0
                beta_theta = 0.0

                if progress >= eps_start:
                    beta_eps = self.step_length("epsilon", vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr)

                if progress >= delta_start:
                    beta_delta = self.step_length("delta", vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr)

                if self.pmt.approximation == "TTI" and progress >= theta_start:
                    beta_theta = self.step_length("theta", vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, X, itr)

                # Line search multiparameter
                if progress < eps_start:
                    alpha = 1.0
                else:
                    alpha = self.step_length_multiparameter(vp, epsilon, delta, theta, p_vp, p_eps, p_delta, p_theta, g_vp, g_eps, g_delta, g_theta, beta_vp, beta_eps, beta_delta, beta_theta, X, itr)    
            
                # Atualização do modelo
                vp_new = vp + alpha * beta_vp * p_vp
                vp_new = np.clip(vp_new, vp_min, vp_max)

                epsilon_new = epsilon
                delta_new = delta

                if progress >= eps_start:
                    epsilon_new = epsilon + alpha * beta_eps * p_eps
                    epsilon_new = np.clip(epsilon_new, self.pmt.epsmin, self.pmt.epsmax)
                    
                if progress >= delta_start:
                    delta_new = delta + alpha * beta_delta * p_delta
                    delta_new = np.clip(delta_new, self.pmt.deltamin, self.pmt.deltamax)

                theta_new = theta
                if self.pmt.approximation == "TTI":
                    theta_new = theta + alpha * beta_theta * p_theta
                    theta_new = np.clip(theta_new, self.pmt.thetamin, self.pmt.thetamax)

                X_new = self.objective_function(vp_new, epsilon_new, delta_new, theta_new, save_residual=True)
                g_vp_new, g_eps_new, g_delta_new, g_theta_new = self.calculate_gradient(vp_new, epsilon_new, delta_new, theta_new)

                self.history.append([X_new/X0, fmax])

                # Atualização da memória LBFGS
                s_vp = (vp_new - vp)
                y_vp = (g_vp_new - g_vp)
                sy_vp = np.sum(s_vp * y_vp)

                if sy_vp > 0:
                    s_vp_store.append(s_vp)
                    y_vp_store.append(y_vp)

                if progress >= eps_start:
                    s_eps = (epsilon_new - epsilon)
                    y_eps = (g_eps_new - g_eps)
                    sy_eps = np.sum(s_eps * y_eps)

                    if sy_eps > 0:
                        s_eps_store.append(s_eps)
                        y_eps_store.append(y_eps)

                if progress >= delta_start:
                    s_delta = (delta_new - delta)
                    y_delta = (g_delta_new - g_delta)
                    sy_delta = np.sum(s_delta * y_delta)

                    if sy_delta > 0:
                        s_delta_store.append(s_delta)
                        y_delta_store.append(y_delta)

                if self.pmt.approximation == "TTI" and progress >= theta_start:
                    s_theta = (theta_new - theta)
                    y_theta = (g_theta_new - g_theta)
                    sy_theta = np.sum(s_theta * y_theta)

                    if sy_theta > 0:
                        s_theta_store.append(s_theta)
                        y_theta_store.append(y_theta)

                if len(s_vp_store) > 8:
                    s_vp_store.pop(0)
                    y_vp_store.pop(0)

                if progress >= eps_start:
                    if len(s_eps_store) > 8:
                        s_eps_store.pop(0)
                        y_eps_store.pop(0)

                if progress >= delta_start:
                    if len(s_delta_store) > 8:
                        s_delta_store.pop(0)
                        y_delta_store.pop(0)

                if self.pmt.approximation == "TTI" and progress >= theta_start:
                    if len(s_theta_store) > 8:
                        s_theta_store.pop(0)
                        y_theta_store.pop(0)

                vp = vp_new.copy()
                g_vp = g_vp_new.copy()
                X = X_new

                if progress >= eps_start:
                    epsilon = epsilon_new.copy()
                    g_eps = g_eps_new.copy()

                if progress >= delta_start:    
                    delta = delta_new.copy()
                    g_delta = g_delta_new.copy()

                if self.pmt.approximation == "TTI" and progress >= theta_start:
                    theta = theta_new.copy()
                    g_theta = g_theta_new.copy()
                
                m_it = 1.0 / np.sqrt(vp)
                model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_vp_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                m_it.astype(np.float32).tofile(model_file)
                print(f"info: Model of {itr+1} iteration saved to {model_file}")

                if progress >= eps_start:
                    model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_epsilon_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                    epsilon.astype(np.float32).tofile(model_file)
                    print(f"info: Epsilon model of {itr+1} iteration saved to {model_file}")

                if progress >= delta_start:
                    model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_delta_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                    delta.astype(np.float32).tofile(model_file)
                    print(f"info: Delta model of {itr+1} iteration saved to {model_file}")

                if self.pmt.approximation == "TTI" and progress >= theta_start:
                    model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_theta_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                    theta.astype(np.float32).tofile(model_file)
                    print(f"info: Theta model of {itr+1} iteration saved to {model_file}")
            
        history = np.array(self.history, dtype=np.float32)
        history_file = (f"../outputs/history.txt")
        np.savetxt(history_file,history)

        print(f"info: FWI history saved to {history_file}")

        end_time = time.time()
        print(f"\ninfo: FWI finished in {end_time - start_time:.2f} s")

    def solveFullWaveformInversion(self):
        if self.pmt.multiparameter == True:
            self.solveFullWaveformInversionMultiparameterSimultaneos()
        else:
            self.solveFullWaveformInversionMonoparameter()
    
    