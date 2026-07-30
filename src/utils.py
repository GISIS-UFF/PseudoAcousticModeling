import numpy as np
from numba import jit,prange, njit
import math
import cupy as cp
from CudaKernels import updateWaveEquationKernel
from CudaKernels import updateWaveEquationVTIKernel
from CudaKernels import updateWaveEquationTTIKernel
from CudaKernels import AbsorbingBoundaryCudaKernel
from CudaKernels import updatePsiKernel
from CudaKernels import updateZetaKernel
from CudaKernels import updateWaveEquationCPMLKernel
from CudaKernels import updateWaveEquationVTICPMLKernel
from CudaKernels import calculateGradientVTIKernel
from CudaKernels import calculateGradientTTIKernel
from CudaKernels import calculateAdjointVTIProductsKernel
from CudaKernels import updateAdjointWaveEquationVTIKernel
from CudaKernels import calculateAdjointTTIProductsKernel
from CudaKernels import updateAdjointWaveEquationTTIKernel

#Auxiliar Functions
def ricker(f0, t, t_lag, nt, dt):
    pi = np.pi
    f = f0 / (3 * np.sqrt(pi)) 
    td  = t - t_lag
    source = (1 - 2 * pi * (pi * f * td) * (pi * f * td)) * np.exp(-pi * (pi * f * td) * (pi * f * td)) 
    w = 2.0*pi*np.fft.fftfreq(nt, dt)
    source = np.real(np.fft.ifft(np.fft.fft(source)*np.sqrt(complex(0,1)*w))) 
    return source

@jit(parallel=True)
def Mute(seismogram, shot, rec_x, rec_z, shot_x, shot_z, dt,tlag,shift,window,v0=1500): 
    result = np.zeros_like(seismogram)
    Nt = seismogram.shape[0]
    Nrec = seismogram.shape[1]  
    for rec in prange(Nrec):
        dist = np.sqrt((rec_z[rec] - shot_z[shot])**2 + (rec_x[rec] - shot_x[shot])**2)
        traveltimes = dist/v0 + tlag + shift
        t1 = traveltimes
        t2 = t1 + window
        for i in prange(Nt):
            t = i*dt
            if t <t1:
                result[i,rec] = 0.0
            elif t>=t1 and t<t2:
                result[i,rec] = (t-t1)/(t2-t1) * seismogram[i,rec]
            elif t>=t2:
                result[i,rec] = seismogram[i,rec]
            
    return result

@jit(nopython=True)
def gaussian_kernel(x, z, sigma):
    fator = 1. / (2.*np.pi*sigma*sigma)
    expoente = -(x * x + z * z)/(2.*sigma*sigma)
    return fator * np.exp(expoente)

@jit(nopython=True)
def gaussian_filter2D(sigma):
    kernel_size = int(np.ceil(6.0 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel2d = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    total = 0.0

    for lin in range(kernel_size):
        for col in range(kernel_size):
            x = lin - kernel_size // 2
            y = col - kernel_size // 2
            val = gaussian_kernel(x, y, sigma)
            kernel2d[lin, col] = val
            total += val

    kernel2d /= total

    return kernel2d

@jit(nopython=True, parallel=True)
def smooth_model(f, sigma, water_mask, parameter=False):
    if parameter:
        s = f.copy()
    else:
        s = np.float32(1.0) / f

    s_old = s.copy()

    kernel = gaussian_filter2D(sigma)
    ksize = kernel.shape[0]
    half = ksize // 2

    nz, nx = np.shape(s)

    for z in prange(half, nz - half):
        for x in range(half, nx - half):

            new_value = 0.0
            total = 0.0

            for i in range(ksize):
                for j in range(ksize):
                    zz = z + i - half
                    xx = x + j - half

                    new_value += (kernel[i,j]* s_old[zz,xx])

                    total += kernel[i, j]

            if total > 0.0:
                s[z, x] = new_value / total

    for z in range(half):
        s[z, :] = s[half, :]
        s[nz - 1 - z, :] = s[nz - 1 - half, :]

    for x in range(half):
        s[:, x] = s[:, half]
        s[:, nx - 1 - x] = s[:, nx - 1 - half]

    for x in prange(nx):
        for z in range(nz):
            if water_mask[z, x]:
                s[z, x] = s_old[z, x]
            else:
                break

    if parameter:
        return s

    return np.float32(1.0) / s

def low_pass_filter(data, cutoff, dt, transition=0.3, axis=0): 
    data = np.asarray(data)
    nt = data.shape[axis]

    # Padding
    nt_pad = 2 * nt

    if data.ndim == 1:
        data_pad = np.zeros(nt_pad, dtype=data.dtype)
        data_pad[:nt] = data

    elif data.ndim == 2:
        data_pad = np.zeros((nt_pad, data.shape[1]), dtype=data.dtype)
        data_pad[:nt, :] = data

    fft_data = np.fft.rfft(data_pad, axis=axis)
    frequencies = np.fft.rfftfreq(nt_pad, d=dt)

    fpass = cutoff 
    fstop = cutoff * (1.0 + transition)

    mask = np.ones_like(frequencies)

    mask[frequencies >= fstop] = 0.0

    for idx in range(len(frequencies)):
        if frequencies[idx] > fpass and frequencies[idx] < fstop:
            mask[idx] = 1.0 - (frequencies[idx] - fpass) / (fstop - fpass)

    if data.ndim == 2:
        mask = mask[:, np.newaxis]

    data_filt_pad = np.fft.irfft(fft_data * mask, n=nt_pad, axis=axis)

    # Remove o padding
    if data.ndim == 1:
        data_filt = data_filt_pad[:nt]

    elif data.ndim == 2:
        data_filt = data_filt_pad[:nt, :]

    return data_filt.astype(data.dtype)

@jit(nopython=True, parallel=True)
def AGC(data, dt, window=0.2):
    nt, nrec = data.shape
    window_samples = int(round(window / dt))
    half_window = window_samples // 2
    agc_data = np.zeros_like(data, dtype=np.float32)

    for i in prange(nt):
        start = max(0, i - half_window)
        end = min(nt, i + half_window + 1)

        for rec in prange(nrec):
            soma = 0.0

            for k in prange(start, end):
                soma += abs(data[k, rec])

            amplitude = soma / (end - start)

            agc_data[i, rec] = data[i, rec] / (amplitude + 1e-10)

    return agc_data

# CPML Auxiliar Functions
@njit(inline = "always")
def horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j):
    d = 0.
    alpha = 0.
    if i < N_abc:
        points_CPML = (N_abc - i - 1.)*dx
        posicao_relativa = points_CPML / (N_abc * dx)
        d = d0/(2 * N_abc * dx) * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1. - posicao_relativa**2)

    elif i >= nx_abc - N_abc:
        points_CPML = (i - nx_abc + N_abc)*dx
        posicao_relativa = points_CPML / (N_abc * dx)
        d = d0/(2 * N_abc * dx) * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1. - posicao_relativa**2)

    ax = np.exp(-(d + alpha) * dt)
    if (np.abs((d + alpha)) > 1e-10):
        bx = (d / (d + alpha)) * (ax - 1.)
    
    return ax, bx

@njit(inline = "always")
def vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j):
    d = 0.
    alpha = 0. 
    if j < N_abc:
        points_CPML = (N_abc - j - 1.)*dz
        posicao_relativa = points_CPML / (N_abc * dz)
        d = d0/(2 * N_abc * dz) * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1. - posicao_relativa**2)

    elif j >= nz_abc - N_abc:
        points_CPML = (j - nz_abc + N_abc)*dz
        posicao_relativa = points_CPML / (N_abc * dz)
        d = d0/(2 * N_abc * dz) * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1. - posicao_relativa**2)

    az = np.exp(-(d + alpha) * dt)
    if (np.abs((d + alpha)) > 1e-10):
        bz = (d / (d + alpha)) * (az - 1.)
       
    return az, bz

@jit(nopython=True, parallel=True)
def updatePsi(PsixFR, PsixFL, PsizFU, PsizFD, nx_abc, nz_abc, Uc, dx,dz, N_abc, f_pico, d0, dt, vp):
    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0

    for j in prange(4, nz_abc - 4):
        for i in prange(4, N_abc):

            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)

            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / dx
            
            PsixFL[j, i] = ax * PsixFL[j, i] + bx * px

    for j in prange(4, nz_abc - 4):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)

            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)

            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / dx
            
            PsixFR[j, idx] = ax * PsixFR[j, idx] + bx * px

    for j in prange(4, N_abc):
        for i in prange(4, nx_abc - 4):

            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)
            
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz 
            
            PsizFU[j, i] = az * PsizFU[j, i] + bz * pz

    for j in prange(nz_abc - N_abc, nz_abc - 4):  
        jdx = j - (nz_abc - N_abc)
        for i in prange(4, nx_abc - 4):

            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)

            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz 
            
            PsizFD[jdx, i] = az * PsizFD[jdx, i] + bz * pz

    return PsixFR, PsixFL, PsizFU, PsizFD

@jit(nopython=True, parallel=True)
def updateZeta(PsixFR, PsixFL, ZetaxFR, ZetaxFL,PsizFU, PsizFD, ZetazFU, ZetazFD, nx_abc, nz_abc, Uc, dx, dz, N_abc, f_pico, d0, dt, vp):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0
    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0
    for j in prange(4, nz_abc - 4):
        for i in prange(4, N_abc):
            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)

            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx

            ZetaxFL[j, i] = ax * ZetaxFL[j, i] + bx * (pxx + psix)

    for j in prange(4, nz_abc - 4):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc) 

            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)

            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx

            ZetaxFR[j, idx] = ax * ZetaxFR[j, idx] + bx * (pxx + psix)

    for j in prange(4, N_abc):
        for i in prange(4, nx_abc - 4):
            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)
           
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz
            
            ZetazFU[j, i] = az * ZetazFU[j, i] + bz * (pzz + psiz)

    for j in prange(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc) 
        for i in prange(4, nx_abc - 4):
            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)
            
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            ZetazFD[jdx, i] = az * ZetazFD[jdx, i] + bz * (pzz + psiz) 

    return ZetaxFR, ZetaxFL, ZetazFU, ZetazFD

# Cerjan Apply
@jit(parallel=True, nopython=True)
def AbsorbingBoundary(N_abc, nz_abc, nx_abc, f, A):
    for y in prange(nz_abc):
        for x in range(N_abc):
            f[y, x] *= A[x]
        for x in range(nx_abc - N_abc, nx_abc):
            f[y, x] *= A[nx_abc - 1 - x] 
    for x in prange(nx_abc):
        for y in range(N_abc):
            f[y, x] *= A[y]
        for y in range(nz_abc - N_abc, nz_abc):
            f[y, x] *= A[nz_abc - 1 - y]  
            
    return f

# WaveEquation every type
@jit(nopython=True,parallel=True)
def updateWaveEquation(Uf,Uc,vp,nz,nx,dz,dx,dt):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0

    for j in prange(4,nz-4):
        for i in prange(4,nx-4):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) + c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) + c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * Uc[j, i] - Uf[j, i]

    return Uf

@jit(nopython=True,parallel=True)
def updateWaveEquationVTI(Uf, Uc, nx, nz, dt, dx, dz, vp, epsilon, delta):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0

    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0

    for j in prange(4,nz-4):
        for i in prange(4,nx-4):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*(px*px)*(pz*pz)
            den = (1.0 + 2.0*epsilon[j,i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0*(1.0 + delta[j,i])*(px*px)*(pz*pz)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * pxx + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * pzz

    return Uf

@jit(nopython=True,parallel=True)
def updateWaveEquationTTI(Uf, Uc, nx, nz, dt, dx, dz, vp, epsilon, delta, theta):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0
    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0
    
    for j in prange(4,nz-4):
        for i in prange(4,nx-4):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
            a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
            a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
            a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

            a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
            a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
            a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
            a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

            a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
            a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
            a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
            a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

            a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
            a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
            a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
            a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz

            xi = (px*np.cos(theta[j,i]) - pz*np.sin(theta[j,i]))
            eta = (px*np.sin(theta[j,i]) + pz*np.cos(theta[j,i]))
            xi2 = xi*xi
            eta2 = eta*eta
            xi4 = xi2*xi2
            eta4 = eta2*eta2

            num = (-2.0* (epsilon[j,i]-delta[j,i])* xi2* eta2)
            den = ((1.0 + 2.0 * epsilon[j,i]) * xi4 + eta4 + 2.0* (1.0 + delta[j,i]) * xi2 * eta2)

            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * pxx + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * pzz - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.* theta[j,i]) * pxz

    return Uf

#Adjoint WaveEquation every type
@jit(nopython=True,parallel=True)
def updateAdjointWaveEquationVTI(Uf, Uc, P, AUc, BUc, QCxUc, QCzUc, nx, nz, dt, dx, dz, vp, epsilon, delta):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0

    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0

    AUc.fill(0)
    BUc.fill(0)
    QCxUc.fill(0)
    QCzUc.fill(0)

    for j in range(4, nz - 4):
        for i in prange(4, nx - 4):
        
            pxx = (c0 * P[j, i]
                + c1 * (P[j, i + 1] + P[j, i - 1])
                + c2 * (P[j, i + 2] + P[j, i - 2])
                + c3 * (P[j, i + 3] + P[j, i - 3])
                + c4 * (P[j, i + 4] + P[j, i - 4])) / (dx * dx)

            pzz = (c0 * P[j, i]
                + c1 * (P[j + 1, i] + P[j - 1, i])
                + c2 * (P[j + 2, i] + P[j - 2, i])
                + c3 * (P[j + 3, i] + P[j - 3, i])
                + c4 * (P[j + 4, i] + P[j - 4, i])) / (dz * dz)

            px = (a1 * (P[j, i + 1] - P[j, i - 1])
                + a2 * (P[j, i + 2] - P[j, i - 2])
                + a3 * (P[j, i + 3] - P[j, i - 3])
                + a4 * (P[j, i + 4] - P[j, i - 4])) / dx

            pz = (a1 * (P[j + 1, i] - P[j - 1, i])
                + a2 * (P[j + 2, i] - P[j - 2, i])
                + a3 * (P[j + 3, i] - P[j - 3, i])
                + a4 * (P[j + 4, i] - P[j - 4, i])) / dz

            eps = epsilon[j, i]
            delt = delta[j, i]

            px2 = px * px
            pz2 = pz * pz

            px4 = px2 * px2
            pz4 = pz2 * pz2

            num = (-2.0* (eps - delt)* px2* pz2)

            den = ((1.0 + 2.0 * eps) * px4 + pz4+ 2.0 * (1.0 + delt) * px2 * pz2)

            if abs(den) < 1.0e-12:
                Sd = 0.0
                Cx = 0.0
                Cz = 0.0

            else:
                Sd = num / den

                den2 = np.float64(den * den)

                factor = np.float64((4.0 * (eps - delt) * ((1.0 + 2.0 * eps) * px4 - pz4)/ den2))

                Cx = factor * px * pz2
                Cz = -factor * px2 * pz

            A = 1.0 + 2.0 * eps + Sd
            B = 1.0 + Sd
            Q = pxx + pzz

            AUc[j, i] = A * Uc[j, i]
            BUc[j, i] = B * Uc[j, i]

            QCxUc[j, i] = Uc[j, i] * Q * Cx
            QCzUc[j, i] = Uc[j, i] * Q * Cz

    for j in range(4, nz - 4):
        for i in prange(4, nx - 4):
        

            dxx_AUc = (c0 * AUc[j, i]
                + c1 * (AUc[j, i + 1] + AUc[j, i - 1])
                + c2 * (AUc[j, i + 2] + AUc[j, i - 2])
                + c3 * (AUc[j, i + 3] + AUc[j, i - 3])
                + c4 * (AUc[j, i + 4] + AUc[j, i - 4])) / (dx * dx)

            dzz_BUc = (c0 * BUc[j, i]
                + c1 * (BUc[j + 1, i] + BUc[j - 1, i])
                + c2 * (BUc[j + 2, i] + BUc[j - 2, i])
                + c3 * (BUc[j + 3, i] + BUc[j - 3, i])
                + c4 * (BUc[j + 4, i] + BUc[j - 4, i])) / (dz * dz)

            dx_QCxUc = (a1 * (QCxUc[j, i + 1] - QCxUc[j, i - 1])
                        + a2 * (QCxUc[j, i + 2] - QCxUc[j, i - 2])
                        + a3 * (QCxUc[j, i + 3] - QCxUc[j, i - 3])
                        + a4 * (QCxUc[j, i + 4] - QCxUc[j, i - 4])) / dx

            dz_QCzUc = (a1 * (QCzUc[j + 1, i] - QCzUc[j - 1, i])
                        + a2 * (QCzUc[j + 2, i] - QCzUc[j - 2, i])
                        + a3 * (QCzUc[j + 3, i] - QCzUc[j - 3, i])
                        + a4 * (QCzUc[j + 4, i] - QCzUc[j - 4, i])) / dz

            Uf[j, i] = (2.0 * Uc[j, i] - Uf[j, i] + vp[j, i] * vp[j, i] * dt * dt * (dxx_AUc + dzz_BUc - dx_QCxUc - dz_QCzUc))

    return Uf

@jit(nopython=True,parallel=True)
def updateAdjointWaveEquationTTI(Uf, Uc, P, AUc, BUc, HUc, QCxUc, QCzUc, nx, nz, dt, dx, dz, vp, epsilon, delta, theta):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0

    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0

    AUc.fill(0)
    BUc.fill(0)
    HUc.fill(0)
    QCxUc.fill(0)
    QCzUc.fill(0)

    for j in range(4, nz - 4):
        for i in prange(4, nx - 4):
            pxx = (c0 * P[j, i] + 
                   c1 * (P[j, i+1] + P[j, i-1]) + 
                   c2 * (P[j, i+2] + P[j, i-2]) +
                   c3 * (P[j, i+3] + P[j, i-3]) +
                   c4 * (P[j, i+4] + P[j, i-4])) / (dx * dx)
            pzz = (c0 * P[j, i] + 
                   c1 * (P[j+1, i] + P[j-1, i]) + 
                   c2 * (P[j+2, i] + P[j-2, i]) + 
                   c3 * (P[j+3, i] + P[j-3, i]) + 
                   c4 * (P[j+4, i] + P[j-4, i])) / (dz * dz)
            px = (a1*(P[j, i+1] - P[j, i-1]) +
                a2*(P[j, i+2] - P[j, i-2]) +
                a3*(P[j, i+3] - P[j, i-3]) +
                a4*(P[j, i+4] - P[j, i-4])) / dx
            pz = (a1 * (P[j+1, i] - P[j-1, i]) +
                a2 * (P[j+2, i] - P[j-2, i]) +
                a3 * (P[j+3, i] - P[j-3, i]) +
                a4 * (P[j+4, i] - P[j-4, i])) / dz

            xi = (px*np.cos(theta[j,i]) - pz*np.sin(theta[j,i]))
            eta = (px*np.sin(theta[j,i]) + pz*np.cos(theta[j,i]))
            xi2 = xi*xi
            eta2 = eta*eta
            xi4 = xi2*xi2
            eta4 = eta2*eta2

            num = (-2.0* (epsilon[j,i]-delta[j,i])* xi2* eta2)
            den = ((1.0 + 2.0 * epsilon[j,i]) * xi4 + eta4 + 2.0* (1.0 + delta[j,i]) * xi2 * eta2)

            if abs(den) < 1e-12:
                Sd = 0.0
                Cx = 0.0
                Cz = 0.0
            else:
                Sd = num / den
                den2 = np.float64(den * den)
                K = ((1.0 + 2.0 * epsilon[j,i]) * xi4 - eta4)
                factor = np.float64((4.0 * (epsilon[j,i]-delta[j,i]) * xi * eta * K/ den2))

                Cx = factor * pz 
                Cz = -factor * px 

            cos2 = np.cos(theta[j,i]) * np.cos(theta[j,i])
            sin2 = np.sin(theta[j,i]) * np.sin(theta[j,i])

            A = ((1.0 + 2.0 * epsilon[j,i]) * cos2 + sin2 + Sd)
            B = ((1.0 + 2.0 * epsilon[j,i]) * sin2 + cos2 + Sd)
            H = (2.0 * epsilon[j,i] * np.sin(2.0 * theta[j,i]))
            Q = pxx + pzz

            AUc[j, i] = A * Uc[j, i]
            BUc[j, i] = B * Uc[j, i]
            HUc[j, i] = H * Uc[j, i]
            QCxUc[j, i] = (Q*Cx*Uc[j, i])
            QCzUc[j, i] = (Q*Cz*Uc[j, i])

    for j in prange(4, nz - 4):
        for i in range(4, nx - 4):

            dxx_AUc = (c0 * AUc[j, i]+
            c1 * (AUc[j, i + 1] + AUc[j, i - 1])+
            c2 * (AUc[j, i + 2] + AUc[j, i - 2])+
            c3 * (AUc[j, i + 3] + AUc[j, i - 3])+
            c4 * (AUc[j, i + 4] + AUc[j, i - 4])) / (dx*dx)

            dzz_BUc = (c0 * BUc[j, i]+
            c1 * (BUc[j + 1, i] + BUc[j - 1, i])+
            c2 * (BUc[j + 2, i] + BUc[j - 2, i])+
            c3 * (BUc[j + 3, i] + BUc[j - 3, i])+
            c4 * (BUc[j + 4, i] + BUc[j - 4, i])) / (dz*dz)

            dxz_HUc = (a1 * a1 * (HUc[j + 1, i + 1] - HUc[j - 1, i + 1] + HUc[j - 1, i - 1] - HUc[j + 1, i - 1])+ 
            a1 * a2 * (HUc[j + 2, i + 1] - HUc[j - 2, i + 1] + HUc[j - 2, i - 1] - HUc[j + 2, i - 1])+ 
            a1 * a3 * (HUc[j + 3, i + 1] - HUc[j - 3, i + 1] + HUc[j - 3, i - 1] - HUc[j + 3, i - 1])+ 
            a1 * a4 * (HUc[j + 4, i + 1] - HUc[j - 4, i + 1] + HUc[j - 4, i - 1] - HUc[j + 4, i - 1])+

            a2 * a1 * (HUc[j + 1, i + 2] - HUc[j - 1, i + 2] + HUc[j - 1, i - 2] - HUc[j + 1, i - 2])+ 
            a2 * a2 * (HUc[j + 2, i + 2] - HUc[j - 2, i + 2] + HUc[j - 2, i - 2] - HUc[j + 2, i - 2])+ 
            a2 * a3 * (HUc[j + 3, i + 2] - HUc[j - 3, i + 2] + HUc[j - 3, i - 2] - HUc[j + 3, i - 2])+ 
            a2 * a4 * (HUc[j + 4, i + 2] - HUc[j - 4, i + 2] + HUc[j - 4, i - 2] - HUc[j + 4, i - 2])+

            a3 * a1 * (HUc[j + 1, i + 3] - HUc[j - 1, i + 3] + HUc[j - 1, i - 3] - HUc[j + 1, i - 3])+ 
            a3 * a2 * (HUc[j + 2, i + 3] - HUc[j - 2, i + 3] + HUc[j - 2, i - 3] - HUc[j + 2, i - 3])+ 
            a3 * a3 * (HUc[j + 3, i + 3] - HUc[j - 3, i + 3] + HUc[j - 3, i - 3] - HUc[j + 3, i - 3])+ 
            a3 * a4 * (HUc[j + 4, i + 3] - HUc[j - 4, i + 3] + HUc[j - 4, i - 3] - HUc[j + 4, i - 3])+

            a4 * a1 * (HUc[j + 1, i + 4] - HUc[j - 1, i + 4] + HUc[j - 1, i - 4] - HUc[j + 1, i - 4])+ 
            a4 * a2 * (HUc[j + 2, i + 4] - HUc[j - 2, i + 4] + HUc[j - 2, i - 4] - HUc[j + 2, i - 4])+ 
            a4 * a3 * (HUc[j + 3, i + 4] - HUc[j - 3, i + 4] + HUc[j - 3, i - 4] - HUc[j + 3, i - 4])+ 
            a4 * a4 * (HUc[j + 4, i + 4] - HUc[j - 4, i + 4] + HUc[j - 4, i - 4] - HUc[j + 4, i - 4])) / (dx * dz)

            dx_QCxUc = (a1 * (QCxUc[j, i + 1] - QCxUc[j, i - 1])+ 
            a2 * (QCxUc[j, i + 2] - QCxUc[j, i - 2])+ 
            a3 * (QCxUc[j, i + 3] - QCxUc[j, i - 3])+ 
            a4 * (QCxUc[j, i + 4] - QCxUc[j, i - 4])) / dx

            dz_QCzUc = (a1 * (QCzUc[j + 1, i] - QCzUc[j - 1, i])+ 
            a2 * (QCzUc[j + 2, i] - QCzUc[j - 2, i])+ 
            a3 * (QCzUc[j + 3, i] - QCzUc[j - 3, i])+ 
            a4 * (QCzUc[j + 4, i] - QCzUc[j - 4, i])) / dz

            Uf[j, i] = (2.0 * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) *(dxx_AUc + dzz_BUc - dxz_HUc - dx_QCxUc - dz_QCzUc))

    return Uf

# CPML WaveEquation types
@jit(nopython=True, parallel=True)
def updateWaveEquationCPML(Uf, Uc, vp, nx_abc, nz_abc, dz, dx, dt, PsixFR, PsixFL, PsizFU, PsizFD, ZetaxFR, ZetaxFL, ZetazFU, ZetazFD, N_abc):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0
    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0

    # Região Interior 
    for j in prange(N_abc, nz_abc - N_abc):
        for i in prange(N_abc, nx_abc - N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * Uc[j, i] - Uf[j, i]

    # Região Esquerda 
    for j in prange(N_abc, nz_abc - N_abc):
        for i in prange(4, N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx

            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + ZetaxFL[j, i]) + 2 * Uc[j, i] - Uf[j, i]
            
    # Região Direita
    for j in range(N_abc, nz_abc - N_abc):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
                idx = i - (nx_abc - N_abc)
                pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
                pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
                psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                        a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                        a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                        a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx
    
                Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + ZetaxFR[j, idx]) + 2 * Uc[j, i] - Uf[j, i]

    # Região Superior 
    for j in prange(4, N_abc):
        for i in range(N_abc, nx_abc - N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz          

            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psiz + ZetazFU[j, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Região Inferior
    for j in prange(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc)
        for i in range(N_abc, nx_abc - N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psiz + ZetazFD[jdx, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Quina Superior Esquerda
    for j in range(4, N_abc):
        for i in prange(4, N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz   
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFL[j, i] + ZetazFU[j, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Quina Superior Direita 
    for j in range(4, N_abc):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                        a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                        a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                        a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz          
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFR[j, idx] + ZetazFU[j, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Quina Inferior Esquerda 
    for j in range(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc)
        for i in prange(4, N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFL[j, i] + ZetazFD[jdx, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Quina Inferior Direita 
    for j in range(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc)
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)
        
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx   
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFR[j, idx] + ZetazFD[jdx, i]) + 2 * Uc[j, i] - Uf[j, i]

    return Uf

@jit(nopython=True, parallel=True)
def updateWaveEquationVTICPML(Uf, Uc, dt, dx, dz, vp, epsilon, delta,
                               nx_abc, nz_abc, PsixFR, PsixFL,PsizFU,PsizFD, ZetaxFR, ZetaxFL,ZetazFU, ZetazFD, N_abc):
    
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0
    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0

    # Região Interior
    for j in prange(N_abc, nz_abc - N_abc):
        for i in prange(N_abc, nx_abc - N_abc):  
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*(px*px)*(pz*pz)
            den = (1.0 + 2.0*epsilon[j,i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0*(1.0 + delta[j,i])*(px*px)*(pz*pz)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * pxx + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * pzz

    # Região Esquerda
    for j in range(N_abc, nz_abc - N_abc):
        for i in prange(4, N_abc):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * pzz          
                  
    # Região Direita
    for j in range(N_abc, nz_abc - N_abc):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)
        
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * pzz          
                     
    # Região Superior
    for j in range(4, N_abc):
        for i in prange(N_abc, nx_abc - N_abc):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFU[j,i])                   

    # Região Inferior
    for j in range(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc)
        for i in prange(N_abc, nx_abc - N_abc):
       
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz   
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFD[jdx,i])                   

    # Quina Superior Esquerda
    for j in range(4, N_abc):
        for i in prange(4, N_abc):
    
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx            
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFU[j,i])                   

    # Quina Superior Direita
    for j in range(4, N_abc):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)
        
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx          
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFU[j,i])                   
    
    # Quina Inferior Esquerda
    for j in range(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc)
        for i in prange(4, N_abc):
        
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFD[jdx,i])                   
    
    # Quina Inferior Direita
    for j in range(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc)
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)
        
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFD[jdx,i])                   

    return Uf

#Anisotropic Gradients
@jit(nopython=True,parallel=True)
def calculateGradientVTI(current, adj, epsilon_partial, delta_partial, dx, dz, nx, nz,epsilon,delta):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0
    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0

    for j in prange(4,nz-4):
        for i in prange(4,nx-4):
    
            pxx = (c0 * current[j, i] + 
                c1 * (current[j, i+1] + current[j, i-1]) + 
                c2 * (current[j, i+2] + current[j, i-2]) +
                c3 * (current[j, i+3] + current[j, i-3]) +
                c4 * (current[j, i+4] + current[j, i-4])) / (dx * dx)
            pzz = (c0 * current[j, i] + 
                c1 * (current[j+1, i] + current[j-1, i]) + 
                c2 * (current[j+2, i] + current[j-2, i]) + 
                c3 * (current[j+3, i] + current[j-3, i]) + 
                c4 * (current[j+4, i] + current[j-4, i])) / (dz * dz)
            px = (a1*(current[j, i+1] - current[j, i-1]) +
                a2*(current[j, i+2] - current[j, i-2]) +
                a3*(current[j, i+3] - current[j, i-3]) +
                a4*(current[j, i+4] - current[j, i-4])) / dx
            pz = (a1 * (current[j+1, i] - current[j-1, i]) +
                a2 * (current[j+2, i] - current[j-2, i]) +
                a3 * (current[j+3, i] - current[j-3, i]) +
                a4 * (current[j+4, i] - current[j-4, i])) / dz
            
            eps = np.float64(epsilon[j, i])
            delta = np.float64(delta[j, i])

            num = -2.0*(eps-delta)*(px*px)*(pz*pz)
            den = (1.0 + 2.0*eps)*(px*px*px*px) + (pz*pz*pz*pz) + 2.0*(1.0 + delta)*(px*px)*(pz*pz)

            dnum_deps = -2.0*px*px*pz*pz
            dnum_ddelta = 2.0*px*px*pz*pz
            dden_deps = 2.0*px*px*px*px
            dden_ddelta = 2.0*px*px*pz*pz

            if abs(den) < 1e-150:
                dSd_deps = 0.0
                dSd_ddelta = 0.0                    
            else:
                dSd_deps = (dnum_deps*den - num*dden_deps)/(den*den)
                dSd_ddelta = (dnum_ddelta*den - num*dden_ddelta)/(den*den)

            dP_deps = ((-2.0 - dSd_deps)*pxx - dSd_deps*pzz)
            dP_ddelta = (-dSd_ddelta*(pxx + pzz))

            epsilon_partial[j,i] += adj[j,i]*dP_deps
            delta_partial[j,i] += adj[j,i]*dP_ddelta
    
    return epsilon_partial,delta_partial

@jit(nopython=True,parallel=True)
def calculateGradientTTI(current, adj, epsilon_partial, delta_partial, theta_partial, dx, dz, nx, nz, epsilon, delta, theta):
    c0 = -1435.0 / 504.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0
    a1 = 4.0 / 5.0
    a2 = -1.0 / 5.0
    a3 = 4.0 / 105.0
    a4 = -1.0 / 280.0

    for j in prange(4,nz-4):
        for i in prange(4,nx-4):
        
            pxx = (c0 * current[j, i] + 
                c1 * (current[j, i+1] + current[j, i-1]) + 
                c2 * (current[j, i+2] + current[j, i-2]) +
                c3 * (current[j, i+3] + current[j, i-3]) +
                c4 * (current[j, i+4] + current[j, i-4])) / (dx * dx)
            pzz = (c0 * current[j, i] + 
                c1 * (current[j+1, i] + current[j-1, i]) + 
                c2 * (current[j+2, i] + current[j-2, i]) + 
                c3 * (current[j+3, i] + current[j-3, i]) + 
                c4 * (current[j+4, i] + current[j-4, i])) / (dz * dz)
            pxz = (a1*a1*(current[j+1,i+1] - current[j-1,i+1] + current[j-1,i-1] - current[j+1,i-1]) +
                    a1*a2*(current[j+2,i+1] - current[j-2,i+1] + current[j-2,i-1] - current[j+2,i-1]) +
                    a1*a3*(current[j+3,i+1] - current[j-3,i+1] + current[j-3,i-1] - current[j+3,i-1]) +
                    a1*a4*(current[j+4,i+1] - current[j-4,i+1] + current[j-4,i-1] - current[j+4,i-1]) +

                    a2*a1*(current[j+1,i+2] - current[j-1,i+2] + current[j-1,i-2] - current[j+1,i-2]) +
                    a2*a2*(current[j+2,i+2] - current[j-2,i+2] + current[j-2,i-2] - current[j+2,i-2]) +
                    a2*a3*(current[j+3,i+2] - current[j-3,i+2] + current[j-3,i-2] - current[j+3,i-2]) +
                    a2*a4*(current[j+4,i+2] - current[j-4,i+2] + current[j-4,i-2] - current[j+4,i-2]) +

                    a3*a1*(current[j+1,i+3] - current[j-1,i+3] + current[j-1,i-3] - current[j+1,i-3]) +
                    a3*a2*(current[j+2,i+3] - current[j-2,i+3] + current[j-2,i-3] - current[j+2,i-3]) +
                    a3*a3*(current[j+3,i+3] - current[j-3,i+3] + current[j-3,i-3] - current[j+3,i-3]) +
                    a3*a4*(current[j+4,i+3] - current[j-4,i+3] + current[j-4,i-3] - current[j+4,i-3]) +

                    a4*a1*(current[j+1,i+4] - current[j-1,i+4] + current[j-1,i-4] - current[j+1,i-4]) +
                    a4*a2*(current[j+2,i+4] - current[j-2,i+4] + current[j-2,i-4] - current[j+2,i-4]) +
                    a4*a3*(current[j+3,i+4] - current[j-3,i+4] + current[j-3,i-4] - current[j+3,i-4]) +
                    a4*a4*(current[j+4,i+4] - current[j-4,i+4] + current[j-4,i-4] - current[j+4,i-4])) / (dz * dx)
            px = (a1*(current[j, i+1] - current[j, i-1]) +
                a2*(current[j, i+2] - current[j, i-2]) +
                a3*(current[j, i+3] - current[j, i-3]) +
                a4*(current[j, i+4] - current[j, i-4])) / dx
            pz = (a1 * (current[j+1, i] - current[j-1, i]) +
                a2 * (current[j+2, i] - current[j-2, i]) +
                a3 * (current[j+3, i] - current[j-3, i]) +
                a4 * (current[j+4, i] - current[j-4, i])) / dz
            
            eps = np.float64(epsilon[j, i])
            delt = np.float64(delta[j, i])
            th = np.float64(theta[j, i])

            h = px*np.cos(th) - pz*np.sin(th)
            q = px*np.sin(th) + pz*np.cos(th)

            num = -2.0*(eps-delt)*(h*h)*(q*q)
            den = (1.0 + 2.0*eps)*(h*h*h*h) + (q*q*q*q) + 2.0*(1.0 + delt)*(h*h)*(q*q)

            dnum_deps = -2.0*h*h*q*q
            dnum_ddelta = 2.0*h*h*q*q
            dden_deps = 2.0*h*h*h*h
            dden_ddelta = 2.0*h*h*q*q

            dnum_dtheta = -2.0*(eps-delt)*(-2.0*h*q*q*q + 2.0*h*h*h*q)
            dden_dtheta = (-4.0*(1.0 + 2.0*eps)*h*h*h*q+ 4.0*h*q*q*q+ 2.0*(1.0 + delt)*(-2.0*h*q*q*q + 2.0*h*h*h*q))

            if abs(den) < 1e-150:
                Sd = 0.0
                dSd_deps = 0.0
                dSd_ddelta = 0.0
                dSd_dtheta = 0.0
            else:
                Sd = num / den
                dSd_deps = (dnum_deps*den - num*dden_deps)/(den*den)
                dSd_ddelta = (dnum_ddelta*den - num*dden_ddelta)/(den*den)
                dSd_dtheta = (dnum_dtheta*den - num*dden_dtheta)/(den*den)

            dA_dtheta = 2.0*eps*np.sin(2.0*th) - dSd_dtheta
            dB_dtheta = -2.0*eps*np.sin(2.0*th) - dSd_dtheta
            dC_dtheta = 4.0*eps*np.cos(2.0*th)

            dP_deps = (-(2.0*np.cos(th)*np.cos(th) + dSd_deps)*pxx - (2.0*np.sin(th)*np.sin(th) + dSd_deps)*pzz + 2.0*np.sin(2.0*th)*pxz)
            dP_ddelta = (-dSd_ddelta*(pxx + pzz))
            dP_dtheta = (dA_dtheta*pxx + dB_dtheta*pzz + dC_dtheta*pxz)

            epsilon_partial[j,i] += adj[j,i]*dP_deps
            delta_partial[j,i] += adj[j,i]*dP_ddelta
            theta_partial[j,i] += adj[j,i]*dP_dtheta
    
    return epsilon_partial, delta_partial, theta_partial

#GPU Cerjan Apply
def AbsorbingBoundaryGPU(Uf,Uc,N_abc,nx,nz,A):
    total_size = nz * nx
    treads_per_block = 256
    blocks_per_grid = (total_size + treads_per_block - 1) // treads_per_block
    AbsorbingBoundaryCudaKernel((blocks_per_grid,), (treads_per_block,), (Uf,Uc,N_abc,nz,nx,A))
    return Uf,Uc

#GPU WaveEquation 
@staticmethod
def updateWaveEquationGPU(Uf, Uc, vp, nz, nx, dz, dx, dt):
    total_pixels = nz * nx
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    updateWaveEquationKernel((blocks_per_grid,),(threads_per_block,),(Uf,Uc,vp,np.int32(nz),np.int32(nx),np.float32(dz),np.float32(dx),np.float32(dt)))

@staticmethod
def updateWaveEquationVTIGPU(Uf, Uc, nx, nz, dt, dx, dz, vp, epsilon, delta):
    total_pixels = nz * nx
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    updateWaveEquationVTIKernel((blocks_per_grid,),(threads_per_block,),(Uf,Uc,np.int32(nx),np.int32(nz),np.float32(dt),np.float32(dx),np.float32(dz),vp,epsilon,delta))
 
@staticmethod
def updateWaveEquationTTIGPU(Uf, Uc, nx, nz, dt, dx, dz, vp, epsilon, delta, theta):
    total_pixels = nz * nx
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    updateWaveEquationTTIKernel((blocks_per_grid,),(threads_per_block,),(Uf,Uc,np.int32(nx),np.int32(nz),np.float32(dt),np.float32(dx),np.float32(dz),vp,epsilon,delta,theta))

# GPU Adjoint WaveEquation
def solveAdjointWaveEquationVTICuda(Uf,Uc,P,AUc,BUc,QCxUc,QCzUc,dt,dx,dz,nx,nz,vp,epsilon,delta):
    total_pixels = nz * nx
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block
    calculateAdjointVTIProductsKernel((blocks_per_grid,),(threads_per_block,),(Uc,P,AUc,BUc,QCxUc,QCzUc,np.int32(nx),np.int32(nz),np.float32(dx),np.float32(dz),epsilon,delta))
    updateAdjointWaveEquationVTIKernel((blocks_per_grid,),(threads_per_block,),(Uf,Uc,AUc,BUc,QCxUc,QCzUc,np.int32(nx),np.int32(nz),np.float32(dt),np.float32(dx),np.float32(dz),vp))

def solveAdjointWaveEquationTTICuda(Uf, Uc, P, AUc, BUc, HUc, QCxUc, QCzUc, dt, dx, dz, nx, nz, vp, epsilon, delta, theta):
    total_size = nx * nz
    threads_per_block = 256
    blocks_per_grid = (total_size + threads_per_block - 1) // threads_per_block
    calculateAdjointTTIProductsKernel((blocks_per_grid,),(threads_per_block,),(Uc,P,AUc,BUc,HUc,QCxUc,QCzUc,np.int32(nx),np.int32(nz),np.float32(dx),np.float32(dz),epsilon,delta,theta))
    updateAdjointWaveEquationTTIKernel((blocks_per_grid,),(threads_per_block,),(Uf,Uc,AUc,BUc,HUc,QCxUc,QCzUc,np.int32(nx),np.int32(nz),np.float32(dt),np.float32(dx),np.float32(dz),vp))

# CPML Auxiliar Functions
def updatePsiGPU(PsixFR, PsixFL, PsizFU, PsizFD, nx_abc, nz_abc, Uc, dx,dz, N_abc, f_pico, d0, dt, vp):
    total_pixels = nz_abc * nx_abc
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    updatePsiKernel((blocks_per_grid,),(threads_per_block,),(PsixFR, PsixFL, PsizFU, PsizFD,np.int32(nx_abc),np.int32(nz_abc),Uc,np.float32(dx),np.float32(dz),np.int32(N_abc),np.float32(f_pico),np.float32(d0),np.float32(dt),vp))

def updateZetaGPU(PsixFR, PsixFL, ZetaxFR, ZetaxFL,PsizFU, PsizFD, ZetazFU, ZetazFD, nx_abc, nz_abc, Uc, dx, dz, N_abc, f_pico, d0, dt, vp):
    total_pixels = nz_abc * nx_abc
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    updateZetaKernel((blocks_per_grid,),(threads_per_block,),(PsixFR, PsixFL,ZetaxFR, ZetaxFL, PsizFU, PsizFD,ZetazFU, ZetazFD,np.int32(nx_abc),np.int32(nz_abc),Uc,np.float32(dx),np.float32(dz),np.int32(N_abc),np.float32(f_pico),np.float32(d0),np.float32(dt),vp))

# CPML WaveEquation types
def updateWaveEquationCPMLGPU(Uf, Uc, vp, nx_abc, nz_abc, dz, dx, dt, PsixFR, PsixFL, PsizFU, PsizFD, ZetaxFR, ZetaxFL, ZetazFU, ZetazFD, N_abc):
    total_pixels = nz_abc * nx_abc
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    updateWaveEquationCPMLKernel((blocks_per_grid,),(threads_per_block,),(Uf,Uc,vp,np.int32(nx_abc),np.int32(nz_abc),np.float32(dz),np.float32(dx),np.float32(dt),PsixFR, PsixFL, PsizFU, PsizFD, ZetaxFR, ZetaxFL, ZetazFU, ZetazFD, np.int32(N_abc)))

def updateWaveEquationVTICPMLGPU(Uf, Uc, dt, dx, dz, vp, epsilon, delta,nx_abc, nz_abc, PsixFR, PsixFL, PsizFU, PsizFD,ZetaxFR, ZetaxFL, ZetazFU, ZetazFD, N_abc):
    total_pixels = nz_abc * nx_abc
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    updateWaveEquationVTICPMLKernel((blocks_per_grid,),(threads_per_block,),(Uf, Uc, vp, epsilon, delta,np.int32(nx_abc), np.int32(nz_abc),np.float32(dz), np.float32(dx), np.float32(dt),PsixFR, PsixFL, PsizFU, PsizFD,ZetaxFR, ZetaxFL, ZetazFU, ZetazFD,np.int32(N_abc)))

#Anisotropic Gradients Cuda
def calculateGradientVTICuda(current, adj, epsilon_partial, delta_partial, dx, dz, nx, nz,epsilon,delta):
    total_pixels = nz * nx
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    calculateGradientVTIKernel((blocks_per_grid,),(threads_per_block,),(current, adj, epsilon_partial, delta_partial, np.float32(dx), np.float32(dz), np.int32(nx), np.int32(nz),epsilon,delta))

def calculateGradientTTICuda(current, adj, epsilon_partial, delta_partial, theta_partial, dx, dz, nx, nz, epsilon, delta, theta):
    total_pixels = nz * nx
    threads_per_block = 256
    blocks_per_grid = (total_pixels + threads_per_block - 1) // threads_per_block

    calculateGradientTTIKernel((blocks_per_grid,),(threads_per_block,),(current, adj, epsilon_partial, delta_partial, theta_partial, np.float32(dx), np.float32(dz), np.int32(nx), np.int32(nz),epsilon,delta,theta))    


