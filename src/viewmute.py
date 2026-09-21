from survey import parameters
import matplotlib.pyplot as plt
import numpy 
from numba import jit, prange

@jit(parallel=True)
def Mute(seismogram, shot, rec_x, rec_z, shot_x, shot_z, dt,shift,window,v0=1500): 
    result = np.zeros_like(seismogram)
    Nt = seismogram.shape[0]
    Nrec = seismogram.shape[1]  
    for rec in prange(Nrec):
        dist = np.sqrt((rec_z[rec] - shot_z[shot])**2 + (rec_x[rec] - shot_x[shot])**2)
        traveltimes = dist/v0 + shift
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

pmt = parameters("../inputs/Parameters.json")
shot_file = 5
shot_idx = shot_file - 1
shift = 0.1
window = 0.05
seismogramFile = (f"/home/processamento/PseudoAcousticModeling/outputs/seismograms/seismogram_shot_5_Nt4001_Nrec100_fcut20.0.bin")
seismogram = numpy.fromfile(seismogramFile,dtype=numpy.float32).reshape(pmt.nt_data, pmt.Nrec)
muted_seismogram = Mute(seismogram,shot_idx,pmt.rec_x,pmt.rec_z,pmt.shot_x,pmt.shot_z,pmt.dt,shift,window,v0=1500)

dist = numpy.sqrt((pmt.rec_z - pmt.shot_z[shot_idx])**2 + (pmt.rec_x - pmt.shot_x[shot_idx])**2)
traveltimes = dist / 1500  + pmt.tlag + shift
travel_idx = traveltimes/pmt.dt   

plt.figure()
plt.plot(seismogram[:, 25], label="seismogram")
plt.plot(muted_seismogram[:, 25], label="muted")
plt.legend()

plt.figure()
plt.imshow(muted_seismogram, aspect="auto", cmap="gray")
# plt.plot(numpy.arange(pmt.Nrec), travel_idx, 'r', linewidth=2, alpha = 0.7)
plt.colorbar()
plt.show()