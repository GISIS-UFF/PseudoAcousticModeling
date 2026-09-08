from Viewdata import plotting 
from survey import parameters

pmt = parameters("../inputs/Parameters.json")
plt = plotting(pmt)

plt.viewModel(f"../outputs/estimated_models/fwi_vp_acoustic_Nx301_Nz301_itr1_freq10.0.bin")
# plt.viewHistory()
# plt.viewSnapshot("VTIforward_shot_1_Nx301_Nz301_Nt4001_frame_600.bin","/home/juanmarques/workspace/PseudoAcousticModeling/inputs/models/layer2vp_Nz301_Nx301.bin")
# plt.movieSnapshot(f"acousticforward_shot_10_Nx301_Nz301_Nt5178_frame", f"/home/processamento/PseudoAcousticModeling/inputs/models/layer2vp_Nz301_Nx301.bin",backward=False,interval=200,savegif=True)
# plt.viewSeismogram(f"/home/juanmarques/workspace/PseudoAcousticModeling/outputs/seismograms/seismogram_shot_5_Nt5001_Nrec50_fcut10.0.bin", perc=99)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewImage(f"../outputs/gradients/vp_gradient_fwi_iter_1_acoustic_Nx301_Nz301_freq10.0.bin",laplacian=True,perc=99)

# plt.plotImageTrace(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin", f"../inputs/layer2vp_Nz{pmt.nz}_Nx{pmt.nx}.bin", laplacian = True, ix=None, perc=99)
# plt.movieSnapshot(f"VTIbackward_shot_10_Nx301_Nz301_Nt5178_frame", f"/home/processamento/PseudoAcousticModeling/inputs/models/layer2vp_Nz301_Nx301.bin",backward=True,interval=200,savegif=True)

import numpy as np
import matplotlib.pyplot as plt

model_smooth = np.fromfile("/home/juanmarques/workspace/PseudoAcousticModeling/inputs/models/initial_crosstalk_vp_Nz301_Nx301.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)
model_ref = np.fromfile("/home/juanmarques/workspace/PseudoAcousticModeling/inputs/models/crosstalk_vp_Nz301_Nx301.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)
model_fwi = np.fromfile("../outputs/estimated_models/fwi_vp_acoustic_Nx301_Nz301_itr1_freq10.0.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)

D = np.linspace(0, pmt.nz * pmt.dz, pmt.nz, endpoint = False)
plt.figure()
plt.plot(model_ref[:,150],D,label = "ref")
plt.plot(model_smooth[:,150],D,label = "smooth")
plt.plot(model_fwi[:,150], D, label = "fwi")
plt.ylim(D[-1],0)
plt.legend()
plt.show()

