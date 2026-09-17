from Viewdata import plotting 
from survey import parameters

pmt = parameters("../inputs/Parameters.json")
plt = plotting(pmt)

plt.viewModel(f"/home/juanmarques/workspace/PseudoAcousticModeling/inputs/models/layer2epsilon_Nz301_Nx301.bin")
# plt.viewHistory()
# plt.viewSnapshot("VTIforward_shot_1_Nx301_Nz301_Nt4001_frame_600.bin","/home/juanmarques/workspace/PseudoAcousticModeling/inputs/models/layer2vp_Nz301_Nx301.bin")
# plt.movieSnapshot(f"VTIforward_shot_15_Nx301_Nz301_Nt4178_frame", f"/home/juanmarques/workspace/PseudoAcousticModeling/inputs/models/diffractorvp_Nz301_Nx301.bin",backward=False,interval=100,savegif=False)
plt.viewSeismogram(f"../outputs/seismograms/seismogram_shot_15_Nt4001_Nrec170_fcut20.0.bin", perc=90)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewImage(f"../outputs/gradients/delta_gradient_fwi_iter_1_VTI_Nx301_Nz301_freq20.0.bin",laplacian=True,perc=99)
# plt.plotImageTrace(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin", f"../inputs/layer2vp_Nz{pmt.nz}_Nx{pmt.nx}.bin", laplacian = True, ix=None, perc=99)
# plt.movieSnapshot(f"VTIbackward_shot_10_Nx301_Nz301_Nt5178_frame", f"/home/processamento/PseudoAcousticModeling/inputs/models/layer2vp_Nz301_Nx301.bin",backward=True,interval=200,savegif=True)

# import numpy as np
# import matplotlib.pyplot as plt

# model_smooth = np.fromfile("../inputs/models/fwi_vp_smooth_acoustic_Nx681_Nz141.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)
# model_ref = np.fromfile("/home/juanmarques/workspace/PseudoAcousticModeling/inputs/models/vp_marmousi-ii_shape_(2801, 13601)_dh25m_Nz141_Nx681.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)
# model_fwi = np.fromfile("../outputs/estimated_models/fwi_vp_acoustic_Nx681_Nz141_itr25_freq10.0.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)

# D = np.linspace(0, pmt.nz * pmt.dz, pmt.nz, endpoint = False)
# plt.figure()
# plt.plot(model_ref[:,340],D,label = "ref")
# plt.plot(model_smooth[:,340],D,label = "smooth")
# plt.plot(model_fwi[:,340], D, label = "fwi")
# plt.ylim(D[-1],0)
# plt.legend()
# plt.show()

