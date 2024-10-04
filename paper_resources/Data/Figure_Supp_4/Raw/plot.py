import matplotlib.pyplot as plt
import numpy as np
import ase.io as sio

from pyhelpers.store import save_fig

#E_init = -1577.42955322
#E_final = -1577.19410478
E = np.loadtxt('pre_TS_post_merged_outE.txt')
ts_idx = 259
NEB_plot_idx = [i-ts_idx for i in range(len(E))]
atoms = sio.read('pre_TS_post_merged_OUTCAR.xyz', format='extxyz', index=':')
atoms_relax_init = sio.read('pre_NEB_relaxations/pre_OUTCAR.xyz', format='extxyz', index=':') #'-1')
atoms_relax_final = sio.read('pre_NEB_relaxations/post_OUTCAR.xyz', format='extxyz', index=':') #'-1')
E_relax_init = np.loadtxt('pre_NEB_relaxations/pre_out_E.txt') #[-1]
E_relax_final = np.loadtxt('pre_NEB_relaxations/post_out_E.txt') #[-1]
gap = 100
relax_init_idx = np.array([i-ts_idx-1-len(E_relax_init)-gap for i in range(len(E_relax_init))])
relax_final_idx = np.array([-i+len(E)-ts_idx+1+len(E_relax_final)+gap for i in range(len(E_relax_final))])
N_idx = 228
cation_N_z = np.array([atoms[i].positions[N_idx,2] for i in range(len(atoms))])
#cation_N_z_relax_init = atoms_relax_init.positions[N_idx,2]
#cation_N_z_relax_final = atoms_relax_final.positions[N_idx,2]
cation_N_z_relax_init = np.array([atoms_relax_init[i].positions[N_idx,2] for i in range(len(atoms_relax_init))])
cation_N_z_relax_final = np.array([atoms_relax_final[i].positions[N_idx,2] for i in range(len(atoms_relax_final))])
C1_idx = 180
I1_idx = 236
CI1_d = np.array([np.linalg.norm(atoms[i].positions[C1_idx]-atoms[i].positions[I1_idx]) for i in range(len(atoms))])
#CI1_d_relax_init = np.linalg.norm(atoms_relax_init.positions[C1_idx]-atoms_relax_init.positions[I1_idx])
#CI1_d_relax_final = np.linalg.norm(atoms_relax_final.positions[C1_idx]-atoms_relax_final.positions[I1_idx])
CI1_d_relax_init = np.array([np.linalg.norm(atoms_relax_init[i].positions[C1_idx]-atoms_relax_init[i].positions[I1_idx]) for i in range(len(atoms_relax_init))])
CI1_d_relax_final = np.array([np.linalg.norm(atoms_relax_final[i].positions[C1_idx]-atoms_relax_final[i].positions[I1_idx]) for i in range(len(atoms_relax_final))])
C2_idx = 179
I2_idx = 255
CI2_d = np.array([np.linalg.norm(atoms[i].positions[C2_idx]-atoms[i].positions[I2_idx]) for i in range(len(atoms))])
#CI2_d_relax_init = np.linalg.norm(atoms_relax_init.positions[C2_idx]-atoms_relax_init.positions[I2_idx])
#CI2_d_relax_final = np.linalg.norm(atoms_relax_final.positions[C2_idx]-atoms_relax_final.positions[I2_idx])
CI2_d_relax_init = np.array([np.linalg.norm(atoms_relax_init[i].positions[C2_idx]-atoms_relax_init[i].positions[I2_idx]) for i in range(len(atoms_relax_init))])
CI2_d_relax_final = np.array([np.linalg.norm(atoms_relax_final[i].positions[C2_idx]-atoms_relax_final[i].positions[I2_idx]) for i in range(len(atoms_relax_final))])


plt.figure(0, figsize=(7,12))

ax = [plt.subplot(4,1,i+1) for i in range(4)]

#ax[0].sharex(ax[3])
ax[0].plot(NEB_plot_idx, E, linewidth=2)
ax[0].plot(relax_init_idx,E_relax_init, linewidth=2,linestyle='--',c='m')
ax[0].plot(relax_final_idx,E_relax_final, linewidth=2,linestyle='--',c='brown')
ax[0].scatter([0],[E[ts_idx]],c='r')
ax[0].scatter([-ts_idx],[E[0]],c='k')
ax[0].scatter([len(E)-1-ts_idx],[E[len(E)-1]],c='g')
ax[0].set_xticklabels([])
ax[0].yaxis.set_tick_params(labelsize=11)
ax[0].set_ylabel(r'$E$ (eV)', fontsize=14)
ax[0].set_ylim([-1578,-1576])
ax[0].scatter(relax_init_idx[-1],E_relax_init[-1] ,c='m')
ax[0].scatter(relax_final_idx[-1],E_relax_final[-1], c='brown')
#ax[0].plot([-220,-180],[E_relax_init[-1],E_relax_init[-1]],'k')
#ax[0].plot([-220,-180],[E[ts_idx],E[ts_idx]],'k')
#ax[0].arrow(-600,-1576.5,200,-0.4, color='m',head_width=0.1,head_length=10)

# Dump source data  for Nat. Comm. editorial requirement
out = np.vstack([np.array(NEB_plot_idx),E]).T
np.savetxt('Fig_Supp_4a_E_NEB.csv', out, delimiter=",")
out = np.vstack([relax_init_idx,E_relax_init]).T
np.savetxt('Fig_Supp_4a_E_relax_init.csv', out, delimiter=",")
out = np.vstack([relax_final_idx,E_relax_final]).T
np.savetxt('Fig_Supp_4a_E_relax_final.csv', out, delimiter=",")

#ax[1].sharex(ax[3])
ax[1].plot(NEB_plot_idx, CI1_d, linewidth=2)
ax[1].plot(relax_init_idx,CI1_d_relax_init, linewidth=2,linestyle='--',c='m')
ax[1].plot(relax_final_idx,CI1_d_relax_final, linewidth=2,linestyle='--',c='brown')
ax[1].scatter([0],[CI1_d[ts_idx]],c='r')
ax[1].scatter([-ts_idx],[CI1_d[0]],c='k')
ax[1].scatter([len(E)-1-ts_idx],[CI1_d[len(E)-1]],c='g')
ax[1].set_xticklabels([])
ax[1].yaxis.set_tick_params(labelsize=11)
ax[1].set_ylabel(r'$r_{\mathrm{C-I,1}}$ ($\AA$)', fontsize=14)
ax[1].scatter(relax_init_idx[-1],CI1_d_relax_init[-1], c='m')
ax[1].scatter(relax_final_idx[-1],CI1_d_relax_final[-1], c='brown')

# Dump source data  for Nat. Comm. editorial requirement
out = np.vstack([np.array(NEB_plot_idx),CI1_d]).T
np.savetxt('Fig_Supp_4b_CI1_d_NEB.csv', out, delimiter=",")
out = np.vstack([relax_init_idx,CI1_d_relax_init]).T
np.savetxt('Fig_Supp_4b_CI1_d_relax_init.csv', out, delimiter=",")
out = np.vstack([relax_final_idx,CI1_d_relax_final]).T
np.savetxt('Fig_Supp_4b_CI1_d_relax_final.csv', out, delimiter=",")

#ax[2].sharex(ax[3])
ax[2].plot(NEB_plot_idx, CI2_d, linewidth=2)
ax[2].plot(relax_init_idx,CI2_d_relax_init, linewidth=2,linestyle='--',c='m')
ax[2].plot(relax_final_idx,CI2_d_relax_final, linewidth=2,linestyle='--',c='brown')
ax[2].scatter([0],[CI2_d[ts_idx]],c='r')
ax[2].scatter([-ts_idx],[CI2_d[0]],c='k')
ax[2].scatter([len(E)-1-ts_idx],[CI2_d[len(E)-1]],c='g')
ax[2].set_xticklabels([])
ax[2].yaxis.set_tick_params(labelsize=11)
ax[2].set_ylabel(r'$r_{\mathrm{C-I,2}}$ ($\AA$)', fontsize=14)
ax[2].scatter(relax_init_idx[-1],CI2_d_relax_init[-1], c='m')
ax[2].scatter(relax_final_idx[-1],CI2_d_relax_final[-1], c='brown')

# Dump source data  for Nat. Comm. editorial requirement
out = np.vstack([np.array(NEB_plot_idx),CI2_d]).T
np.savetxt('Fig_Supp_4c_CI2_d_NEB.csv', out, delimiter=",")
out = np.vstack([relax_init_idx,CI2_d_relax_init]).T
np.savetxt('Fig_Supp_4c_CI2_d_relax_init.csv', out, delimiter=",")
out = np.vstack([relax_final_idx,CI2_d_relax_final]).T
np.savetxt('Fig_Supp_4c_CI2_d_relax_final.csv', out, delimiter=",")

ax[3].plot(NEB_plot_idx, cation_N_z, linewidth=2)
ax[3].plot(relax_init_idx,cation_N_z_relax_init, linewidth=2,linestyle='--',c='m')
ax[3].plot(relax_final_idx,cation_N_z_relax_final, linewidth=2,linestyle='--',c='brown')
ax[3].scatter([0],[cation_N_z[ts_idx]],c='r')
ax[3].scatter([-ts_idx],[cation_N_z[0]],c='k')
ax[3].scatter([len(E)-1-ts_idx],[cation_N_z[len(E)-1]],c='g')
ax[3].legend(['Reaction Trajectory','Relaxed Pre-Snapshot','Relaxed Post-Snapshot','Transition State','TS-->Pre Relaxation','TS-->Post Relaxation'], loc='lower right')
ax[3].yaxis.set_tick_params(labelsize=11)
ax[3].set_ylabel(r'$z_{\mathrm{N,TE4PBA}}$ ($\AA$)', fontsize=14)
ax[3].set_xlabel('Step', fontsize=14)
#ax[3].set_xlim([-1200,1000])
ax[3].scatter(relax_init_idx[-1],cation_N_z_relax_init[-1], c='m')
ax[3].scatter(relax_final_idx[-1],cation_N_z_relax_final[-1], c='brown')

# Dump source data  for Nat. Comm. editorial requirement
out = np.vstack([np.array(NEB_plot_idx),cation_N_z]).T
np.savetxt('Fig_Supp_4d_cation_N_z_NEB.csv', out, delimiter=",")
out = np.vstack([relax_init_idx,cation_N_z_relax_init]).T
np.savetxt('Fig_Supp_4d_cation_N_z_relax_init.csv', out, delimiter=",")
out = np.vstack([relax_final_idx,cation_N_z_relax_final]).T
np.savetxt('Fig_Supp_4d_cation_N_z_relax_final.csv', out, delimiter=",")

for i in range(4):
    ax[i].set_xlim([-750,1000])

plt.subplots_adjust(wspace=0, hspace=0)

plt.tight_layout()
plt.savefig('Fig_NEB.tiff', dpi=600)

# Create EMF vector graphic for Nat. Comm. editorial requirement
save_fig('Fig_NEB_vFinal.svg', dpi=600, conv_svg_to_emf=True, verbose=True)

plt.show()