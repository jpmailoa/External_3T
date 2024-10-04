import os
import matplotlib.pyplot as plt
import numpy as np

from pyhelpers.store import save_fig

def load_vasp_E(folder):
    scaler = 1.0 / (1.602e-19 / 4.184 / 1e3 * 6.02e23)  # this will convert kcal/mol to eV
    files = [file for file in os.listdir(folder) if file.endswith('outE.txt') and not file.endswith('all_outE.txt')]
    files.sort()
    E = []
    for file in files:
        lines = open(os.path.join(folder,file),'r').read().strip().split('\n')
        energies = [float(line)*scaler for line in lines]
        E += energies
    return E

oxidation_id = ['0x16d19a92d8634cc6',
                '0x208a9db3b56886bf',
                '0x51f72a19e29afadd']
rxn_step_Sella = {'0x16d19a92d8634cc6': [23],  # 23: oEC-DMC reaction
                  '0x208a9db3b56886bf': [97, 101, 250],  # 97: CO3+VC reaction, 101: oEC+EC reaction, 250: oEC reaction
                  '0x51f72a19e29afadd': [20, 72]}  # 20: oEC reaction, 72: CO3+VC reaction
rxn_step_3T = {'0x16d19a92d8634cc6': [4, 71],  # 4: oEC_0 reaction (b), 71: CO3_1+VC_2 reaction (e)
               '0x208a9db3b56886bf': [16, 23, 107],  # 1?: reverse DMC oxidation?, 16: CO3_0+VC_0 reaction (e), 23: oEC_1 reaction (a), 107: oEC_0 reaction (a)
               '0x51f72a19e29afadd': [6, 27]}  # 6: oEC_0 reaction (b), 27: CO3_0+VC_2 reaction (e)?
plt.figure(1, figsize=(7,10))
ax = [plt.subplot(3,1,i+1) for i in range(3)]
for i, id in enumerate(oxidation_id):
    E_Sella = load_vasp_E('rxn_process/Oxidation/' + id)
    E_3T = load_vasp_E('rxn_process/Oxidation/' + id + '/3T_reference')
    dE = E_Sella[-1] - E_3T[-1]
    y_shift_1 = [2,-20,0]
    y_shift_2 = [5,-14,3]
    ax[i].plot(E_Sella, linewidth=2, color='tab:blue')
    ax[i].plot(E_3T, linewidth=2, color='tab:orange')
    if i == 0:
        ax[i].legend(['Sella-VASP','3T-VASP'], fontsize=14, loc='upper right')
    ax[i].scatter(rxn_step_Sella[id], [E_Sella[j] for j in rxn_step_Sella[id]], c='tab:blue')
    ax[i].scatter(rxn_step_3T[id], [E_3T[j] for j in rxn_step_3T[id]], c='tab:orange')
    ax[i].yaxis.set_tick_params(labelsize=11)
    ax[i].set_ylabel(r'$E$ (eV)', fontsize=14)
    ax[i].text(x=50+10, y=max(E_Sella)-y_shift_1[i], fontsize=11, s='ID = ' + id)
    ax[i].text(x=50+10, y=max(E_Sella)-y_shift_2[i], fontsize=11, s=r'$E_{\mathrm{Sella}} - E_{\mathrm{3T}} = $' + str(round(dE,2)) + r' $eV$')
    if i==len(oxidation_id)-1:
        ax[i].xaxis.set_tick_params(labelsize=11)
        ax[i].set_xlabel('DFT call', fontsize=14)
    else:
        ax[i].set_xticklabels([])
    ax[i].set_xlim([0,300])

    # Dump source data  for Nat. Comm. editorial requirement
    out_E_Sella = np.zeros([len(E_Sella), 2])
    out_E_Sella[:,0] = np.arange(len(E_Sella))
    out_E_Sella[:,1] = E_Sella
    np.savetxt('E_Sella-VASP_'+id+'.csv', out_E_Sella, delimiter=",")

    out_E_3T = np.zeros([len(E_3T), 2])
    out_E_3T[:,0] = np.arange(len(E_3T))
    out_E_3T[:,1] = E_3T
    np.savetxt('E_3T-VASP_'+id+'.csv', out_E_3T, delimiter=",")


plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig('Fig_Supp_Sella_3T_VASP_Oxidation.tiff', dpi=600)

# Create EMF vector graphic for Nat. Comm. editorial requirement
save_fig('Fig_Supp_Sella_3T_VASP_Oxidation_vFinal.svg', dpi=600, conv_svg_to_emf=True, verbose=True)

plt.show()