import matplotlib.pyplot as plt
import numpy as np

from pyhelpers.store import save_fig

E = open('energy_FF.txt','r').read().strip().split('\n')    # kcal/mol
scaler = 1.602e-19 / 4.184 / 1e3 * 6.02e23  
E = [float(i)/scaler for i in E]   # eV
xlim = 1000
ylim = 350

plt.figure(0, figsize=(5,4))
plt.plot(E, color='#FAB558', linewidth=1.5)
plt.legend(['3T-FF energy'], fontsize=14)
plt.axis([0,xlim,0,ylim])
plt.plot([200,200], [0,ylim], 'k--', linewidth=0.5)
plt.plot([400,400], [0,ylim], 'k--', linewidth=0.5)
plt.plot([600,600], [0,ylim], 'k--', linewidth=0.5)
plt.plot([800,800], [0,ylim], 'k--', linewidth=0.5)

plt.xlabel('FF call',fontsize=18)
plt.ylabel(r'$E$ (eV)',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('Fig_Supp_Cycle_Transition.tiff', dpi=600)

# Create EMF vector graphic for Nat. Comm. editorial requirement
save_fig('Fig_Supp_Cycle_Transition_vFinal.svg', dpi=600, conv_svg_to_emf=True, verbose=True)

# Dump source data  for Nat. Comm. editorial requirement
out_E = np.zeros([len(E),2])
out_E[:,0] = np.arange(len(E))
out_E[:,1] = np.array(E)
np.savetxt("Fig_Supp_Cycle_Transition.csv", out_E, delimiter=",")

plt.show()
