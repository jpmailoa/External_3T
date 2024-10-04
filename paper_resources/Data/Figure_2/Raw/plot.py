import os
import matplotlib.pyplot as plt
import numpy as np

from pyhelpers.store import save_fig

def load_vasp_outcar(outcar):
    energies = []
    lines = open(outcar,'r').readlines()
    for line in lines:
        if ('energy' in line) and ('without' in line) and ('entropy=' in line):
            energy = float(line.strip().split()[3])     # eV
        if ('LOOP+:' in line) and ('cpu' in line) and ('time' in line):
            energies.append(energy)
    return energies

def load_3T(dirname):
    energies = []
    lines = open( os.path.join(dirname,'VASP_step1_outE.txt'),'r').readlines() 
    for line in lines:
        energy = float( line.strip() )      # kcal/mol
        scaler = 1.602e-19 / 4.184 / 1e3 * 6.02e23
        energy = energy / scaler          # eV
        energies.append(energy)
    return energies

#E_mol = -254.55292892   # eV
#E_surface = -1314.89        # eV

filenames = ['VASP/mol/OUTCAR',
             'VASP/surface/OUTCAR',
             'VASP/0/OUTCAR',
             'VASP/1/OUTCAR',
             'VASP/2/OUTCAR']
energies_CG = []

for i, filename in enumerate(filenames):
    energies = load_vasp_outcar(filename)
    if '/mol/OUTCAR' in filename:
        E_mol = energies[-1]
    elif '/surface/OUTCAR' in filename:
        E_surface = energies[-1]
    else:
        energies = [(E - E_mol - E_surface) for E in energies]
        energies_CG.append( energies )

filenames = ['CREST_VASP/try0/OUTCAR',
             'CREST_VASP/try1/OUTCAR',
             'CREST_VASP/try2/OUTCAR']
energies_Crest = []

for i, filename in enumerate(filenames):
    energies = load_vasp_outcar(filename)
    energies = [(E - E_mol - E_surface) for E in energies]
    energies_Crest.append( energies )

filenames = ['FF_VASP/0/OUTCAR',
             'FF_VASP/1/OUTCAR',
             'FF_VASP/2/OUTCAR']
energies_postFF = []

for i, filename in enumerate(filenames):
    energies = load_vasp_outcar(filename)
    energies = [(E - E_mol - E_surface) for E in energies]
    energies_postFF.append( energies )

dirnames = ['3T-FF_3T-VASP/0x6c127cb6be9ea69a',
            '3T-FF_3T-VASP/0x32372cc9bcb7a382',
            '3T-FF_3T-VASP/0x212696dc24adfacf']
energies_3T = []

for dirname in dirnames:
    energies = load_3T(dirname)
    energies = [(E - E_mol - E_surface) for E in energies]
    energies_3T.append( energies )

E_baseline = energies_CG[2][-1]

plt.figure(1, figsize=(5.2,5.5))
plt.plot(energies_CG[0], '#04D288', linewidth=1.5)
plt.plot(energies_Crest[0], '#A0B0F2', linewidth=1.5)
plt.plot(energies_postFF[0], '#DCAFFF', linewidth=1.5)
plt.plot(energies_3T[0], '#FAB558', linewidth=1.5)
plt.legend(['VASP', 'CREST + VASP', 'FF + VASP', '3T-FF + 3T-VASP'], fontsize=14)
for k in range(1,len(energies_CG)):
    plt.plot(energies_CG[k], '#04D288', linewidth=1.5)
for k in range(1,len(energies_Crest)):
    plt.plot(energies_Crest[k], '#A0B0F2', linewidth=1.5)
for k in range(1,len(energies_postFF)):
    plt.plot(energies_postFF[k], '#DCAFFF', linewidth=1.5)
for k in range(1,len(energies_3T)):
    plt.plot(energies_3T[k], '#FAB558', linewidth=1.5)
plt.plot([-10,600],[E_baseline,E_baseline],'k--')
#plt.axis([-10.5,600,-7.5,-2.0])
plt.axis([-10.5,600,-8.0,-2.0])
for k in range(len(energies_CG)):
    plt.scatter([len(energies_CG[k])-1],[energies_CG[k][-1]], color='#026C45', s=40)
for k in range(len(energies_Crest)):
    plt.scatter([len(energies_Crest[k])-1],[energies_Crest[k][-1]], color='#3925F7', s=40)
for k in range(len(energies_postFF)):
    plt.scatter([len(energies_postFF[k])-1],[energies_postFF[k][-1]], color='#8200E8', s=40)
for k in range(len(energies_3T)):
    plt.scatter([len(energies_3T[k])-1],[energies_3T[k][-1]], color='#B06505', s=40)
plt.xlabel('DFT call',fontsize=18)
plt.ylabel(r'$E_{\mathrm{binding}}$ (eV)',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('Fig_2f_vFinal.tiff', dpi=600)
plt.savefig('Fig_2f_vFinal.pdf', dpi=600)

# Create EMF vector graphic for Nat. Comm. editorial requirement
save_fig('Fig_2f_vFinal.svg', dpi=600, conv_svg_to_emf=True, verbose=True)

# Dump source data  for Nat. Comm. editorial requirement
for i in range(3):
    temp = energies_CG[i]
    out = np.vstack([np.arange(len(temp)),np.array(temp)]).T
    np.savetxt('Fig_2f_E_VASP_'+str(i)+'.csv', out, delimiter=",")
    temp = energies_Crest[i]
    out = np.vstack([np.arange(len(temp)),np.array(temp)]).T
    np.savetxt('Fig_2f_E_CREST_VASP_'+str(i)+'.csv', out, delimiter=",")
    temp = energies_postFF[i]
    out = np.vstack([np.arange(len(temp)),np.array(temp)]).T
    np.savetxt('Fig_2f_E_FF_VASP_'+str(i)+'.csv', out, delimiter=",")
    temp = energies_3T[i]
    out = np.vstack([np.arange(len(temp)),np.array(temp)]).T
    np.savetxt('Fig_2f_E_3T-FF_3T-VASP_'+str(i)+'.csv', out, delimiter=",")

plt.show()
