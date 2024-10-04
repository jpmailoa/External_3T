import ase
import ase.io as sio
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl

from pyhelpers.store import save_fig


E_3T_FF_ASE = pd.read_csv('3T-FF-ASE/3T_ASE_0_outE.txt', header=None, names=['Energy'], sep='\s+')
E_3T_ASE = pd.read_csv('3T-ASE/3T_ASE_only_0_outE.txt', header=None, names=['Energy'], sep='\s+')
E_sella = pd.read_csv('Sella/sella.log', sep='\s+')

with plt.style.context(['science', 'no-latex']):
    mpl.rc('font', family='times new roman')
    fig, ax = plt.subplots(figsize=(7, 5))
    mpl.rcParams['axes.labelpad'] = -5 # default is 5

    ax.plot(E_sella.Step[:-1], E_sella.Energy[:-1]*1.602e-19 / 4.184 / 1e3 * 6.02e23, c='r', label='Sella-NWChem', linewidth=1.5)
    ax.plot(E_3T_ASE.index, E_3T_ASE.Energy, c='g', label='3T-NWChem', linewidth=1.5)
    ax.plot(E_3T_FF_ASE.index, E_3T_FF_ASE.Energy, c='b', label='3T-FF + 3T-NWChem', linewidth=1.5)
    plt.xlabel('NWChem call',fontsize=18)
    plt.ylabel(r'Energy (kcal mol$^{\mathrm{-1}}$)',fontsize=18)
    ax.legend(fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    ax.set_xlim((0, 50))
    plt.savefig('Fig_Supp_Sella_3T_NWChem.tiff', dpi=600)

    # Create EMF vector graphic for Nat. Comm. editorial requirement
    save_fig('Fig_Supp_Sella_3T_NWChem_vFinal.svg', dpi=600, conv_svg_to_emf=True, verbose=True)

    # Dump source data  for Nat. Comm. editorial requirement
    out_E_Sella = np.zeros([len(E_sella.Step)-1, 2])
    out_E_Sella[:,0] = E_sella.Step[:-1]
    out_E_Sella[:,1] = E_sella.Energy[:-1]*1.602e-19 / 4.184 / 1e3 * 6.02e23
    np.savetxt("E_Sella-NWChem.csv", out_E_Sella, delimiter=",")

    out_E_3T_ASE = np.zeros([len(E_3T_ASE.index), 2])
    out_E_3T_ASE[:,0] = E_3T_ASE.index
    out_E_3T_ASE[:,1] = E_3T_ASE.Energy
    np.savetxt("E_3T-NWChem.csv", out_E_3T_ASE, delimiter=",")

    out_E_3T_FF_ASE = np.zeros([len(E_3T_ASE.index), 2])
    out_E_3T_FF_ASE[:,0] = E_3T_FF_ASE.index
    out_E_3T_FF_ASE[:,1] = E_3T_FF_ASE.Energy
    np.savetxt("E_3T-FF-NWChem.csv", out_E_3T_FF_ASE, delimiter=",")
    

    plt.show()