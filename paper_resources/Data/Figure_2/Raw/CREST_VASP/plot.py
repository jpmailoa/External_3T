import ase
import ase.io as sio
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl

E_try0 = pd.read_csv('try0_outE.log', header=None, names=['Energy'], sep='\s+')
E_try1 = pd.read_csv('try1_outE.log', header=None, names=['Energy'], sep='\s+')
E_try2 = pd.read_csv('try2_outE.log', header=None, names=['Energy'], sep='\s+')
E_00 = -1577.42955322

with plt.style.context(['science', 'no-latex']):
    mpl.rc('font', family='times new roman')
    fig, ax = plt.subplots(figsize=(6, 4))
    mpl.rcParams['axes.labelpad'] = -5 # default is 5

    ax.plot(E_try0.index, E_try0.Energy-E_00, c='r', label='try0')
    ax.plot(E_try1.index, E_try1.Energy-E_00, c='g', label='try1')
    ax.plot(E_try2.index, E_try2.Energy-E_00, c='b', label='try2')
    ax.set(xlabel='Step')
    ax.set(ylabel='Energy (eV)')
    ax.legend()
    ax.autoscale(tight=True)
    #ax.set_xlim((0, 50))
    plt.savefig('Crest/Fix/Crest_CG.png', dpi=300)
    plt.show()