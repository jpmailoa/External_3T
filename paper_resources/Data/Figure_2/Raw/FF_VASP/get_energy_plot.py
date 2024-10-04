import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
import pandas as pd

E_0 = pd.read_csv('./0/outE.log', header=None, names=['Energy'], sep='\s+')
E_1 = pd.read_csv('./0/outE.log', header=None, names=['Energy'], sep='\s+')
E_2 = pd.read_csv('./2/outE.log', header=None, names=['Energy'], sep='\s+')

E_00 = 0

with plt.style.context(['science', 'no-latex']):
    mpl.rc('font', family='times new roman')
    fig, ax = plt.subplots(figsize=(6, 4))
    mpl.rcParams['axes.labelpad'] = -5 # default is 5

    ax.plot(E_0.index, E_0.Energy-E_00, c='r', label='0')
    ax.plot(E_1.index, E_1.Energy-E_00, c='b', label='1')
    ax.plot(E_2.index, E_2.Energy-E_00, c='g', label='2')
    ax.set(xlabel='Step')
    ax.set(ylabel='Energy (eV)')
    ax.legend()
    ax.autoscale(tight=True)
    #ax.set_xlim((0, 50))
    #plt.savefig('Crest/Fix/Crest_CG.png', dpi=300)
    plt.show()