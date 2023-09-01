import os
import matplotlib.pyplot as plt

def load_vasp_outcar(outcar):
    energies, times = [], []
    lines = open(outcar,'r').readlines()
    for line in lines:
        if ('energy' in line) and ('without' in line) and ('entropy=' in line):
            energy = float(line.strip().split()[3])     # eV
        if ('LOOP+:' in line) and ('cpu' in line) and ('time' in line):
            time = float(line.strip().split()[-1])      # s
            energies.append(energy)
            if len(times)==0:
                times.append(time)
            else:
                times.append(time + times[-1])
        if ('Elapsed time (sec)' in line):
            real_time = float(line.strip().split()[-1])
            last_time = times[-1]
            times = [time*real_time/last_time for time in times]
    return energies, times

def load_3T(dirname):
    energies, times = [], []
    lines = open( os.path.join(dirname,'default.log'),'r').readlines() 
    for line in lines:
        if 'Step:0' in line: times = []
        time = float( line.strip().split()[-1].split(':')[-1] )
        times.append(time)
    lines = open( os.path.join(dirname,'VASP_step1_outE.txt'),'r').readlines() 
    for line in lines:
        energy = float( line.strip() )      # kcal/mol
        scaler = 1.602e-19 / 4.184 / 1e3 * 6.02e23
        energy = energy / scaler          # eV
        energies.append(energy)
    return energies, times

#E_mol = -254.55292892   # eV
#E_surface = -1314.89        # eV

filenames = ['baseline_VASP_CG/mol/OUTCAR',
             'baseline_VASP_CG/surface/OUTCAR',
             'baseline_VASP_CG/0/OUTCAR',
             'baseline_VASP_CG/1/OUTCAR',
             'baseline_VASP_CG/2/OUTCAR']
energies_CG = []
times_CG = []

for i, filename in enumerate(filenames):
    energies, times = load_vasp_outcar(filename)
    if '/mol/OUTCAR' in filename:
        E_mol = energies[-1]
    elif '/surface/OUTCAR' in filename:
        E_surface = energies[-1]
    else:
        energies = [(E - E_mol - E_surface) for E in energies]
        energies_CG.append( energies )
        times_CG.append( times )

dirnames = ['3T/0x6c127cb6be9ea69a',
            '3T/0x32372cc9bcb7a382',
            '3T/0x212696dc24adfacf']
energies_3T = []
times_3T = []

for dirname in dirnames:
    energies, times = load_3T(dirname)
    energies = [(E - E_mol - E_surface) for E in energies]
    energies_3T.append( energies )
    times_3T.append( times )

E_baseline = energies_CG[2][-1]

#plt.figure(0, figsize=(5,6))
plt.figure(0, figsize=(5,5))
plt.plot(times_CG[0], energies_CG[0], '#04D288', linewidth=1.5)
plt.plot(times_3T[0], energies_3T[0], '#FAB558', linewidth=1.5)
plt.legend(['Standard Relaxation', '3T Relaxation'], fontsize=14)
for k in range(1,len(times_CG)):
    plt.plot(times_CG[k], energies_CG[k], '#04D288', linewidth=1.5)
for k in range(1,len(times_3T)):
    plt.plot(times_3T[k], energies_3T[k], '#FAB558', linewidth=1.5)
plt.plot([0,70000],[E_baseline,E_baseline],'k--')
plt.axis([0,70000,-7.5,-2.0])
for k in range(len(times_CG)):
    plt.scatter([times_CG[k][-1]],[energies_CG[k][-1]], color='#026C45', s=40)
for k in range(len(times_3T)):
    plt.scatter([times_3T[k][-1]],[energies_3T[k][-1]], color='#B06505', s=40)
plt.xlabel('Time (s)',fontsize=18)
plt.ylabel(r'$E$ (eV)',fontsize=18)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

#plt.figure(1, figsize=(5,6))
plt.figure(1, figsize=(5,5))
plt.plot(energies_CG[0], '#04D288', linewidth=1.5)
plt.plot(energies_3T[0], '#FAB558', linewidth=1.5)
plt.legend(['Standard Relaxation', '3T Relaxation'], fontsize=14)
for k in range(1,len(energies_CG)):
    plt.plot(energies_CG[k], '#04D288', linewidth=1.5)
for k in range(1,len(energies_3T)):
    plt.plot(energies_3T[k], '#FAB558', linewidth=1.5)
plt.plot([-10,600],[E_baseline,E_baseline],'k--')
plt.axis([-10.5,600,-7.5,-2.0])
for k in range(len(times_CG)):
    plt.scatter([len(times_CG[k])-1],[energies_CG[k][-1]], color='#026C45', s=40)
for k in range(len(times_3T)):
    plt.scatter([len(times_3T[k])-1],[energies_3T[k][-1]], color='#B06505', s=40)
plt.xlabel('DFT call',fontsize=18)
plt.ylabel(r'$E_{binding}$ (eV)',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('Fig_2e_3T_vs_ref.tiff', dpi=600)

plt.show()
