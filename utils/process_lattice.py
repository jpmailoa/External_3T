import os
import ase.io as sio
from VL_data import data

def cleanup_workspace():
    files = os.listdir()
    for file in files:
        if os.path.isfile(file):
            os.system('rm '+file)
    return

def convert_lattice(lattice_poscar, override=None, VASP_template = None, PWMAT_template = None):
    temp_dir = 'workspace'
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    cwd = os.getcwd()
    os.system('cp '+lattice_poscar+' '+temp_dir)
    os.chdir(temp_dir)

    lattice_poscar = os.path.split(lattice_poscar)[-1]
    lattice = sio.read(lattice_poscar, format='vasp')
    #lattice.wrap()
    
    cleanup_workspace()
    os.chdir(cwd)

    lattice_data = data(lattice, override=override, VASP_template=VASP_template, PWMAT_template=PWMAT_template)    
    
    return lattice_data
