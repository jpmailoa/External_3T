import pymol
from pymol import cmd
import sys, os
import glob, subprocess
import os, time
import ase
import ase.io as sio
import numpy as np
from swiss_util import swiss_func
from charmm2gromacs_util import chm2gmx_func
from GL_data import data
from process_molecule_utils import get_rotatable_bond, build_new_rotbond, cleanup_workspace, check_cache, store_cache
from process_molecule_from_lmp import convert_molecule as convert_molecule_lmp


def prelig4swiss(infile, outfile):
    outfile_0 = infile.replace('mol2', 'mol2')
    cmd.load(infile, 'MOL')
    #cmd.remove('hydrogens')
    #cmd.h_add('MOL')
    cmd.save(outfile_0, 'MOL', format='mol2')
    
    MOL_list = [x for x in open(outfile_0,'r')]
    idx = [i for i,x in enumerate(MOL_list) if x.startswith('@')]
    block = MOL_list[idx[1]+1:idx[2]]
    block = [x.split('\t') for x in block]

    block_new = []
    atom_count = {'C':1, 'N':1, 'O':1, 'S':1, 'P':1, 'F':1, 'Br':1, 'Cl':1, 'I': 1,   
                  'Li':1, 'Na':1, 'K':1, 'Mg':1, 'Al':1, 'Si':1,
                  'Ca':1, 'Cr':1, 'Mn':1, 'Fe':1, 'Co':1, 'Cu':1}
    for i in block:
        at = i[1].strip()
        if 'H' not in at:
            count = atom_count[at]
            atom_count[at]+=1
            at_new = at+str(count)
            at_new = at_new.rjust(4)
            block_new.append([i[0], at_new]+i[2:])
        else:
            block_new.append(i)

    block_new = ['\t'.join(x) for x in block_new]
    MOL_list_new = MOL_list[:idx[1]+1]+block_new+MOL_list[idx[2]:]
    f = open(outfile,'w')
    for i in MOL_list_new:
        f.write(i)
    f.close()
  
def extract_cgenff(lig_mol2, dl_txt, lig_ori, lig_zip, reply_txt):
    os.system(' '.join(['cp',lig_mol2,lig_ori]))
    dl_link = swiss_func(lig_mol2)
    time.sleep(5)
    start_time = time.time()
    while True:
        out = os.system(' '.join(['wget --no-check-certificate',dl_link,'2>',reply_txt]))
        if out == 0: break
        time.sleep(5)
        current_time = time.time()
        if current_time - start_time > 600:
            # 10 minutes without server success, we should terminate this molecule
            raise Exception('SwissParam server mol2 conversion fail')
    with open(reply_txt,'w') as f:
        f.write('y\n')
    os.system(' '.join(['unzip',lig_zip,'<',reply_txt]))
    return

def charmm2gmx(lig_itp, lig_par, lig_prm, lig_ff_folder, lig_bonded):
    chm2gmx_func(lig_itp,lig_par,lig_ff_folder)
    #os.system(' '.join(['python py3charmm2gromacs-pvm.py',lig_itp,lig_par,lig_ff_folder]))
    out_lines = []
    with open(os.path.join(lig_ff_folder,'forcefield.itp'),'r') as f:
        line = f.readline()
        while line:
            words = line.strip().split()
            if len(words)==0:
                out_lines.append(line)
            elif not words[0]=='#include':
                out_lines.append(line)
            line = f.readline()
        out_lines.append('#include '+lig_bonded+'\n')
    with open(lig_prm,'w') as f:
        for line in out_lines:
            f.write(line)
    os.system(' '.join(['mv', os.path.join(lig_ff_folder,'ffbonded.itp'), lig_bonded]))
    return

def build_gro(lig_pdb, lig_gro):
    os.system(' '.join(['gmx editconf -f',lig_pdb,'-o',lig_gro]))
    return

def convert_gromacs_lammps_ligand(lig_itp, lig_prm, lig_gro, lig_top):
    with open(lig_itp,'r') as f:
        content = f.read()
    with open(lig_top,'w') as f:
        f.write('; Include ligand parameters' + '\n' +\
                   '#include "' + lig_prm + '"\n\n')
        f.write(content)
        f.write('\n' +\
                   '[ molecules ]' +'\n' +\
                   '; Compound        #mols' + '\n' +\
                   'LIG                       1' + '\n')
    cwd = os.getcwd()
    lig_gro_full = os.path.join(cwd, lig_gro)
    lig_top_full = os.path.join(cwd, lig_top)
    # os.chdir('../utils/Convert_Gromacs_LAMMPS')
    os.chdir('../../utils/Convert_Gromacs_LAMMPS')
    os.system(' '.join([sys.executable+' Convert_Gromacs_LAMMPS.py',
                        lig_gro_full, lig_top_full, cwd]))
    os.chdir(cwd)
    return

def convert_molecule(mol_xyz, override=None):
    if '.lmp' == mol_xyz[-4:]:
        mol_data = convert_molecule_lmp(mol_xyz, override)
        return mol_data
    
    mol_data = check_cache(mol_xyz)
    
    if mol_data is None:
        temp_dir = 'workspace'
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        cwd = os.getcwd()
        os.system('cp '+mol_xyz+' '+temp_dir)
        os.chdir(temp_dir)

        full_mol_xyz_path = mol_xyz
        mol_xyz = os.path.split(mol_xyz)[-1]
        mol = sio.read(mol_xyz, format='xyz')
        temp_mol2, lig_mol2 = 'temp.mol2', 'LIG.mol2'
        os.system('obabel '+mol_xyz+' -O '+temp_mol2)
        prelig4swiss(temp_mol2, lig_mol2)
        get_rotatable_bond(lig_mol2, 'LIG.rotbond')

        extract_cgenff(lig_mol2, 'download.txt', 'LIG_ori.mol2', 'LIG.zip', 'reply.txt')

        charmm2gmx('LIG.itp', 'LIG.par', 'LIG.prm', 'LIG_ff', 'LIG_bonded.itp')

        build_gro('LIG.pdb', 'LIG.gro')

        convert_gromacs_lammps_ligand('LIG.itp', 'LIG.prm', 'LIG.gro', 'LIG.top')

        build_new_rotbond('LIG_ori.mol2', 'LIG.rotbond', 'LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')

        mol_data = data('LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')

        cleanup_workspace()

        os.chdir(cwd)
        store_cache(full_mol_xyz_path, mol_data)
        
    return mol_data
    
