import torch
import os, glob
import ase
import ase.io as sio
import numpy as np

def calc_E_F_VASP(model):
    # Placeholder
    E_total = torch.zeros(1,).detach().cpu().numpy()
    F_atoms = torch.zeros(model.atom_pos.shape[0],3).detach().cpu().numpy()
    # Need INCAR, KPOINTS, POTCAR, POSCAR
    # Command: 'nohup mpirun -n 4 --allow-run-as-root ~/software/vasp.6.2.1/bin/vasp_std &'
    # Element sequence needs to follow POTCAR element sequence
    if not hasattr(model.lattice_data, 'VASP_template'):
        raise Exception('VASP template directory for the lattice-molecule system is not defined')
    VASP_template = model.lattice_data.VASP_template
    elems = elem_sequence( os.path.join(VASP_template,'POTCAR') )
    ase_obj, reorder_list = create_reordered_ase_obj(model, elems)
    workspace = 'workspace'
    copy_template(VASP_template, workspace)
    write_POSCAR(ase_obj, workspace)
    edit_INCAR(workspace)
    run_VASP(workspace)
    energy, forces = extract_E_F(workspace)
    forces = reorder_forces(forces, reorder_list)
    E_total = torch.Tensor( [energy] )
    F_atoms = torch.Tensor( forces )
    VASP_file_dir = 'VASP_files'
    store_VASP_files(workspace, VASP_file_dir)
    # we will comment out cleanup because we want to reuse the density in the next iteration
    #cleanup(workspace)
    return E_total, F_atoms

def elem_sequence(POTCAR_file):
    with open(POTCAR_file,'r') as f:
        POTCAR_content = f.read()
    blocks = POTCAR_content.split('End of Dataset')
    elems = []
    for block in blocks:
        block = block.strip()
        lines = block.split('\n')
        words = lines[0].strip().split()
        if len(words) > 1:
            elem = words[1]
            elems.append(elem)
    return elems

def create_reordered_ase_obj(model, elems):
    mass_dict = {1:'H', 7:'Li', 12:'C', 14:'N', 16:'O', 19:'F', 31:'P', 127:'I', 207:'Pb'}
    ori_atom_pos = model.atom_pos.detach().cpu().numpy()
    cell = model.cell.detach().cpu().numpy()
    ori_mass = [round(i) for i in model.atom_mass[ model.atom_type ].detach().cpu().numpy().flatten().tolist()]
    ori_atom_type = [mass_dict[i] for i in ori_mass]

    new_atom_pos = []
    new_atom_type = []
    reorder_list = []
    for elem in elems:
        for i, atom_type in enumerate(ori_atom_type):
            if (atom_type == elem) or (atom_type+'_sv' == elem):
              new_atom_type.append(atom_type)
              new_atom_pos.append( ori_atom_pos[i] )
              reorder_list.append( i ) ## this maps old order to new order
    new_atom_pos = np.array(new_atom_pos)
    
    ase_obj = ase.Atoms( new_atom_type, positions=new_atom_pos, cell=cell, pbc=[1,1,1] )

    return ase_obj, reorder_list

def write_POSCAR(ase_obj, workspace):
    filename = os.path.join(workspace,'POSCAR')
    sio.vasp.write_vasp(filename, ase_obj, wrap=True)
    content = open(filename,'r').read()
    content.replace('Li', 'Li_sv')
    open(filename,'w').write(content)
    return
    
def copy_template(VASP_template, workspace):
    os.system(' '.join(['cp', os.path.join(VASP_template,'INCAR'), os.path.join(workspace,'INCAR')]))
    os.system(' '.join(['cp', os.path.join(VASP_template,'POTCAR'), os.path.join(workspace,'POTCAR')]))
    os.system(' '.join(['cp', os.path.join(VASP_template,'KPOINTS'), os.path.join(workspace,'KPOINTS')]))
    return

def edit_INCAR(workspace):
    if os.path.isfile( os.path.join(workspace,'WAVECAR') ):
        incar = os.path.join(workspace,'INCAR')
        lines = open(incar,'r').readlines()
        for i, line in enumerate(lines):
            if 'ISTART' in line: lines[i] = '   ISTART  =  1\n'
            if 'ICHARG' in line: lines[i] = '   ICHARG  =  1\n'
        with open(incar,'w') as f:
            for line in lines:
                f.write( line )
    return

def run_VASP(workspace):
    cwd = os.getcwd()
    os.chdir(workspace)
    n_gpu = str( torch.cuda.device_count() )
    os.system('nohup mpirun -n '+n_gpu+' --allow-run-as-root ~/software/vasp.6.2.1/bin/vasp_std')
    # placeholder for non-VASP hardware
    #os.system('cp ../input/OUTCAR .')
    os.chdir(cwd)
    return

def extract_E_F(workspace):
    OUTCAR_file = os.path.join(workspace, 'OUTCAR')
    with open(OUTCAR_file, 'r') as f:
        OUTCAR_content = f.read()
    OUTCAR_content = OUTCAR_content.split('TOTAL-FORCE (eV/Angst)')[-1].strip()
    lines = OUTCAR_content.split('\n')
    for line in lines:
        if ('energy' in line) and ('without' in line) and ('entropy' in line):
            energy = float(line.strip().split()[3])
    OUTCAR_content = OUTCAR_content.split('total drift:')[0].strip()
    lines = OUTCAR_content.split('\n')[1:-1]
    forces = []
    for line in lines:
        words = line.strip().split()
        force = [float(i) for i in words[3:6]]
        forces.append( force )
    forces = np.array(forces)
    # Convert energy (eV) and forces (eV/A) to kcal/mol & kcal/mol/A
    scaler = 1.602e-19 / 4.184 / 1e3 * 6.02e23
    energy = energy * scaler
    forces = forces * scaler
    return energy, forces

def reorder_forces(forces, reorder_list):
    assert len(forces) == len(reorder_list)
    reordered_forces = np.zeros(forces.shape)
    for i, idx in enumerate(reorder_list):
        reordered_forces[idx] = forces[i]
    return reordered_forces

def store_VASP_files(workspace, VASP_file_dir):
    if not os.path.isdir(VASP_file_dir):
        os.mkdir(VASP_file_dir)
    finished_dirs = glob.glob( VASP_file_dir + '/step_*/' )
    VASP_file_dir = VASP_file_dir + '/step_' + str(len(finished_dirs)) +'/'
    os.mkdir(VASP_file_dir)
    os.system('cp ' + workspace + '/POSCAR ' + VASP_file_dir)
    os.system('cp ' + workspace + '/OUTCAR ' + VASP_file_dir)
    os.system('cp ' + workspace + '/vasprun.xml ' + VASP_file_dir)
    return
    
def cleanup(workspace):
    os.system('rm '+workspace+'/*')
    return
