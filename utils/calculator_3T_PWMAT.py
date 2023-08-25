import torch
import os
import ase
import ase.io as sio
import numpy as np

def calc_E_F_PWMAT(model):
    # Placeholder
    E_total = torch.zeros(1,).detach().cpu().numpy()
    F_atoms = torch.zeros(model.atom_pos.shape[0],3).detach().cpu().numpy()
    # Need IN.SOLVENT, etot.input, and corresponding *.UPF files
    # Command: 'nohup mpirun -np 4 /opt/bin/PWmat &> output'
    if not hasattr(model.lattice_data, 'PWMAT_template'):
        raise Exception('PWMAT template directory for the lattice-molecule system is not defined')
    PWMAT_template = model.lattice_data.PWMAT_template
    ase_obj = create_ase_obj(model)
    workspace = 'workspace'
    copy_template(PWMAT_template, workspace)
    write_struct_config(ase_obj, workspace)
    edit_etot_input(workspace)
    run_PWMAT(workspace)
    energy, forces = extract_E_F(workspace)
    E_total = torch.Tensor( [energy] )
    F_atoms = torch.Tensor( forces )
    # this cleanup comment will explicitly keep files useful for subsequent PWMAT runs
    cleanup(workspace)
    return E_total, F_atoms

def create_ase_obj(model):
    mass_dict = {1:'H', 7:'Li', 12:'C', 14:'N', 16:'O', 19:'F', 31:'P', 127:'I', 207:'Pb'}
    ori_atom_pos = model.atom_pos.detach().cpu().numpy()
    cell = model.cell.detach().cpu().numpy()
    ori_mass = [round(i) for i in model.atom_mass[ model.atom_type ].detach().cpu().numpy().flatten().tolist()]
    ori_atom_type = [mass_dict[i] for i in ori_mass]

    ase_obj = ase.Atoms( ori_atom_type, positions=ori_atom_pos, cell=cell, pbc=[True,True,True] )
    ase_obj.wrap()

    return ase_obj

def copy_template(PWMAT_template, workspace):
    if os.path.isfile( os.path.join(PWMAT_template,'IN.SOLVENT') ):
        os.system(' '.join(['cp', os.path.join(PWMAT_template,'IN.SOLVENT'), os.path.join(workspace,'IN.SOLVENT')]))
    os.system(' '.join(['cp', os.path.join(PWMAT_template,'etot.input'), os.path.join(workspace,'etot.input')]))
    os.system(' '.join(['cp', os.path.join(PWMAT_template,'*.UPF'), os.path.join(workspace,'')]))
    return

def write_struct_config(ase_obj, workspace):
    filename = os.path.join(workspace,'struct.config')
    cell, num, pos = ase_obj.cell, ase_obj.get_atomic_numbers(), ase_obj.positions
    pwmat_pos = np.matmul( pos, np.linalg.inv(cell) )
    lines = []
    lines.append( str(len(ase_obj)) )
    lines.append( 'LATTICE' )
    for i in range(3):
        lines.append( ' '.join([str(cell[i,j]) for j in range(3)]) )
    lines.append( 'POSITION' )
    for i in range(len(ase_obj)):
        lines.append( ' '.join([str(num[i])] + [str(pwmat_pos[i,j]) for j in range(3)] + ['1 1 1']) )
    with open(filename,'w') as f:
        f.write( '\n'.join(lines) )
    return

def edit_etot_input(workspace):
    WG, RHO, VR = False, False, False
    if os.path.isfile( os.path.join(workspace,'OUT.WG') ): WG = True
    if os.path.isfile( os.path.join(workspace,'OUT.RHO') ): RHO = True
    if os.path.isfile( os.path.join(workspace,'OUT.VR') ): VR = True
    etot_input = os.path.join(workspace,'etot.input')
    lines = open(etot_input,'r').readlines()
    for i, line in enumerate(lines):
        if ('IN.WG' in line) and WG:
            os.system('mv ' + os.path.join(workspace,'OUT.WG') + ' ' + os.path.join(workspace,'IN.WG'))
            lines[i] = 'IN.WG = T\n'
        if ('IN.RHO' in line) and RHO:
            os.system('mv ' + os.path.join(workspace,'OUT.RHO') + ' ' + os.path.join(workspace,'IN.RHO'))
            lines[i] = 'IN.RHO = T\n'
        if ('IN.VR' in line) and VR:
            os.system('mv ' + os.path.join(workspace,'OUT.VR') + ' ' + os.path.join(workspace,'IN.VR'))
            lines[i] = 'IN.VR = T\n'
    with open(etot_input,'w') as f:
        for line in lines:
            f.write( line )
    return

def run_PWMAT(workspace):
    cwd = os.getcwd()
    os.chdir(workspace)
    n_gpu = str( torch.cuda.device_count() )
    os.system('nohup mpirun -np '+n_gpu+' /opt/bin/PWmat &> output')
    # placeholder for non-PWMAT hardware
    #os.system('cp ../input/OUT.FORCE .')
    #os.system('cp ../input/REPORT .')
    os.chdir(cwd)
    return

def extract_E_F(workspace):
    REPORT_file = os.path.join(workspace, 'REPORT')
    forces_file = os.path.join(workspace, 'OUT.FORCE')
    lines = open(REPORT_file, 'r').readlines()
    success = False
    E_line = None
    for line in lines:
        if line.startswith(' E_tot(eV)'):
            E_line = line
        if line.startswith(' total computation time'):
            success = True
    if not success: raise Exception('PWMAT calculation failure')
    energy = float(E_line.strip().split('=')[1].strip().split()[0])     # energy (eV)
    lines = open(forces_file, 'r').readlines()
    forces = []
    #past_removal = False
    pre_removal = True
    #for line in lines:
    for line in lines[1:]:
        if '*' in line:
            pre_removal = False
        #if past_removal:
        if pre_removal:
            words = line.strip().split()
            if len(words) == 4:
                force = [float(i) for i in words[1:4]]
                forces.append( force )
        #else:
        #    if 'force after remove' in line:
        #        past_removal = True
    forces = np.array(forces)
    # Convert energy (eV) and forces (eV/A) to kcal/mol & kcal/mol/A
    scaler = 1.602e-19 / 4.184 / 1e3 * 6.02e23
    energy = energy * scaler
    forces = forces * scaler
    return energy, forces

def cleanup(workspace):
    keep_files = ['OUT.WG', 'OUT.RHO', 'OUT.VR', 'OUT.FERMI']
    dir_files = os.listdir(workspace)
    for dir_file in dir_files:
        if not dir_file in keep_files:
            os.system('rm ' + os.path.join(workspace, dir_file))
    return
