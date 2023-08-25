import os
import json
from process_molecule import convert_molecule
from process_lattice import convert_lattice
from potential_model_3T import PotentialModel
import torch
import torch.optim as optim
import ase
import ase.io as io
import time
import numpy as np

def parse_config(config_json):
    blocks = json.load(open(config_json,'r'))
    if type(blocks) is dict:
        blocks = [blocks]
    for i, block in enumerate(blocks):
        if i==0:
            assert 'lattice_poscar' in block
            assert 'molecule_xyz' in block
            if not 'override' in block['lattice_poscar']:
                block['lattice_poscar']['override'] = None
            if not 'VASP_template' in block['lattice_poscar']:
                block['lattice_poscar']['VASP_template'] = None
            if not 'PWMAT_template' in block['lattice_poscar']:
                block['lattice_poscar']['PWMAT_template'] = None
            for moldict in block['molecule_xyz']:
                if not 'override' in moldict:
                    moldict['override'] = None
        else:
            if 'lattice_poscar' in block:
                assert block['lattice_poscar']['file'] == blocks[0]['lattice_poscar']['file']
                if not 'override' in block['lattice_poscar']:
                    block['lattice_poscar']['override'] = None
                if not 'VASP_template' in block['lattice_poscar']:
                    block['lattice_poscar']['VASP_template'] = None
                if not 'PWMAT_template' in block['lattice_poscar']:
                    block['lattice_poscar']['PWMAT_template'] = None
            if 'molecule_xyz' in block:
                for j, moldict in enumerate(block['molecule_xyz']):
                    init_moldict = blocks[0]['molecule_xyz'][j]
                    assert moldict['file'] == init_moldict['file']
                    assert moldict['count'] == init_moldict['count']
                    if not 'override' in moldict:
                        moldict['override'] = None
            if not 'lattice_poscar' in block:
                block['lattice_poscar'] = blocks[i-1]['lattice_poscar']
            if not 'molecule_xyz' in block:
                block['molecule_xyz'] = blocks[i-1]['molecule_xyz']
    return blocks

def pack_molecules(lattice_data, molecules_data):
    # First, determine bounding box
    cell = lattice_data.cell # 3 lattice vectors
    pos = lattice_data.atom_pos
    na = len(pos)
    cell_inv = np.linalg.inv(cell)
    proj_pos = np.matmul(pos, cell_inv)
    cell_abs = np.linalg.norm(cell, axis=1)
    proj_pos = proj_pos * cell_abs
    proj_dist = np.max(proj_pos, axis=0) - np.min(proj_pos, axis=0)
    empty_space = cell_abs - proj_dist
    big_idx = np.argmax(empty_space)
    box = np.array([ cell_abs[0], cell_abs[1], cell_abs[2] ])
    box[big_idx] = empty_space[big_idx] - 2.0
            
    temp_dir = 'workspace'
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    cwd = os.getcwd()
    os.chdir(temp_dir)
    os.system('rm -r *')

    mass_elem_dict = {1:'H', 7:'Li', 12:'C', 16:'O', 19:'F', 31:'P'}
    with open('pack_in.inp', 'w') as f:
        f.write('tolerance 2.0\n')
        f.write('filetype xyz\n')
        f.write('seed -1\n')
        f.write('output pack_out.xyz\n')

        count = 0
        for molecule_data in molecules_data:
            mol_elem = [mass_elem_dict[ round(molecule_data.atom_mass[ atom_type ]) ] for atom_type in molecule_data.atom_type]
            mol_pos = molecule_data.atom_pos
            in_xyz = 'mol_'+str(count)+'.xyz'
            with open(in_xyz,'w') as f2:
                f2.write(str(len(mol_pos))+'\n')
                for i in range(len(mol_pos)):
                    f2.write('\n' + ' '.join([str(j) for j in [mol_elem[i]] + mol_pos[i].tolist()]))

            f.write('structure '+in_xyz+'\n')
            f.write('  number 1\n')
            f.write('  inside box 0 0 0 ' + ' '.join([str(i) for i in box]) + '\n')
            f.write('end structure\n')
            count += 1

    os.system('packmol < pack_in.inp')

    with open('pack_out.xyz','r') as f:
        lines = f.read().strip().split('\n')[2:]
        pack_xyz = [[float(word) for word in line.split()[1:]] for line in lines]
        pack_xyz = np.array(pack_xyz)
        assert len(pack_xyz) == sum([len(molecule_data.atom_pos) for molecule_data in molecules_data])

    # Put back into lattice cell (automated + reshaping for orthorombus cell)
    real_box_ortho = np.array([ cell_abs[0], cell_abs[1], cell_abs[2] ])
    real_box_ortho[big_idx] = proj_dist[big_idx]
    real_diag = np.zeros(3)
    for i in range(3):
        real_diag += cell[i] / cell_abs[i] * real_box_ortho[i]
    proj_to_diag = np.sum(pos * real_diag, axis=1) / np.linalg.norm(real_diag)
    min_idx = np.argmin(proj_to_diag)
    max_idx = np.argmax(proj_to_diag)
    if not hasattr(lattice_data,'movable_group'):
        ref_idx = min_idx
        box_vec = 1.0
    else:
        temp = []
        for group in lattice_data.movable_group: temp += group
        if max_idx in temp:
            ref_idx = min_idx
            box_vec = 1.0
        elif min_idx in temp:
            ref_idx = max_idx
            box_vec = -1.0
        else:
            ref_idx = min_idx
            box_vec = 1.0

    pack_xyz *= box_vec
    pack_xyz = pack_xyz + proj_pos[ref_idx,:]
    pack_xyz[:,big_idx] += (proj_dist[big_idx] + 1.0) * box_vec
    pack_xyz = pack_xyz / cell_abs
    pack_xyz = np.matmul(pack_xyz, cell)

    base = 0
    for molecule_data in molecules_data:
        nm = len(molecule_data.atom_pos)
        molecule_data.atom_pos[:,:] = pack_xyz[base:base+nm,:]
        base += nm

    os.system('rm -r *')
    os.chdir(cwd)
                             
    return
    

def create_model(block, base_model=None):
    # Build lattice data
    lattice_data = convert_lattice(block['lattice_poscar']['file'],
                                   override = block['lattice_poscar']['override'],
                                   VASP_template = block['lattice_poscar']['VASP_template'],
                                   PWMAT_template = block['lattice_poscar']['PWMAT_template'])
    # Build molecules data
    molecules_data = []
    for moldict in block['molecule_xyz']:
        for i in range(moldict['count']):
            molecule_data = convert_molecule(moldict['file'],
                                             override = moldict['override'])
            molecules_data.append( molecule_data )
    pack_molecules(lattice_data, molecules_data)    # this will change molecules_data coordinates
    # Build model
    model = PotentialModel(lattice_data, molecules_data, block['mode'])
    if not (base_model is None):
        # Replace coordinates with those of base_model
        old_atom_pos = base_model.atom_pos.detach().cpu()
        if block['mode'] == 'FF':
            model.reset_positions(old_atom_pos, move_macro_group_into_pbc=True)
        else:
            model.reset_positions(old_atom_pos, move_macro_group_into_pbc=False)
    return model

def create_optimizers(model, block):
    # We directly modify the atom xyz coordinates. 
    # This is just a computation trick equivalent to modifying T_xyz, which saves a bit of compute/memory.
    theta_atom_translation = [param for param in model.movable_pos_list]
    optim_params = theta_atom_translation

    # Now we add the micro-groups' translation and rotation
    theta_micro_translation = model.translation_list
    theta_micro_rotation = model.rotation_list
    optim_params += [theta_micro_translation, theta_micro_rotation]
    
    special_rotation, macro_mode = model.special_rotation, model.macro_mode
    # Now we add sidechain micro-groups' rotatable bond axis rotation
    if special_rotation != None:
        theta_micro_axis_rotation = model.special_rotation_list
        optim_params += [theta_micro_axis_rotation]

    # Now we add macro-groups' translation and rotation
    if macro_mode != None:
        theta_macro_translation = model.macro_mode_translation_list
        theta_macro_rotation = model.macro_mode_rotation_list
        optim_params += [theta_macro_translation, theta_macro_rotation]

    optimizer = optim.Adam( optim_params , 3e-2, #1e-2,
                           weight_decay=0)
    optimizers = [ optimizer ]
    return optimizers

def print_log(message, log_file='default.log'):
    with open(log_file,'a') as f:
        f.write( str(message)+'\n')
    return

def out_file_list(out_tag):
    out_xyz = out_tag+'.xyz'
    out_outE = out_tag+'_outE.txt'
    return [out_xyz, out_outE]

def run_model(model_3T, optimizers, block):

    n_epoch = block['n_epoch']
    out_tag =block['out_tag']
    print_freq = block['print_freq']
    schedulers = None

    # determine atom elements for printing convenience later on
    mass_elem_dict = {1:'H', 7:'Li', 9:'Be', 11:'B', 12:'C', 14:'N', 16:'O', 19:'F',
                    23:'Na', 24:'Mg', 27:'Al', 28:'Si', 31:'P', 32:'S', 35:'Cl',
                    39:'K', 40:'Ca', 70:'Ga', 73:'Ge', 75:'As', 79:'Se', 80:'Br',
                    85:'Rb', 88:'Sr', 115:'In', 119:'Sn', 122:'Sb', 128:'Te', 127:'I', 207:'Pb'} # this is rounded mass to elem format
    atom_type = model_3T.atom_type.cpu().detach().numpy().astype(int) # this is already in 0 to n_type-1 format
    temp = model_3T.atom_mass.detach().cpu().numpy().astype(float)
    type_elem_dict = {}
    for i in range(temp.shape[0]):
        type_elem_dict[ i ] = mass_elem_dict[ round(temp[i]) ]
    del temp
    atom_elem = [ type_elem_dict[i] for i in atom_type ]

    out_xyz, out_outE = out_file_list(out_tag)

    loss_hist = np.zeros(n_epoch)
    out_hist = np.zeros(n_epoch)
    na = model_3T.atom_pos.shape[0]
    xyz_hist = np.zeros([n_epoch+1,na,3])
    xyz_hist[0,:,:] = model_3T.atom_pos.detach().cpu().numpy()

    start = time.time()
    for epoch in range(n_epoch):
        outp_E, outp_C = model_3T()
        loss = outp_C
    
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model_3T.parameters(), 1e11)
        for optimizer in optimizers:
            optimizer.step()
        if schedulers:
            for scheduler in schedulers:
                scheduler.step()
        
        loss_hist[epoch] = loss.detach().cpu().numpy()
        out_hist[epoch] = outp_E
        xyz_hist[epoch+1,:,:] = model_3T.atom_pos.detach().cpu().numpy()

        delta = time.time() - start
        if epoch % print_freq == 0: print_log('Step:'+str(epoch)+'\tTime:'+str(delta))

        atoms = ase.Atoms(symbols=atom_elem, positions=xyz_hist[epoch+1])
        if epoch == 0:
            with open(out_outE,'w') as f: f.write(str(out_hist[epoch])+'\n')
            io.write(out_xyz, atoms, format='xyz', append=False)
        else:
            with open(out_outE,'a') as f: f.write(str(out_hist[epoch])+'\n')
            io.write(out_xyz, atoms, format='xyz', append=True)
            
    # Clear the gradient after finishing the minimization
    for optimizer in optimizers:
        optimizer.zero_grad()

##    with open(out_outE,'w') as f:
##        for i in range(out_hist.shape[0]):
##            f.write(str(out_hist[i])+'\n')

    # We only add printout of the last xyz
##    atoms = ase.Atoms(symbols=atom_elem, positions=xyz_hist[n_epoch])
##    io.write(out_xyz, atoms, format='xyz', append=False)

    # Use this section if we want to print out the entire minimization trajectory instead
##    for i in range(xyz_hist.shape[0]):
##        atoms = ase.Atoms(symbols=atom_elem, positions=xyz_hist[i])
##        if i == 0: io.write(out_xyz, atoms, format='xyz', append=False)
##        else: io.write(out_xyz, atoms, format='xyz', append=True)

    return

#blocks = parse_config('configs/example_Li110_EC_DMC_VC_v1.json')
#model = None
#for block in blocks:
#    model = create_model(block, base_model=model)
#    optimizers = create_optimizers(model, block)
#    run_model(model, optimizers, block)
