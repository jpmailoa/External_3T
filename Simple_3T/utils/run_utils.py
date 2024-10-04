from process_molecule import convert_molecule
from potential_model_3T import PotentialModel
import torch
import torch.optim as optim
import ase.io as io
import time
import numpy as np

def create_model(xyz_files, atoms, mode, base_model=None):
    # Build molecules data
    molecules_data = []
    for xyz_file in xyz_files:
        molecule_data = convert_molecule(xyz_file)
        molecules_data.append( molecule_data )
    # Build model
    model = PotentialModel(molecules_data, atoms, mode)
    if not (base_model is None):
        # Replace coordinates with those of base_model
        old_atom_pos = base_model.atom_pos.detach().cpu()
        model.reset_positions(old_atom_pos, move_macro_group_into_pbc=False)
    return model

def create_optimizers(model):
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

def run_model(model_3T, optimizers, n_epoch, out_tag='3T', print_freq=1):

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

        model_3T.atoms.positions = xyz_hist[epoch+1]
        if epoch == 0:
            with open(out_outE,'w') as f: f.write(str(out_hist[epoch])+'\n')
            io.write(out_xyz, model_3T.atoms, format='xyz', append=False)
        else:
            with open(out_outE,'a') as f: f.write(str(out_hist[epoch])+'\n')
            io.write(out_xyz, model_3T.atoms, format='xyz', append=True)
            
    # Clear the gradient after finishing the minimization
    for optimizer in optimizers:
        optimizer.zero_grad()

    return
