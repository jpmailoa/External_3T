from __future__ import print_function, division

import torch

import torch.nn as nn
from torch.nn import ParameterList
from torch.nn.parameter import Parameter

import numpy as np
from calculator_3T_FF import calc_E_F_forcefield

class PotentialModel(nn.Module):
    def __init__(self, molecules_data, atoms, mode):
        super(PotentialModel, self).__init__()
        self.molecules_data = molecules_data
        self.atoms = atoms
        self.combine_data(molecules_data)
        self.change_mode(mode)
        return

    def combine_data(self, molecules_data):
        nm_atom = [len(molecule_data.atom_pos) for molecule_data in molecules_data]
        nm_atom_type = [len(set(molecule_data.atom_type.tolist())) for molecule_data in molecules_data]

        for i in range(len(nm_atom_type)):
            for j in range(nm_atom_type[i]):
                assert j in molecules_data[i].atom_type
        
        self.cell = Parameter(torch.Tensor( np.array(self.atoms.cell) ), requires_grad=False)
        
        temp = [torch.LongTensor([])]
        for molecule_data in molecules_data:
            n_temp = len(set(torch.cat( temp, dim=0 ).cpu().numpy().tolist()))
            temp.append( torch.LongTensor(molecule_data.atom_type + n_temp) )
        self.atom_type = torch.cat( temp, dim=0 )

        temp = [torch.Tensor(molecule_data.atom_charge) for molecule_data in molecules_data]
        self.atom_charge = Parameter( torch.cat(temp, dim=0), requires_grad=False )

        temp = [torch.Tensor(molecule_data.atom_mass) for molecule_data in molecules_data]
        self.atom_mass = Parameter( torch.cat(temp, dim=0), requires_grad=False )

        temp = [torch.LongTensor([])]
        for molecule_data in molecules_data:
            n_temp = len(set(torch.cat( temp, dim=0 ).cpu().numpy().tolist()))
            temp.append( torch.LongTensor(molecule_data.atom_molid + n_temp) )
        self.atom_molid = torch.cat( temp, dim=0 )

        bond_idx, angle_idx, dihedral_idx, improper_idx, base = [], [], [], [], 0
        for molecule_data in molecules_data:
            bond_idx.append( torch.LongTensor( molecule_data.bond_idx + base ) )
            angle_idx.append( torch.LongTensor( molecule_data.angle_idx + base ) )
            dihedral_idx.append( torch.LongTensor( molecule_data.dihedral_idx + base ) )
            improper_idx.append( torch.LongTensor( molecule_data.improper_idx + base ) )
            base += len(molecule_data.atom_pos)
        self.bond_idx = torch.cat(bond_idx, dim=0)
        self.angle_idx = torch.cat(angle_idx, dim=0)
        self.dihedral_idx = torch.cat(dihedral_idx, dim=0)
        self.improper_idx = torch.cat(improper_idx, dim=0)

        n_atom_types = [0] + nm_atom_type
        n = sum(n_atom_types)
        self.epsilon = Parameter(torch.zeros(n,n), requires_grad=False)
        self.sigma = Parameter(torch.zeros(n,n), requires_grad=False)
        base = 0
        for k, molecule_data in enumerate(molecules_data):
            end = base + nm_atom_type[k]
            self.epsilon[ base:end, base:end ] = torch.Tensor( molecule_data.epsilon )
            self.sigma[ base:end, base:end ] = torch.Tensor( molecule_data.sigma )
            for i in range(end):
                for j in range(base, end):
                    self.epsilon[i,j] = torch.sqrt( self.epsilon[i,i] * self.epsilon[j,j] )
                    self.sigma[i,j] = 0.5 * ( self.sigma[i,i] + self.sigma[j,j] )
                    self.epsilon[j,i] = self.epsilon[i,j]
                    self.sigma[j,i] = self.sigma[i,j]
            base = end

        temp1, temp2, base = [], [], 0
        for molecule_data in molecules_data:
            temp1.append( torch.LongTensor( molecule_data.bond_harmonic_idx + base ) )
            temp2.append( torch.Tensor( molecule_data.bond_harmonic_coeffs ) )
            base += len(molecule_data.bond_idx)
        self.bond_harmonic_idx = torch.cat( temp1, dim=0 )
        self.bond_harmonic_coeffs = Parameter( torch.cat(temp2, dim=0), requires_grad=False )

        temp1, temp2, base = [], [], 0
        for molecule_data in molecules_data:
            temp1.append( torch.LongTensor( molecule_data.angle_harmonic_idx + base ) )
            temp2.append( torch.Tensor( molecule_data.angle_harmonic_coeffs ) )
            base += len(molecule_data.angle_idx)
        self.angle_harmonic_idx = torch.cat( temp1, dim=0 )
        self.angle_harmonic_coeffs = Parameter( torch.cat(temp2, dim=0), requires_grad=False )

        temp1, temp2, base = [], [], 0
        for molecule_data in molecules_data:
            temp1.append( torch.LongTensor( molecule_data.angle_charmm_idx + base ) )
            temp2.append( torch.Tensor( molecule_data.angle_charmm_coeffs ) )
            base += len(molecule_data.angle_idx)
        self.angle_charmm_idx = torch.cat( temp1, dim=0 )
        self.angle_charmm_coeffs = Parameter( torch.cat(temp2, dim=0), requires_grad=False )

        temp1, temp2, base = [], [], 0
        for molecule_data in molecules_data:
            temp1.append( torch.LongTensor( molecule_data.dihedral_multiharm_idx + base ) )
            temp2.append( torch.Tensor( molecule_data.dihedral_multiharm_coeffs ) )
            base += len(molecule_data.dihedral_idx)
        self.dihedral_multiharm_idx = torch.cat( temp1, dim=0 )
        self.dihedral_multiharm_coeffs = Parameter( torch.cat(temp2, dim=0), requires_grad=False )

        temp1, temp2, base = [], [], 0
        for molecule_data in molecules_data:
            temp1.append( torch.LongTensor( molecule_data.dihedral_charmm_idx + base ) )
            temp2.append( torch.Tensor( molecule_data.dihedral_charmm_coeffs ) )
            base += len(molecule_data.dihedral_idx)
        self.dihedral_charmm_idx = torch.cat( temp1, dim=0 )
        self.dihedral_charmm_coeffs = Parameter( torch.cat(temp2, dim=0), requires_grad=False )

        temp1, temp2, base = [], [], 0
        for molecule_data in molecules_data:
            temp1.append( torch.LongTensor( molecule_data.improper_harmonic_idx + base ) )
            temp2.append( torch.Tensor( molecule_data.improper_harmonic_coeffs ) )
            base += len(molecule_data.improper_idx)
        self.improper_harmonic_idx = torch.cat( temp1, dim=0 )
        self.improper_harmonic_coeffs = Parameter( torch.cat(temp2, dim=0), requires_grad=False )

        na = self.atom_type.shape[0]
        self.sb_mask = Parameter(torch.ones(na,na), requires_grad=False)
        # Gromacs-LAMMPS files have special_bonds set to:
        # 1st neighbor = 0, 2nd neighbor = 0.0, 3rd neighbor = 1.0
        # The rest of LJ & Coulomb interactions are calculated normally
        self.sb_mask[self.bond_idx[:,0], self.bond_idx[:,1]] = 0
        self.sb_mask[self.bond_idx[:,1], self.bond_idx[:,0]] = 0
        self.sb_mask[self.angle_idx[:,0], self.angle_idx[:,2]] = 0
        self.sb_mask[self.angle_idx[:,2], self.angle_idx[:,0]] = 0
        self.sb_mask[self.dihedral_idx[:,0], self.dihedral_idx[:,3]] = 1.0
        self.sb_mask[self.dihedral_idx[:,3], self.dihedral_idx[:,0]] = 1.0

        self.ij_mask = Parameter(torch.nonzero(torch.triu(torch.ones(na,na, dtype=int), diagonal=1), as_tuple=False), requires_grad=False)
        self.coulomb_coeff = 8.99e9 * 1.602e-19 * 1.602e-19 / 1e-10 / 4.184 / 1e3 * 6.022e23
        #self.coulomb_coeff = 332.073

        movable_idx_list = []
        base = 0
        for molecule_data in molecules_data:
            molecule_movable_group = [ [i+base for i in group] for group in molecule_data.micro_group ]
            base += len(molecule_data.atom_pos)
            movable_idx_list += molecule_movable_group

        special_rotation = None
        base_micro = 0
        base_atom = 0
        for molecule_data in molecules_data:
            if molecule_data.special_rotation is not None:
                if special_rotation is None: special_rotation = dict()
                for group_id in molecule_data.special_rotation:
                    ori = molecule_data.special_rotation[group_id]
                    special_rotation[ group_id+base_micro ] = [ ori[0]+base_atom, ori[1]+base_atom, ori[2] ]
            base_micro += len(molecule_data.micro_group)
            base_atom += len(molecule_data.atom_pos)

        macro_mode = None
        base_micro = 0
        for molecule_data in molecules_data:
            if molecule_data.macro_mode is not None:
                if macro_mode is None: macro_mode = []
                macro_mode += [[(j+base_micro) for j in group] for group in molecule_data.macro_mode]
                base_micro += len(molecule_data.micro_group)

        xyz = []
        for molecule_data in molecules_data:
            xyz.append( torch.Tensor(molecule_data.atom_pos) )
        xyz = torch.cat( xyz, dim=0 )
        self.device = xyz.device
        self.attach_init_inputs(xyz, movable_idx_list, special_rotation=special_rotation, macro_mode=macro_mode)

        # Now we need to ensure that the ASE Atoms object symbols match the internal representation we have in 3T model
        mass_elem_dict = {1:'H', 7:'Li', 9:'Be', 11:'B', 12:'C', 14:'N', 16:'O', 19:'F',
                    23:'Na', 24:'Mg', 27:'Al', 28:'Si', 31:'P', 32:'S', 35:'Cl',
                    39:'K', 40:'Ca', 70:'Ga', 73:'Ge', 75:'As', 79:'Se', 80:'Br',
                    85:'Rb', 88:'Sr', 115:'In', 119:'Sn', 122:'Sb', 128:'Te', 127:'I', 207:'Pb'} # this is rounded mass to elem format
        atom_type = self.atom_type.cpu().detach().numpy().astype(int) # this is already in 0 to n_type-1 format
        temp = self.atom_mass.detach().cpu().numpy().astype(float)
        type_elem_dict = {}
        for i in range(temp.shape[0]):
            type_elem_dict[ i ] = mass_elem_dict[ round(temp[i]) ]
        del temp
        atom_elem = [ type_elem_dict[i] for i in atom_type ]
        self.atoms.symbols = atom_elem

        return        

    def to(self, device):
        super(PotentialModel, self).to(device)
        self.device = device
        return self

    def change_mode(self, mode):
        assert mode in ['FF', 'ASE_CALC']
        self.mode = mode
        return

    def attach_init_inputs(self, xyz, movable_idx_list, special_rotation = None, macro_mode = None):
        # Ensure movable_idx content is unique
        # Ideally movable_idx is ordered, but it is fine if it is not ordered. 
        movable_dict = dict()
        for movable_idx in movable_idx_list:
            for idx in movable_idx:
                if idx in movable_dict: raise Exception('Movable atom index',idx,'appears more than once')
                movable_dict[idx] = True
        na = xyz.shape[0]
        fixed_idx = []
        for i in range(na):
            if not (i in movable_dict):
                fixed_idx.append(i)
        self.movable_idx_list = ParameterList( [Parameter(torch.LongTensor(movable_idx), requires_grad=False)
                                                for movable_idx in movable_idx_list] )
        self.fixed_idx = torch.LongTensor(fixed_idx)
        self.movable_pos_list = ParameterList( [Parameter(xyz[movable_idx,:], requires_grad = True)
                                                for movable_idx in self.movable_idx_list] )
        self.fixed_pos = Parameter(xyz[self.fixed_idx,:], requires_grad = False)

        self.translation_list = Parameter(torch.zeros(len(movable_idx_list),1,3), requires_grad = True)
        self.rotation_list = Parameter(torch.zeros(len(movable_idx_list),3), requires_grad = True)

        # If special rotation centers are defined
        # special_rotation will be dictionary of movable_idx_list group -> bonded atom idx
        self.special_rotation = special_rotation
        if special_rotation != None:
            assert len(special_rotation)<=len(movable_idx_list)
            for group_id in special_rotation:
                assert group_id in range(len(movable_idx_list))
                assert special_rotation[group_id][0] in range(self.atom_type.shape[0])
                assert special_rotation[group_id][1] in range(self.atom_type.shape[0])
                assert special_rotation[group_id][2] in range(2)
            self.special_rotation_idx = Parameter(torch.LongTensor([(i,j[0],j[1],j[2]) for i,j in special_rotation.items()]), requires_grad=False)
            self.special_rotation_list = Parameter(torch.zeros(len(special_rotation),1), requires_grad = True)
        else:
            self.special_rotation_idx = None
            self.special_rotation_list = None

        self.macro_mode = macro_mode
        if macro_mode != None:
            flat_group = [item for sublist in macro_mode for item in sublist]
            unique_group = list(set(flat_group))
            assert len(flat_group) == len(unique_group)
            for group_id in flat_group:
                assert group_id in range(len(movable_idx_list))
            self.macro_mode_idx = ParameterList( [Parameter(torch.LongTensor(group_list), requires_grad=False)
                                                  for group_list in macro_mode] )
            self.macro_mode_translation_list = Parameter(torch.zeros(len(macro_mode),1,3), requires_grad = True)
            self.macro_mode_rotation_list = Parameter(torch.zeros(len(macro_mode),3), requires_grad = True)
        else:
            self.macro_mode_idx = None
            self.macro_mode_translation_list = None
            self.macro_mode_rotation_list = None                            

        self.to(self.device)
        self.atom_pos = self.arrange_atom_pos(self.movable_pos_list, self.fixed_pos)
        return

    def reset_positions(self, xyz, move_macro_group_into_pbc=False):
        na = len(self.fixed_pos) + sum([len(movable_pos) for movable_pos in self.movable_pos_list])
        assert xyz.shape[0] == na
        self.movable_pos_list = ParameterList( [Parameter(xyz[movable_idx,:], requires_grad = True)
                                                for movable_idx in self.movable_idx_list] )
        self.fixed_pos = Parameter(xyz[self.fixed_idx,:], requires_grad = False)

        self.translation_list.requires_grad = False
        self.translation_list[:,:,:] = 0
        self.translation_list.requires_grad = True
        
        self.rotation_list.requires_grad = False
        self.rotation_list[:,:] = 0
        self.rotation_list.requires_grad = True

        if not self.special_rotation_idx is None:
            self.special_rotation_list.requires_grad = False
            self.special_rotation_list[:,:] = 0
            self.special_rotation_list.requires_grad = True

        if not self.macro_mode_idx is None:
            self.macro_mode_translation_list.requires_grad = False
            self.macro_mode_translation_list[:,:,:] = 0
            self.macro_mode_translation_list.requires_grad = True

            self.macro_mode_rotation_list.requires_grad = False
            self.macro_mode_rotation_list[:,:] = 0
            self.macro_mode_rotation_list.requires_grad = True

            if move_macro_group_into_pbc:
                cell_inv = self.cell.inverse()
                nmac = len(self.macro_mode_idx)
                for movable_pos in self.movable_pos_list:
                    movable_pos.requires_grad = False
                for i in range(nmac):
                    macro_pos = torch.cat([ self.movable_pos_list[j] for j in self.macro_mode_idx[i] ], dim=0)
                    macro_com = macro_pos.mean(dim=0).view(1,3)
                    macro_rel_pos = macro_pos - macro_com
                    com_cell_rel_pos = torch.matmul(macro_com, cell_inv)
                    com_cell_rel_pos = com_cell_rel_pos - com_cell_rel_pos.floor()
                    macro_com = torch.matmul(com_cell_rel_pos, self.cell)
                    macro_pos = macro_rel_pos + macro_com
                    zero = torch.LongTensor([0]).to(self.device)
                    ng = torch.LongTensor([len(self.movable_pos_list[j]) for j in self.macro_mode_idx[i] ]).to(self.device)
                    indices = torch.cumsum(torch.cat([zero, ng], dim=0), dim=0)
                    for j,k in enumerate(self.macro_mode_idx[i]):
                        self.movable_pos_list[k][:,:] = macro_pos[indices[j]:indices[j+1],:]
                for movable_pos in self.movable_pos_list:
                    movable_pos.requires_grad = True
        
        self.to(self.device)
        self.atom_pos = self.arrange_atom_pos(self.movable_pos_list, self.fixed_pos)
        return
    
    def arrange_atom_pos(self, movable_pos_list, fixed_pos):
        if self.special_rotation_idx != None:
            movable_pos_list = self.axis_rotate(movable_pos_list, fixed_pos)

        movable_pos_list = self.micro_rotate_translate(movable_pos_list)

        if self.macro_mode_idx != None:
            movable_pos_list = self.macro_rotate_translate(movable_pos_list)
        
        na = sum([movable_pos.shape[0] for movable_pos in movable_pos_list]) + fixed_pos.shape[0]
        atom_pos = torch.zeros(na,3).to(self.device)
        for i in range(len(movable_pos_list)):
            atom_pos[self.movable_idx_list[i],:] = movable_pos_list[i]
        atom_pos[self.fixed_idx,:] = fixed_pos

        return atom_pos

    def generate_rot_matrix(self, na, rot_angles):
        a, b, c = rot_angles[:,0], rot_angles[:,1], rot_angles[:,2]
        sin_a, cos_a = torch.sin(a), torch.cos(a)
        sin_b, cos_b = torch.sin(b), torch.cos(b)
        sin_c, cos_c = torch.sin(c), torch.cos(c)
        Ra = torch.zeros(na,3,3).to(self.device)
        Rb = torch.zeros(na,3,3).to(self.device)
        Rc = torch.zeros(na,3,3).to(self.device)
        Ra[:,0,0] = cos_a
        Ra[:,0,1] = -sin_a
        Ra[:,1,0] = sin_a
        Ra[:,1,1] = cos_a
        Ra[:,2,2] = 1
        Rb[:,0,0] = cos_b
        Rb[:,0,2] = sin_b
        Rb[:,2,0] = -sin_b
        Rb[:,2,2] = cos_b
        Rb[:,1,1] = 1
        Rc[:,1,1] = cos_c
        Rc[:,1,2] = -sin_c
        Rc[:,2,1] = sin_c
        Rc[:,2,2] = cos_c
        Rc[:,0,0] = 1
        R = torch.matmul(Ra, torch.matmul(Rb, Rc))
        return R

    def micro_rotate_translate(self, in_xyz_list):
        zero = torch.LongTensor([0]).to(self.device)
        nm = len(in_xyz_list)
        ng = torch.LongTensor([len(i) for i in in_xyz_list]).to(self.device)
        na = torch.sum(ng)
        in_xyz = torch.cat([in_xyz_list[i] for i in range(nm)], dim=0)
        com_xyz = torch.cat([in_xyz_list[i].mean(dim=0).expand(ng[i],3) for i in range(nm)], dim=0)
        trans_xyz = torch.cat([self.translation_list[i,:,:].expand(ng[i],3) for i in range(nm)], dim=0)
        #rot_angles = torch.cat([self.rotation_list[i,:].expand(ng[i],3) for i in range(nm)], dim=0)
        rot_angles = torch.repeat_interleave(self.rotation_list, ng, dim=0)
        R = self.generate_rot_matrix(na, rot_angles)
        frame_xyz = in_xyz - com_xyz
        rot_xyz = torch.matmul(frame_xyz.unsqueeze(1), R.transpose(1,2)).view(na,3)
        out_xyz = rot_xyz + com_xyz + trans_xyz
        
        indices = torch.cumsum(torch.cat([zero, ng], dim=0), dim=0)
        out_xyz_list = [out_xyz[ indices[i]:indices[i+1], :] for i in range(nm)]

        return out_xyz_list

    def macro_rotate_translate(self, in_xyz_list):
        zero = torch.LongTensor([0]).to(self.device)
        nm = len(in_xyz_list)
        ng = torch.LongTensor([len(i) for i in in_xyz_list]).to(self.device)
        na = torch.sum(ng)
        nmac = len(self.macro_mode_idx)
        indices = torch.cumsum(torch.cat([zero, ng], dim=0), dim=0)
        in_xyz = torch.cat([in_xyz_list[i] for i in range(nm)], dim=0)
        com_xyz = torch.cat([in_xyz_list[i].mean(dim=0).expand(ng[i],3) for i in range(nm)], dim=0)
        trans_xyz = torch.zeros(na,3).to(self.device)
        rot_angles = torch.zeros(na,3).to(self.device)
        rot_mode_macro = True
        for i in range(nmac):
            macro_movable_idx = torch.cat([ torch.arange(indices[j],indices[j+1]) for j in self.macro_mode_idx[i] ])
            trans_xyz[macro_movable_idx,:] = self.macro_mode_translation_list[i]
            rot_angles[macro_movable_idx,:] = self.macro_mode_rotation_list[i]
            if rot_mode_macro:
                # We need to replace com_xyz of the macro groups with the macro centers
                com_xyz[macro_movable_idx,:] = in_xyz[macro_movable_idx,:].mean(dim=0)

        R = self.generate_rot_matrix(na, rot_angles)
        frame_xyz = in_xyz - com_xyz
        rot_xyz = torch.matmul(frame_xyz.unsqueeze(1), R.transpose(1,2)).view(na,3)
        # Use the following one for rotating entire macro group
        out_xyz = rot_xyz + com_xyz + trans_xyz
##        # Use the following to disable macro group rotation
##        out_xyz = in_xyz + trans_xyz

        out_xyz_list = [out_xyz[ indices[i]:indices[i+1], :] for i in range(nm)]

        return out_xyz_list

    def axis_rotate(self, movable_pos_list, fixed_pos):
        na = sum([movable_pos.shape[0] for movable_pos in movable_pos_list]) + fixed_pos.shape[0]
        atom_pos = torch.zeros(na,3).to(self.device)
        for i in range(len(movable_pos_list)):
            atom_pos[self.movable_idx_list[i],:] = movable_pos_list[i]
        atom_pos[self.fixed_idx,:] = fixed_pos
        zero = torch.LongTensor([0]).to(self.device)

        rot_xyz_list = [movable_pos for movable_pos in movable_pos_list]

        ns = torch.LongTensor([self.movable_idx_list[gi].shape[0] for gi in self.special_rotation_idx[:,0]]).to(self.device)
        C = torch.cat([atom_pos[self.movable_idx_list[gi],:] for gi in self.special_rotation_idx[:,0]], dim=0)
        A = torch.repeat_interleave(atom_pos[self.special_rotation_idx[:,1],:], ns, dim=0)
        B = torch.repeat_interleave(atom_pos[self.special_rotation_idx[:,2],:], ns, dim=0)
        theta = torch.repeat_interleave(self.special_rotation_list[:,[0]], ns, dim=0).expand(torch.sum(ns),3)
        U = B-A
        R = C-A
        u = U/torch.linalg.norm(U,axis=1).view(-1,1)
        Z = (R*u).sum(axis=1).view(-1,1)*u
        x = R-Z
        y = torch.cross(u, x, dim=1)
        rot_pos = A + Z + x*torch.cos(theta) + y*torch.sin(theta)
        indices = torch.cumsum(torch.cat([zero, ns], dim=0), dim=0)
        for i in range(len(ns)):
            gi = self.special_rotation_idx[i,0]
            rot_xyz_list[gi] = rot_pos[ indices[i]:indices[i+1], :]

        return rot_xyz_list

    def forward(self):
        # self.atom_pos needs to be updated because self.movable_pos is updated on each epoch
        self.atom_pos = self.arrange_atom_pos(self.movable_pos_list, self.fixed_pos)

        # now we update the atom positions within the ASE object
        self.atoms.positions = self.atom_pos.detach().cpu().numpy()

        orig_xyz = self.molecules_data[0].atom_pos
        after_T0_xyz = self.atoms.positions
        print(np.abs(after_T0_xyz-orig_xyz).sum(axis=0))


        # Now we use available energy and forces from calculator (force field or DFT)
        # Mind that these are numpy arrays, not pytorch tensors
        if self.mode == 'FF':
            E_total, F_atoms = calc_E_F_forcefield(self)
        elif self.mode == 'ASE_CALC':
            # ASE uses units eV and eV/A.
            # Let's convert this to kcal/mol and kcal/mol/A for consistency
            scaler = 1.602e-19 / 4.184 / 1e3 * 6.02e23
            E_total = scaler * self.atoms.get_potential_energy()
            F_atoms = scaler * self.atoms.get_forces()
        else: raise Exception('Unimplemented 3T mode')

        # Finally we define a new output, the cost function C to be used in the backward chain rule.
        # This output C only makes sense for chain rule purposes to optimize 3T model parameters.
        F_atoms = torch.Tensor(F_atoms).to(self.device)
        C_total = -torch.sum( self.atom_pos * F_atoms )
        
        return E_total, C_total
