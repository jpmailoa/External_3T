from __future__ import print_function, division

import torch

import torch.nn as nn
from torch.nn import ParameterList
from torch.nn.parameter import Parameter

from calculator_3T_FF import calc_E_F_forcefield
from calculator_3T_VASP import calc_E_F_VASP
from calculator_3T_PWMAT import calc_E_F_PWMAT

class PotentialModel(nn.Module):
    def __init__(self, lattice_data, molecules_data, mode):
        super(PotentialModel, self).__init__()
        self.lattice_data = lattice_data
        self.molecules_data = molecules_data
        self.combine_data(lattice_data, molecules_data)
        self.change_mode(mode)
        return

    def combine_data(self, lattice_data, molecules_data):
        nl_atom = len(lattice_data.atom_pos)
        nl_atom_type = len(set(lattice_data.atom_type.tolist()))
        nl_atom_molid = len(set(lattice_data.atom_molid.tolist()))
        nm_atom = [len(molecule_data.atom_pos) for molecule_data in molecules_data]
        nm_atom_type = [len(set(molecule_data.atom_type.tolist())) for molecule_data in molecules_data]

        for i in range(nl_atom_type):
            assert i in lattice_data.atom_type
        for i in range(len(nm_atom_type)):
            for j in range(nm_atom_type[i]):
                assert j in molecules_data[i].atom_type
        
        self.cell = Parameter(torch.Tensor( lattice_data.cell ), requires_grad=False)

        temp = [torch.LongTensor(lattice_data.atom_type)]
        for molecule_data in molecules_data:
            n_temp = len(set(torch.cat( temp, dim=0 ).cpu().numpy().tolist()))
            temp.append( torch.LongTensor(molecule_data.atom_type + n_temp) )
        self.atom_type = torch.cat( temp, dim=0 )

        temp = [torch.Tensor(lattice_data.atom_charge)] + [torch.Tensor(molecule_data.atom_charge) for molecule_data in molecules_data]
        self.atom_charge = Parameter( torch.cat(temp, dim=0), requires_grad=False )

        temp = [torch.Tensor(lattice_data.atom_mass)] + [torch.Tensor(molecule_data.atom_mass) for molecule_data in molecules_data]
        self.atom_mass = Parameter( torch.cat(temp, dim=0), requires_grad=False )

        temp = [torch.LongTensor(lattice_data.atom_molid)]
        for molecule_data in molecules_data:
            n_temp = len(set(torch.cat( temp, dim=0 ).cpu().numpy().tolist()))
            temp.append( torch.LongTensor(molecule_data.atom_molid + n_temp) )
        self.atom_molid = torch.cat( temp, dim=0 )

        bond_idx, angle_idx, dihedral_idx, improper_idx, base = [], [], [], [], nl_atom
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

        n_atom_types = [nl_atom_type] + nm_atom_type
        n = sum(n_atom_types)
        self.epsilon = Parameter(torch.zeros(n,n), requires_grad=False)
        self.sigma = Parameter(torch.zeros(n,n), requires_grad=False)
        self.epsilon[ :nl_atom_type, :nl_atom_type ] = torch.Tensor( lattice_data.epsilon )
        self.sigma[ :nl_atom_type, :nl_atom_type ] = torch.Tensor( lattice_data.sigma )
        base = nl_atom_type
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

        movable_idx_list, nl_micro = [], 0
        if hasattr(lattice_data, 'movable_group'):
            movable_idx_list = lattice_data.movable_group
            nl_micro = len(lattice_data.movable_group)
        base = nl_atom
        for molecule_data in molecules_data:
            molecule_movable_group = [ [i+base for i in group] for group in molecule_data.micro_group ]
            base += len(molecule_data.atom_pos)
            movable_idx_list += molecule_movable_group

        special_rotation = None
        base_micro = nl_micro
        base_atom = nl_atom
        for molecule_data in molecules_data:
            if molecule_data.special_rotation is not None:
                if special_rotation is None: special_rotation = dict()
                for group_id in molecule_data.special_rotation:
                    ori = molecule_data.special_rotation[group_id]
                    special_rotation[ group_id+base_micro ] = [ ori[0]+base_atom, ori[1]+base_atom, ori[2] ]
            base_micro += len(molecule_data.micro_group)
            base_atom += len(molecule_data.atom_pos)

        macro_mode = None
        base_micro = nl_micro
        for molecule_data in molecules_data:
            if molecule_data.macro_mode is not None:
                if macro_mode is None: macro_mode = []
                macro_mode += [[(j+base_micro) for j in group] for group in molecule_data.macro_mode]
                base_micro += len(molecule_data.micro_group)

        xyz = [ torch.Tensor(lattice_data.atom_pos) ]
        for molecule_data in molecules_data:
            xyz.append( torch.Tensor(molecule_data.atom_pos) )
        xyz = torch.cat( xyz, dim=0 )
        self.device = xyz.device
        self.attach_init_inputs(xyz, movable_idx_list, special_rotation=special_rotation, macro_mode=macro_mode)
        return        

    def to(self, device):
        super(PotentialModel, self).to(device)
        self.device = device
        return self

    def change_mode(self, mode):
        assert mode in ['FF', 'VASP']
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
        #self.rearrange_movable_pos_list_pbc()
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
        #self.rearrange_movable_pos_list_pbc()
        self.atom_pos = self.arrange_atom_pos(self.movable_pos_list, self.fixed_pos)
        return
    
    def rearrange_movable_pos_list_pbc(self):
        # We need to move atoms within the same micro group to be within the same side of PBC box
        # Otherwise, micro-group rotation will create very large movement due to misplaced centers
        for micro_group in self.movable_pos_list:
            micro_group.requires_grad = False
        for i, micro_group in enumerate(self.movable_pos_list):
            na = len(micro_group)
            new_pos = torch.zeros(3,3,3,na,3).to(self.device)
            new_pos.requires_grad = False
            for a in range(-1,2):
                for b in range(-1,2):
                    for c in range(-1,2):
                        new_pos[a+1,b+1,c+1] = micro_group + a*self.cell[0] + b*self.cell[1] + c*self.cell[2]
            new_pos = new_pos.view(27,na,3)
            dist = torch.linalg.norm(new_pos[:,1:] - micro_group[0], dim=2)
            min_idx = torch.argmin(dist, dim=0)
            #min_dist = torch.min(dist, dim=0)
            if na > 1:
                #print(min_dist)
                for j in range(1, na):
                    micro_group[j,:] = new_pos[ min_idx[j-1] ,j,:]
                #print( torch.sum(self.movable_pos_list[i] - micro_group) )
                # Do sanity check to ensure all micro-group distances are the shortest within PBC periodicity
                new_pos = torch.zeros(3,3,3,na,3).to(self.device)
                for a in range(-1,2):
                    for b in range(-1,2):
                        for c in range(-1,2):
                            new_pos[a+1,b+1,c+1] = micro_group + a*self.cell[0] + b*self.cell[1] + c*self.cell[2]
                new_pos = new_pos.view(27,na,3)
                dist = torch.linalg.norm(new_pos.unsqueeze(2).repeat(1,1,na,1) - micro_group, dim=3)
                min_idx = torch.argmin(dist, dim=0)
                #print(min_idx.shape)
                #print(min_idx)
                assert torch.all( min_idx == 13 ), 'Sanity check fails. This should work most of the time, but there is no theoretical guarantee. Check manually.'
        for micro_group in self.movable_pos_list:
            micro_group.requires_grad = True
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

    def recenter_initial_molecule(self, new_mol_ctr):
        old_mol_pos = self.molecule_data.atom_pos
        new_mol_pos = old_mol_pos - old_mol_pos.mean(axis=0) + new_mol_ctr
        self.molecule_data.atom_pos = new_mol_pos

        xyz = torch.cat( [torch.Tensor(self.lattice_data.atom_pos), torch.Tensor(self.molecule_data.atom_pos)], dim=0).to(self.device)
        #self.attach_init_inputs(xyz, movable_idx_list, special_rotation=special_rotation, macro_mode=macro_mode)
        self.movable_pos_list = ParameterList( [Parameter(xyz[movable_idx,:], requires_grad = True)
                                                for movable_idx in self.movable_idx_list] )
        self.fixed_pos = Parameter(xyz[self.fixed_idx,:], requires_grad = False)

        self.to(self.device)
        #self.rearrange_movable_pos_list_pbc()
        self.atom_pos = self.arrange_atom_pos(self.movable_pos_list, self.fixed_pos)
        
        return        

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

    def jolt_movable_atoms(self, seed = None, max_translation = 5.0, max_rotation = 3.14, ignore_last = False):
        if not (seed==None):
            torch.manual_seed(seed)
        self.translation_list.requires_grad, self.rotation_list.requires_grad = False, False
        translation_shape = [i for i in self.translation_list.shape] #self.translation_list.shape
        rotation_shape = [i for i in self.rotation_list.shape] #self.rotation_list.shape
        if ignore_last:
            translation_shape[0] += -1 #translation_shape = [translation_shape[0] - 1, translation_shape[1], translation_shape[2]]
            rotation_shape[0] += -1 #rotation_shape = [rotation_shape[0] - 1, rotation_shape[1]]
            self.translation_list[:-1,:,:] = (torch.rand(translation_shape) - 0.5) * max_translation * 2
            self.rotation_list[:-1,:] = (torch.rand(rotation_shape) - 0.5) * max_rotation * 2
        else:
            self.translation_list[:,:,:] = torch.rand(translation_shape) * max_translation
            self.rotation_list[:,:] = torch.rand(rotation_shape) * max_rotation
        self.translation_list.requires_grad, self.rotation_list.requires_grad = True, True

        # Take care of special rotation
        if self.special_rotation_idx != None:
            rotation_shape = [i for i in self.special_rotation_list.shape] #self.special_rotation_list.shape
            self.special_rotation_list.requires_grad = False
            self.special_rotation_list[:,:] = (torch.rand(rotation_shape) - 0.5) * max_rotation * 2
            for j in torch.where(self.special_rotation_idx[:,3]==1)[0]:
                if torch.rand(1) > 0.5 and max_rotation != 0.0:
                    self.special_rotation_list[j,:] += 3.1416   # with 50% probability, initially rotate these groups 180 degrees
            self.special_rotation_list.requires_grad = True

        # Take care of macro mode. We only reset this to 0. We will not give a kick as the groups are too big.
        if self.macro_mode_idx != None:
            self.macro_mode_translation_list.requires_grad = False
            self.macro_mode_rotation_list.requires_grad = False
            self.macro_mode_translation_list[:,:,:] = 0
            self.macro_mode_rotation_list[:,:] = 0
            self.macro_mode_translation_list.requires_grad = True
            self.macro_mode_rotation_list.requires_grad = True            

        return            

    def random_full_rotation(self, rotated_indices, seed = None):
        if not (seed==None):
            torch.manual_seed(seed)
        all_pos = torch.Tensor(self.atom_pos.detach().cpu().numpy())
        rot_pos = all_pos[ rotated_indices,: ]
        rot_com = rot_pos.mean(dim=0)
        pi = torch.acos(torch.Tensor([-1]))
        theta, phi, z = 2*pi*torch.rand([1]), 2*pi*torch.rand([1]), 2*torch.rand([1])
        sin_t, sin_p = torch.sin(theta), torch.sin(phi)
        cos_t, cos_p = torch.cos(theta), torch.cos(phi)
        r = torch.sqrt(z)
        V = torch.Tensor([sin_p*r, cos_p*r, torch.sqrt(2-z)])
        S = torch.Tensor([V[0]*cos_t - V[1]*sin_t, V[0]*sin_t + V[1]*cos_t])
        R = torch.Tensor([ [ V[0]*S[0]-cos_t, V[0]*S[1]-sin_t, V[0]*V[2] ],
                                     [ V[1]*S[0]+sin_t, V[1]*S[1]-cos_t, V[1]*V[2] ],
                                     [ V[2]*S[0], V[2]*S[1], 1-z ] ])
        rot_pos = rot_com + torch.matmul(rot_pos-rot_com, R.transpose(0,1))
        all_pos[ rotated_indices,: ] = rot_pos
        self.attach_init_inputs(all_pos, [rotated_indices] )
        return        

    def forward(self):
        # self.atom_pos needs to be updated because self.movable_pos is updated on each epoch
        self.atom_pos = self.arrange_atom_pos(self.movable_pos_list, self.fixed_pos)

        # Now we use available energy and forces from calculator (force field or DFT)
        # Mind that these are numpy arrays, not pytorch tensors
        if self.mode == 'FF':
            #E_total, F_atoms = self.calc_E_F_forcefield()
            E_total, F_atoms = calc_E_F_forcefield(self)
        elif self.mode == 'VASP':
            E_total, F_atoms = calc_E_F_VASP(self)
        elif self.mode == 'PWMAT':
            E_total, F_atoms = calc_E_F_PWMAT(self)
        else: raise Exception('Unimplemented 3T mode')

        # Finally we define a new output, the cost function C to be used in the backward chain rule.
        # This output C only makes sense for chain rule purposes to optimize 3T model parameters.
        F_atoms = torch.Tensor(F_atoms).to(self.device)
        C_total = -torch.sum( self.atom_pos * F_atoms )
        
        return E_total, C_total
