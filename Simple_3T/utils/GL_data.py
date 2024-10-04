import numpy as np
import time

class data:
    def __init__(self, in_file, data_file, rotbond_file=None):
        self._parse_default()
        self.parse_in_file(in_file)
        self.parse_data_file(data_file)
        if rotbond_file is not None:
            self.parse_rotbond_file(rotbond_file)
        return

    def parse_in_file(self, in_file):
        coeff_lines = []
        n_atomtype = 0
        with open(in_file, 'r') as f:
            line = f.readline()
            while line:
                words = line.strip().split()
                if len(words) == 0:
                    pass
                elif words[0] == 'pair_coeff':
                    coeff_lines.append(words)
                    n_atomtype = max([n_atomtype, int(words[1]), int(words[2])])
                elif (words[0] == 'pair_modify') and (words[1] == 'mix'):
                    if not (words[2] == 'arithmetic'):
                        raise Exception('GROMACS LAMMPS pair_style mix not "arithmetic"')
                else:
                    pass
                line = f.readline()
        self.epsilon = np.zeros([n_atomtype, n_atomtype])
        self.sigma = np.zeros([n_atomtype, n_atomtype])
        self.epsilon[:,:] = np.nan
        self.sigma[:,:] = np.nan
        for words in coeff_lines:
            i = int(words[1]) - 1
            j = int(words[2]) - 1
            # For debugging only
            #if not np.isnan(self.epsilon[i,j]):
            #    if (self.epsilon[i,j] != float(words[3])) or (self.sigma[i,j] != float(words[4])):
            #        print('Conflicting values',i+1,j+1)
            #        print('Expect:',self.epsilon[i,j],self.sigma[i,j],'obtained:',float(words[3]),float(words[4]))
            self.epsilon[i,j] = float(words[3])
            self.epsilon[j,i] = float(words[3])
            self.sigma[i,j] = float(words[4])
            self.sigma[j,i] = float(words[4])
        # Mixing rule for LAMMPS pair_modify mix arithmetic
        for i in range(n_atomtype):
            for j in range(n_atomtype):
                if np.isnan(self.epsilon[i,j]):
                    self.epsilon[i,j] = np.sqrt( self.epsilon[i,i] * self.epsilon[j,j] )
                if np.isnan(self.sigma[i,j]):
                    self.sigma[i,j] = 0.5 * ( self.sigma[i,i] + self.sigma[j,j] )
        return

    def parse_data_file(self, data_file):
        with open(data_file, 'r') as f:
            line = f.readline()
            self.title = line.strip()
            line = f.readline()
            self.headers = []
            while line:
                words = line.strip().split()
                if len(words) == 0:
                    pass
                else:
                    full_words = ' '.join(words)
                    if full_words in ['Masses','Atoms','Bonds','Angles','Dihedrals','Impropers','Velocities',
                                      'Bond Coeffs','Angle Coeffs','Dihedral Coeffs','Improper Coeffs']:
                        content = self._extract_section(f)
                        #print('Found', full_words, ':', len(content))
                        # Assume content in each section is ordered (it should if GROMACS-LAMMPS convert was used)
                        parse_func_dict = { 'Masses': self._parse_masses,
                                                        'Atoms': self._parse_atoms,
                                                        'Bonds': self._parse_bonds,
                                                        'Angles': self._parse_angles,
                                                        'Dihedrals': self._parse_dihedrals,
                                                        'Impropers': self._parse_impropers,
                                                        'Bond Coeffs': self._parse_bond_coeffs,
                                                        'Angle Coeffs': self._parse_angle_coeffs,
                                                        'Dihedral Coeffs': self._parse_dihedral_coeffs,
                                                        'Improper Coeffs': self._parse_improper_coeffs }
                        if full_words in parse_func_dict:
                            parse_func = parse_func_dict[ full_words ]
                            parse_func(content)                                                     
                    else:
                        self.headers.append(full_words)
                line = f.readline()
        return

    def _extract_section(self, fstream):
        line = fstream.readline()
        line = fstream.readline()
        content = []
        while line:
            words = line.strip().split()
            if len(words) == 0:
                break
            content.append(words)
            line = fstream.readline()
        return content

    def _parse_default(self):
        self.bond_idx = np.array([])
        self.angle_idx = np.array([])
        self.dihedral_idx = np.array([])
        self.improper_idx = np.array([])
        self.bond_harmonic_idx = np.array([])
        self.bond_harmonic_coeffs = np.array([])
        self.angle_harmonic_idx = np.array([])
        self.angle_harmonic_coeffs = np.array([])
        self.angle_charmm_idx = np.array([])
        self.angle_charmm_coeffs = np.array([])
        self.dihedral_multiharm_idx = np.array([])
        self.dihedral_multiharm_coeffs = np.array([])
        self.dihedral_charmm_idx = np.array([])
        self.dihedral_charmm_coeffs = np.array([])
        self.improper_harmonic_idx = np.array([])
        self.improper_harmonic_coeffs = np.array([])
        self.rotbond_idx = np.array([])
        return

    def _parse_masses(self, content):
        atom_mass = []
        for words in content:
            atom_mass.append( float(words[1]) )
        self.atom_mass = np.array(atom_mass)
        return

    def _parse_atoms(self, content):
        atom_molid, atom_type, atom_charge, atom_pos = [], [], [], []
        for words in content:
            atom_molid.append( int(words[1])-1 )
            atom_type.append( int(words[2])-1 )
            atom_charge.append( float(words[3]) )
            atom_pos.append( np.array( [float(word) for word in words[4:7]] ) )
        self.atom_molid = np.array(atom_molid)
        self.atom_type = np.array(atom_type)
        self.atom_charge = np.array(atom_charge)
        self.atom_pos = np.array(atom_pos)
        return
        
    def _parse_bonds(self, content):
        bond_idx = []
        for words in content:
            bond_idx.append( np.array(words[2:4]).astype(int) - 1 )
        self.bond_idx = np.array(bond_idx)
        return

    def _parse_angles(self, content):
        angle_idx = []
        for words in content:
            angle_idx.append( np.array(words[2:5]).astype(int) - 1 )
        self.angle_idx = np.array(angle_idx)
        return

    def _parse_dihedrals(self, content):
        dihedral_idx = []
        for words in content:
            dihedral_idx.append( np.array(words[2:6]).astype(int) - 1 )
        self.dihedral_idx = np.array(dihedral_idx)
        return

    def _parse_impropers(self, content):
        improper_idx = []
        for words in content:
            improper_idx.append( np.array(words[2:6]).astype(int) - 1 )
        self.improper_idx = np.array(improper_idx)
        return

    def _parse_bond_coeffs(self, content):
        bond_harmonic_idx, bond_harmonic_coeffs = [], []
        for words in content:
            if words[1] == 'harmonic':
                bond_harmonic_idx.append( int(words[0])-1 )
                bond_harmonic_coeffs.append( np.array( [float(word) for word in words[2:4]] ) )
            else:
                raise Exception('Unrecognized bond style :',words[1])
        self.bond_harmonic_idx = np.array(bond_harmonic_idx)
        self.bond_harmonic_coeffs = np.array(bond_harmonic_coeffs)
        return

    def _parse_angle_coeffs(self, content):
        angle_harmonic_idx, angle_harmonic_coeffs = [], []
        angle_charmm_idx, angle_charmm_coeffs = [], []
        for words in content:
            if words[1] == 'harmonic':
                angle_harmonic_idx.append( int(words[0])-1 )
                angle_harmonic_coeffs.append( np.array( [float(word) for word in words[2:4]] ) )
            elif words[1] == 'charmm':
                angle_charmm_idx.append( int(words[0])-1 )
                angle_charmm_coeffs.append( np.array( [float(word) for word in words[2:6]] ) )
            else:
                raise Exception('Unrecognized angle style :',words[1])
        self.angle_harmonic_idx = np.array(angle_harmonic_idx)
        self.angle_harmonic_coeffs = np.array(angle_harmonic_coeffs)
        self.angle_charmm_idx = np.array(angle_charmm_idx)
        self.angle_charmm_coeffs = np.array(angle_charmm_coeffs)
        return

    def _parse_dihedral_coeffs(self, content):
        dihedral_multiharm_idx, dihedral_multiharm_coeffs = [], []
        dihedral_charmm_idx, dihedral_charmm_coeffs = [], []
        for words in content:
            if words[1] == 'multi/harmonic':
                dihedral_multiharm_idx.append( int(words[0])-1 )
                dihedral_multiharm_coeffs.append( np.array( [float(word) for word in words[2:7]] ) )
            elif words[1] == 'charmm':
                dihedral_charmm_idx.append( int(words[0])-1 )
                dihedral_charmm_coeffs.append( np.array( [float(word) for word in words[2:6]] ) )
            else:
                raise Exception('Unrecognized dihedral style :',words[1])
        self.dihedral_multiharm_idx = np.array(dihedral_multiharm_idx)
        self.dihedral_multiharm_coeffs = np.array(dihedral_multiharm_coeffs)
        self.dihedral_charmm_idx = np.array(dihedral_charmm_idx)
        self.dihedral_charmm_coeffs = np.array(dihedral_charmm_coeffs)
        return

    def _parse_improper_coeffs(self, content):
        improper_harmonic_idx, improper_harmonic_coeffs = [], []
        for words in content:
            if words[1] == 'harmonic':
                improper_harmonic_idx.append( int(words[0])-1 )
                improper_harmonic_coeffs.append( np.array( [float(word) for word in words[2:4]] ) )
            else:
                raise Exception('Unrecognized improper style :',words[1])
        self.improper_harmonic_idx = np.array(improper_harmonic_idx)
        self.improper_harmonic_coeffs = np.array(improper_harmonic_coeffs)
        return

    def parse_rotbond_file(self, rotbond_file):
        rotbond_idx = []
        with open(rotbond_file, 'r') as f:
            line = f.readline().strip()
            line = f.readline().strip()
            while line:
                words = line.split(',')
                rotbond_idx.append( np.array(words[1:3]).astype(int) - 1 )
                line = f.readline().strip()
        self.rotbond_idx = np.array(rotbond_idx)

        self._assign_micro_group()
        self._assign_special_rotation()
        self._assign_macro_mode()
        return

    def _assign_micro_group(self):
        start = time.time()
        rotbonds, allbonds = dict(), dict()
        for bond_idx in self.rotbond_idx:
            if bond_idx[0] in rotbonds: rotbonds[ bond_idx[0] ] += [ bond_idx[1] ]
            else:                                   rotbonds[ bond_idx[0] ] = [ bond_idx[1] ]
            if bond_idx[1] in rotbonds: rotbonds[ bond_idx[1] ] += [ bond_idx[0] ]
            else:                                   rotbonds[ bond_idx[1] ] = [ bond_idx[0] ]
        for bond in self.bond_idx:
            if bond[0] in allbonds:    allbonds[ bond[0] ] += [ bond[1] ]
            else:                               allbonds[ bond[0] ] = [ bond[1] ]
            if bond[1] in allbonds:    allbonds[ bond[1] ] += [ bond[0] ]
            else:                               allbonds[ bond[1] ] = [ bond[0] ]
        
        n_atoms = self.atom_molid.shape[0]
        lig_atoms = [i for i in range(n_atoms)]
        segments, current_segment = [], []
        while True:
            now = time.time()
            if now - start > 300:
                raise Exception('Infinite loop in ligand segmentation algorithm, check this edge case')
            if len(current_segment) == 0:
                center = lig_atoms.pop(0)
                current_segment.append(center)
            else:
                temp = current_segment.copy()
                restricted = []
                for atom in current_segment:
                    if atom in rotbonds: restricted += rotbonds[atom]
                for atom in current_segment:
                    segment_nbrs = [i for i in allbonds[atom] if not (i in restricted)]
                    temp += segment_nbrs
                current_segment = np.unique(np.array(temp)).tolist()
                remove_count = 0
                for atom in current_segment:
                    if atom in lig_atoms:
                        lig_atoms.remove(atom)
                        remove_count += 1
                if remove_count == 0:
                    segments.append( current_segment.copy() )
                    current_segment = []
            if len(lig_atoms) == 0:
                if len(current_segment) > 0:
                    segments.append( current_segment.copy() )
                    current_segments = []
                break
        for segment in segments:
            segment.sort()
        check = []
        for segment in segments: check += segment
        check = np.unique(np.array(check))
        if len(check) != n_atoms:
            raise Exception('Mistake in ligand segmentation algorithm, check this edge case')
        self.micro_group = segments
        return

    def _assign_special_rotation(self):
        special_rotation = dict()
        bond_idx = self.bond_idx
        movable_idx_list = self.micro_group
        all_movables = []
        for movable_idx in movable_idx_list:
            all_movables += movable_idx
        for group_id, movable_idx in enumerate(movable_idx_list):
            # Find groups which are bonded to at most one other movable group
            group_bonds = []
            for atom_idx in movable_idx:
                pair1 = bond_idx[np.where(bond_idx[:,0]==atom_idx)[0], 1]
                pair2 = bond_idx[np.where(bond_idx[:,1]==atom_idx)[0], 0]
                group_bonds += pair1.tolist() + pair2.tolist()
            group_bonds = [i for i in group_bonds if (not i in movable_idx)]
            group_bonds = list(set(group_bonds))    # make unique
            # We generate special rotation center if and only if there is only 1 anchor atom for this group 
            if len(group_bonds) == 1:
                out_center = group_bonds[0]
                pair1 = bond_idx[np.where(bond_idx[:,0]==out_center)[0], 1]
                pair2 = bond_idx[np.where(bond_idx[:,1]==out_center)[0], 0]
                out_center_bonds = pair1.tolist() + pair2.tolist()
                out_center_bonds = [i for i in out_center_bonds if (not i == out_center)]
                out_center_bonds = list(set(out_center_bonds))  # make unique
                in_bonds = [i for i in out_center_bonds if (i in movable_idx)]
                if len(in_bonds) != 1:
                    raise Exception('Special rotation axes error, center = '+str(out_center)+', group bonds = '+str(in_bonds)+', group idx = '+str(movable_idx))
                in_center = in_bonds[0]
                special_rotation[group_id] = [out_center, in_center, 0]
        # If there is no special rotation, we should return None
        if len(special_rotation)==0:
            self.special_rotation = None
        else:
            self.special_rotation = special_rotation
        return

    def _assign_macro_mode(self):
        n = len(self.micro_group)
        if n == 0:
            self.macro_mode = None
        else:
            self.macro_mode = [[i for i in range(n)]]
        return
        
