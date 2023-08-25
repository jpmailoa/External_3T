import os, json
import numpy as np
import utils

class data:
    def __init__(self, ase_obj, override=None, VASP_template=None, PWMAT_template=None):
        self.parse_ase_obj(ase_obj)
        if not (override is None):
            self.override_params(override)
        if not (VASP_template is None):
            self.VASP_template = VASP_template
        if not (PWMAT_template is None):
            self.PWMAT_template = PWMAT_template
        return

    def parse_ase_obj(self, ase_obj):
        mass_dict = {'H':1, 'C':12, 'N':14, 'O':16, 'I':127, 'Pb':207, 'Li':7}

        atom_types, used_elems = [], []
        self.atom_pos = ase_obj.positions
        self.cell = np.array( ase_obj.cell )
        self.symbols = [i for i in ase_obj.symbols]
        
        LJ_file = os.path.join( os.path.split(utils.__file__)[0], 'LJ_lattice.param' )
        LJ_dict = {}
        with open(LJ_file,'r') as f:
            line = f.readline()
            while line:
                words = line.strip().split()
                assert words[0] == words[1]
                elem, epsilon, sigma = [ words[0], float(words[3]), float(words[4]) ]
                LJ_dict[ elem ] = [epsilon, sigma]
                line = f.readline()
        
        atom_type_dict, type_elem_dict = dict(), dict()
        atom_type = []
        for i in range(len(self.symbols)):
            elem = self.symbols[i]
            if not elem in atom_type_dict:
                atom_type_dict[ elem ] = len(atom_type_dict)
                type_elem_dict[ len(atom_type_dict)-1 ] = elem
            atom_type.append( atom_type_dict[ elem ] )
        self.atom_type = np.array(atom_type)
        self.atom_mass = np.array( [float(mass_dict[type_elem_dict[i]]) for i in range(len(atom_type_dict))] )
        
        n_atomtype = len(atom_type_dict)
        self.epsilon = np.zeros([n_atomtype, n_atomtype])
        self.sigma = np.zeros([n_atomtype, n_atomtype])
        self.epsilon[:,:] = np.nan
        self.sigma[:,:] = np.nan
        # Mixing rule for LAMMPS pair_modify mix arithmetic
        for i in range(n_atomtype):
            epsilon_i, sigma_i = LJ_dict[ type_elem_dict[i] ]
            for j in range(n_atomtype):
                epsilon_j, sigma_j = LJ_dict[ type_elem_dict[j] ]
                self.epsilon[i,j] = np.sqrt( epsilon_i * epsilon_j )
                self.sigma[i,j] = 0.5 * ( sigma_i + sigma_j )

        self.atom_charge = np.zeros( len(self.atom_pos), dtype=float )
        self.atom_molid = np.zeros( len(self.atom_pos), dtype=int )
            
        return

    def override_params(self, override_file):
        override = json.load(open(override_file,'r'))
        if 'atom_charge_proximity' in override:
            self._override_atom_charge_proximity(override['atom_charge_proximity'])
        if 'movable_group' in override:
            self._override_movable_group(override['movable_group'])
        return

    def _override_atom_charge_proximity(self, charge_dict):
        threshold = 1.5 #assume 1.5A is a good threshold for proximity definition

        for elem in charge_dict:
            if '_' in elem:
                center, edge = elem.split('_')
                idx_ctr = np.array([i for i in range(len(self.symbols)) if self.symbols[i]==center], dtype=int)
                idx_edg = np.array([i for i in range(len(self.symbols)) if self.symbols[i]==edge], dtype=int)
                R_ctr = np.expand_dims(self.atom_pos[idx_ctr,:], axis=1).repeat(len(idx_edg), axis=1)
                R_edg = np.expand_dims(np.expand_dims(self.atom_pos[idx_edg,:], axis=0).repeat(len(idx_ctr), axis=0), axis=0).repeat(3*3*3, axis=0)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            R_edg[i*9+j*3+k,:] += (i-1)*self.cell[0] + (j-1)*self.cell[1] + (k-1)*self.cell[2]
                ctr_dist = np.linalg.norm(R_edg - R_ctr, axis=3)
                ctr_dist = np.min( np.min(ctr_dist, axis=0), axis=1 )
                idx_ok = np.array([i for i in range(len(ctr_dist)) if ctr_dist[i] <= threshold], dtype=int)
                self.atom_charge[ idx_ok ] = charge_dict[ elem ]                
            else:
                idx_elem = np.array([i for i in range(len(self.symbols)) if self.symbols[i]==elem], dtype=int)
                self.atom_charge[ idx_elem ] = charge_dict[ elem ]
        return
                
    def _override_movable_group(self, movable_group):
        assert type(movable_group) is list
        for group in movable_group:
            assert type(group) is list
            for i in group: assert type(i) is int
        self.movable_group = movable_group
        return

        
