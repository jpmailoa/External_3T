import sys
sys.path.append('./utils/')
# sys.path.append('./utils/Convert_Gromacs_LAMMPS/')
sys.path.append('../utils/Convert_Gromacs_LAMMPS/')
from run_utils import create_model
from run_utils import create_optimizers
from run_utils import run_model
import torch

import ase
import ase.io as sio

class Simple3T:
    def __init__(self, atoms, id='temp_Simple3T'):
        self.model = None
        self.optimizers = None
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.atoms = atoms
        self.xyz_files = self._create_temp_xyz_files(atoms, id)
        return

    # right now this function assumes that there is only one molecule in the ASE atoms object
    # the list return format is left for future upgrade supporting multiple molecules
    def _create_temp_xyz_files(self, atoms, id):
        xyz_file = id + '.xyz'
        sio.write(xyz_file, atoms, format='xyz')
        return [xyz_file]
    
    def run(self, steps, use_FF=False, out_tag='3T', print_freq=1):
        self.model = create_model(self.xyz_files, self.atoms, 'FF', base_model=self.model).to(self.device)
        self.optimizers = create_optimizers(self.model)
        if use_FF:
            self.model.change_mode('FF')
        else:
            self.model.change_mode('ASE_CALC')
        run_model(self.model, self.optimizers, steps, out_tag=out_tag, print_freq=print_freq)
        return
