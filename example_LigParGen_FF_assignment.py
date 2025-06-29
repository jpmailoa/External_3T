import sys
sys.path.append('./utils/')
from process_molecule_from_lmp import convert_molecule

'''
In general, we recommend the usage of automated force field parametrization webserver such as SwissParam.
However, for some difficult molecules (cations, anions, radicals, resonant structures) SwissParam webserver will fail and
the usage of a different webserver such as LigParGen may be preferred. Uploading such molecule's *.xyz or *.mol2 into LigParGen
will generate *.lmp LAMMPS force field data structure file with the following specifications:
unit = real
pair_style = lj
bond_style = harmonic
angle_style = harmonic
dihedral_style = multi/harmonic
improper_style = harmonic

The behavior of the manual force field assignment function of process_molecule_from_lmp.convert_molecule is:
1. The default pathway is to extract rotatable bonds from *.xyz -> *.rotbond (using xyz2mol + RDKit) -> *.pdb (using RDKit) -> *.mol2 (using obabel)

The xyz2mol library needs to be given a specific molecule formal charge so that it can successfully create
an RDKit mol object with the correct rotatable bond. We handle this by including the charge as the last part of the name:
- CO3_minus2.lmp --> formal charge = -2
- NH4_plus.lmp --> formal charge = +1
- NH3.lmp --> formal charge = 0
- oEC_radical_minus1_minus2.lmp --> formal charge = -2  (the real oEC radical formal charge is -1, but we need to specify -2 here to let xyz2mol succeed)

If you would like to manually use *.lmp to assign molecule force field, we recommend:
1. Run this example script to generate the FF and rotatable bonds for your *.lmp molecule (will be stored in the cache)
2. Modify the 3T-VASP config file to use your *.lmp file (instead of your previous *.xyz or)
'''

# Filename needs to end with *_minus2.lmp
input_file = 'input/CO3_minus2.lmp'
print('Example conversion of ' + input_file)
mol_data = convert_molecule(input_file)
print(mol_data.__dict__)

# Filename needs to end with *_minus2.lmp even though its radical's formal charge is -1
input_file = 'input/oEC_radical_minus1_minus2.lmp'
print('Example conversion of ' + input_file)
mol_data = convert_molecule(input_file)
print(mol_data.__dict__)
