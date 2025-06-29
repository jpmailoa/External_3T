import sys
sys.path.append('./utils/')
from process_molecule_from_data import convert_molecule

'''
In general, we recommend the usage of automated force field parametrization webserver such as SwissParam or LigParGen.
However, for some difficult molecules (cations, anions, radicals, resonant structures) these webservers will fail and manual
force field parametrization by an expert user may be preferred. This example shows how to perform such manual FF parametrization.

The input file has to be formatted in the LAMMPS data structure and explicitly named *.data with the following specifications:
unit = real
pair_style = lj
bond_style = harmonic
angle_style = harmonic
dihedral_style = multi/harmonic
improper_style = harmonic

The behavior of the manual force field assignment function of process_molecule_from_data.convert_molecule depends on 
the type of cation/anion/radical/resonant molecule being used (because different open source software succeeds/fails on
processing specific molecules).
1. The default pathway is to extract rotatable bonds from *.xyz -> *.mol2 (using obabel) -> *.rotbond (using RDKit)
2. If that fails, the next attempted pathway is to use *.xyz -> *.rotbond (using xyz2mol + RDKit) -> *.pdb (using RDKit) -> *.mol2 (using obabel)

In the case of pathway #1, there is no restriction on file name (in the example below, manual_PF6_minus.data can be named PF6.data, for example)
On the other hand, in the case of pathway #2, xyz2mol needs to be given a specific molecule formal charge so that it can successfully create
an RDKit mol object with the correct rotatable bond. We handle this by including the charge as the last part of the name:
- CO3_minus2.data --> formal charge = -2
- NH4_plus.data --> formal charge = +1
- NH3.data --> formal charge = 0
- oEC_radical_minus1_minus2.data --> formal charge = -2  (the real oEC radical formal charge is -1, but we need to specify -2 here to let xyz2mol succeed)

If you would like to manually use *.data to assign molecule force field, we recommend:
1. Run this example script to generate the FF and rotatable bonds for your *.data molecule (will be stored in the cache)
2. Modify the 3T-VASP config file to use your *.data file (instead of your previous *.xyz or *.lmp molecule file)
'''

# For this PF6 anion, pathway #1 will be used
# Filename simply needs to end with *.data, but we end it with *_minus.data as an example of best practice
input_file = 'input/manual_PF6_minus.data'
print('Example conversion of ' + input_file)
mol_data = convert_molecule(input_file)
print(mol_data.__dict__)

# For this CO3 anion (resonant structure), pathway #2 will be used
# Filename needs to end with *_minus2.data
input_file = 'input/manual_CO3_minus2.data'
print('Example conversion of ' + input_file)
mol_data = convert_molecule(input_file)
print(mol_data.__dict__)

# For this oEC radical, pathway #2 will be used
# Filename needs to end with *_minus2.data even though its radical's formal charge is -1
input_file = 'input/manual_oEC_radical_minus1_minus2.data'
print('Example conversion of ' + input_file)
mol_data = convert_molecule(input_file)
print(mol_data.__dict__)
