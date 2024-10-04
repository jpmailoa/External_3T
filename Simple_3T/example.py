import ase.io as sio
from ase.calculators.nwchem import NWChem
from simple3T import Simple3T

# Step 1, just create your ASE atoms object from an xyz file
atoms = sio.read('Raw_12951679.xyz')

# Step 2, create your ASE calculator object. For example, if you have NWChem ASE configured you can do:
atoms.calc = NWChem()

# Step 3, configure Simple3T optimizer to use the ASE atoms object with the calculator attached
opt = Simple3T(atoms)

# Step 4, just run your optimization for the desired number of steps

# If you do have the ASE calculator correctly configured on your machine, use it with this command:
# opt.run(100, use_FF=False)

# Assuming you do not have the ASE calculator configured correctly and you simply want to test functionality / installation correctness,
# you can use the built-in force field (parameter extracted from SwissParam) instead, with this command:
opt.run(100, use_FF=True)
