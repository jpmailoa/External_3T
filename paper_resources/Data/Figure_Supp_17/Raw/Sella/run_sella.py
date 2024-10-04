#!/usr/bin/env python3

import numpy as np 
from ase.io import read 
from ase.calculators.nwchem import NWChem 
from ase.calculators.socketio import SocketIOCalculator 
from sella import Sella, Constraints

socket_name = 'nwchem'
atoms = read('Raw_12951679.xyz')
#nwchem = NWChem(dft=dict(xc='B3LYP',maxiter=2000), basis='6-31+G*', charge=1.0)
nwchem = NWChem (
    memory='20480 mb',
    driver={'socket': {'unix': socket_name}}, 
    task='optimize', dft=dict(xc='B3LYP',maxiter=2000), basis='6-31+G*', charge=1.0)

cons = Constraints(atoms)
#atoms.calc = nwchem
#dyn = Sella(atoms, constraints=cons, internal=True, trajectory='test_nwchem.traj')
#dyn.run(1e-3, 50)

with SocketIOCalculator(nwchem, unixsocket=socket_name) as calc:
    atoms.calc = calc
    dyn = Sella(atoms, constraints=cons, internal=True, trajectory='test_nwchem.traj')
    dyn.run(1e-3, 50)
 
from ase.io.trajectory import Trajectory
import ase.io as sio
traj = Trajectory('test_nwchem.traj')
images = [image for image in traj]
sio.write('test_nwchem.xyz', images, format='xyz')
   
