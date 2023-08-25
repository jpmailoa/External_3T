import glob
import ase
import ase.io as sio

example_dir = '../0x20c7ae974558a91c'


## Trajectory
FF_files = glob.glob(example_dir + '/FF_step*.xyz')
n = len(FF_files)
FF_files = [example_dir + '/FF_step'+str(i)+'.xyz' for i in range(1,n+1)]
VASP_files = glob.glob(example_dir + '/VASP_step*.xyz')
n = len(VASP_files)
VASP_files = [example_dir + '/VASP_step'+str(i)+'.xyz' for i in range(1,n+1)]

contents = []
contents += [open(FF_file,'r').read() for FF_file in FF_files]
contents += [open(VASP_file,'r').read() for VASP_file in VASP_files]

contents = ''.join([content for content in contents])
open(example_dir + '/all_step.xyz','w').write(contents)
atom_frames = sio.read(example_dir + '/all_step.xyz', index=':')

## Lattice
POSCAR_file = glob.glob(example_dir + '/*.vasp')
assert len(POSCAR_file) == 1
POSCAR_file = POSCAR_file[0]
cell = sio.read(POSCAR_file, format='vasp').cell

## Assign the cells
for atom_frame in atom_frames:
    atom_frame.cell = cell
    atom_frame.pbc = [True, True, True]

## Export last frame
sio.write(example_dir + '/last_step.xyz', atom_frames[-1], format='xyz')
