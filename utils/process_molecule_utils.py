import os
import rdkit
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
import numpy as np
import json, pickle, random

def get_rotatable_bond(infile, outfile):
    try:
        mol = Chem.rdmolfiles.MolFromMol2File(infile, removeHs=False)
        rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
        rot_atom_pairs = [(x[0]+1, x[1]+1) for x in rot_atom_pairs]
        f = open(outfile,'w')
        f.write('id,atom1,atom2,type\n')
        for i,(j,k) in enumerate(rot_atom_pairs):
            f.write('%d,%d,%d,1\n'%(i+1,j,k))
        f.close()
    except AttributeError:
        print(infile)
    return

def build_new_rotbond(lig_ori, lig_rotbond, converted_ligand_input, converted_ligand_data, converted_ligand_rotbond):
    lig_data = data(converted_ligand_input, converted_ligand_data)
    new_lig_pos = lig_data.atom_pos
    old_lig_pos = []
    with open(lig_ori, 'r') as f:
        line = f.readline()
        while line:
            words = line.strip().split()
            if len(words)==9:
                old_lig_pos.append([float(i) for i in words[2:5]])
            line = f.readline()
        old_lig_pos = np.array(old_lig_pos)
        if not old_lig_pos.shape[0] == new_lig_pos.shape[0]:
            raise Exception('Unmatched ligand atom count for rotatable bond rearrangement')
    n_atoms = old_lig_pos.shape[0]
    all_dist = np.linalg.norm(np.repeat(old_lig_pos[np.newaxis,:,:], n_atoms, axis=0) -\
                              np.repeat(new_lig_pos[:,np.newaxis,:], n_atoms, axis=1), axis=2)
    old_to_new = np.argmin(all_dist, axis=0)
    content = []
    with open(lig_rotbond,'r') as f:
        content.append(f.readline())
        line = f.readline()
        while line:
            words = line.split(',')
            words[1] = str(old_to_new[ int(words[1])-1 ] + 1)
            words[2] = str(old_to_new[ int(words[2])-1 ] + 1)
            content.append( ','.join(words) )
            line = f.readline()
    with open(converted_ligand_rotbond,'w') as f:
        for line in content:
            f.write(line)
    return

def cleanup_workspace():
    lig_ff = 'LIG_ff'
    if os.path.isdir(lig_ff):
        os.system('rm -r '+lig_ff)
    files = os.listdir()
    for file in files:
        os.system('rm '+file)
    return

def check_cache(mol_xyz):
    if not os.path.isdir('cache'): os.mkdir('cache')
    cache_index = 'cache/index.json'
    if not os.path.isfile(cache_index):
        return None
    index = json.load(open(cache_index,'r'))
    if mol_xyz in index['xyz_to_pkl']:
        mol_pkl = os.path.join('cache', index['xyz_to_pkl'][mol_xyz])
        pkl_out = pickle.load(open(mol_pkl, 'rb'))
        return pkl_out
    else:
        return None

def store_cache(mol_xyz, mol_data):
    cache_index = 'cache/index.json'
    if not os.path.isfile(cache_index):
        index = {'xyz_to_pkl':{}, 'pkl_ids':[]}
    else:
        index = json.load(open(cache_index,'r'))
    pkl_ids = index['pkl_ids']
    xyz_to_pkl = index['xyz_to_pkl']
    if mol_xyz in xyz_to_pkl:
        # update the cache, although this will not be why we usually call this function
        old_pkl = os.path.join('cache', xyz_to_pkl[mol_xyz])
        pkl_ids.remove(xyz_to_pkl[mol_xyz])
        os.system('rm '+old_pkl)
        del xyz_to_pkl[mol_xyz]
    duplicate = True
    while duplicate:
        pkl_id = str(random.randint(0, sys.maxsize))+'.pkl'
        if not pkl_id in pkl_ids:
            duplicate = False
    pkl_ids.append(pkl_id)
    xyz_to_pkl[mol_xyz] = pkl_id
    pickle.dump(mol_data, open( os.path.join('cache', pkl_id), 'wb' ))
    json.dump(index, open(cache_index,'w'))
    return