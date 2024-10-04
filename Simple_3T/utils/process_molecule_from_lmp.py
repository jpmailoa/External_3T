import sys, os
import glob, subprocess
import time
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
from GL_data import data
from xyz2mol import xyz2mol, read_xyz_file
from process_molecule_utils import build_new_rotbond, cleanup_workspace, check_cache, store_cache
import numpy as np

def get_rotatable_bond(xyzfile, mol2file, outfile):
    try:
        atoms, charge, coordinates = read_xyz_file(xyzfile, look_for_charge=True)
        mol = xyz2mol(atoms, coordinates, charge=charge)
        assert len(mol) == 1
        mol = mol[0]
        Chem.rdmolfiles.MolToPDBFile(mol, 'LIG.pdb')
        os.system('obabel -ipdb LIG.pdb -O ' + mol2file)
        rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
        rot_atom_pairs = [(x[0]+1, x[1]+1) for x in rot_atom_pairs]
        f = open(outfile,'w')
        f.write('id,atom1,atom2,type\n')
        for i,(j,k) in enumerate(rot_atom_pairs):
            f.write('%d,%d,%d,1\n'%(i+1,j,k))
        f.close()
    except AttributeError:
        print(xyzfile)
    return

def split_lmp_data_file(data_file, converted_ligand_input, converted_ligand_data, converted_ligand_rotbond):
    # we need to split ParLigGen's LAMMPS *.lmp data format file into input file and data file which is
    # compatible with the rest of our code (originally written for SwissParam Gromacs-LAMMPS conversion

    def _extract_section(fstream):
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

    def _dump_input_file(input_content, input_filename):
        assert len(input_content) == 1
        assert input_content[0][0] == 'Pair Coeffs'
        pair_coeffs_content = input_content[0][1]
        input_lines = []
        for words in pair_coeffs_content:
            words = ['pair_coeff', words[0], words[0], words[1], words[2]]
            input_lines.append( ' '.join(words) )
        with open(input_filename,'w') as f:
            f.write( '\n'.join(input_lines) )
        return

    def _dump_data_file(title, headers, data_content, data_filename):
        data_lines = []
        data_lines.append( title )
        data_lines += headers
        for i in range(len(data_content)):
            section = data_content[i][0]
            data_lines.append( '' )
            data_lines.append( section )
            data_lines.append( '' )
            n_lines = len(data_content[i][1])
            for j in range(len(data_content[i][1])):
                words = data_content[i][1][j]
                if section in ['Bond Coeffs', 'Angle Coeffs']:
                    words = [words[0], 'harmonic', words[1], words[2]]
                elif section == 'Dihedral Coeffs':
                    opls_coeff = [float(word) for word in words[1:5]]
                    multiharm_coeff = np.array( [0,0,0,0,0], dtype=float )
                    multiharm_coeff += 0.5 * opls_coeff[0] * np.array( [1,1,0,0,0] ) #(1 + cosx)
                    multiharm_coeff += 0.5 * opls_coeff[1] * np.array( [2,0,2,0,0] ) #(1 - cos2x)
                    multiharm_coeff += 0.5 * opls_coeff[2] * np.array( [1,-3,0,4,0] ) #(1 + cos3x)
                    multiharm_coeff += 0.5 * opls_coeff[3] * np.array( [0,0,8,0,-8] ) #(1 - cos4x)
                    multiharm_coeff = [str(coeff) for coeff in multiharm_coeff]
                    words = [words[0], 'multi/harmonic'] + multiharm_coeff
                elif section == 'Improper Coeffs':
                    cvff_coeff = [float(word) for word in words[1:4]]
                    assert (cvff_coeff[1] == -1) and (cvff_coeff[2] == 2)
                    # all cvff so far is of the form (sin(theta))^2, we will approximate it as theta^2 for small theta
                    harm_coeff = np.array( [2*cvff_coeff[0],0] )
                    harm_coeff = [str(coeff) for coeff in harm_coeff]
                    words = [words[0], 'harmonic'] + harm_coeff
                else:
                    pass
                data_lines.append( ' '.join(words) )
        with open(data_filename,'w') as f:
            f.write( '\n'.join(data_lines) )
        return

    def _dump_rotbond_file(data_content, rotbond_filename):
        type_elem_dict = {}
        mass_dict = {1:'H', 7:'Li', 12:'C', 14:'N', 16:'O', 19:'F', 31:'P', 127:'I', 207:'Pb'}
        for i in range(len(data_content)):
            if data_content[i][0] == 'Masses':
                masses_content = data_content[i][1]
                for j in range(len(masses_content)):
                    mass = round( float(masses_content[j][1]) )
                    type_elem_dict[ masses_content[j][0] ] = mass_dict[ mass ]
        xyz_lines = []
        for i in range(len(data_content)):
            if data_content[i][0] == 'Atoms':
                atoms_content = data_content[i][1]
                n_atoms = len(atoms_content)
                xyz_lines.append( str(n_atoms) )
                charge = 0
                name = data_file.split('_')
                if ('plus' in name[-1]) or ('minus' in name[-1]):
                    name = name[-1]
                    assert name.endswith('.lmp')
                    name = name[:-4]
                    if name == 'plus': charge = 1
                    elif name == 'minus': charge = -1
                    elif 'plus' in name: charge = int(name[4:])
                    elif 'minus' in name: charge = -int(name[5:])
                    else: raise Exception('unallowed *.lmp filename : '+data_file)
                xyz_lines.append( 'charge=' + str(charge) )
                for j in range(n_atoms):
                    atom_type = atoms_content[j][2]
                    elem = type_elem_dict[ atom_type ]
                    words = [elem, atoms_content[j][4], atoms_content[j][5], atoms_content[j][6]]
                    xyz_lines.append( ' '.join(words) )
        xyz_file = 'LIG.xyz'
        with open(xyz_file,'w') as f:
            f.write( '\n'.join(xyz_lines) )
        get_rotatable_bond('LIG.xyz', 'LIG.mol2', 'LIG.rotbond')
        build_new_rotbond('LIG.mol2', 'LIG.rotbond', 'LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')
        return
                    
    
    with open(data_file, 'r') as f:
        line = f.readline()
        title = line.strip()
        line = f.readline()
        headers = []
        data_content = []
        input_content = []
        while line:
            words = line.strip().split()
            if len(words) == 0:
                pass
            else:
                full_words = ' '.join(words)
                if full_words in ['Masses','Atoms','Bonds','Angles','Dihedrals','Impropers','Velocities',
                                  'Bond Coeffs','Angle Coeffs','Dihedral Coeffs','Improper Coeffs']:
                    content = _extract_section(f)
                    #print('Found', full_words, ':', len(content))
                    data_content.append( [full_words,content] )
                elif full_words in ['Pair Coeffs']:
                    content = _extract_section(f)
                    input_content.append( [full_words,content] )
                else:
                    headers.append(full_words)
            line = f.readline()

    _dump_input_file(input_content, converted_ligand_input)
    _dump_data_file(title, headers, data_content, converted_ligand_data)
    _dump_rotbond_file(data_content, converted_ligand_rotbond)
    
    return

def convert_molecule(mol_lmp, override=None):
    # Caution, this code is exclusively meant for *.lmp format generated by LigParGen
    # bond_style and angle_style = harmonic, dihedral style = opls, and improper_style = cvff
    # In addition to that, the name of the *.lmp file should specify the charge state. Here are examples:
    # ammonium_plus.lmp
    # CO3_minus2.lmp
    # CH3COO_minus.lmp
    # benzene-1,2-diaminium_plus2.lmp
    mol_data = check_cache(mol_lmp)
    
    if mol_data is None:
        temp_dir = 'workspace'
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        cwd = os.getcwd()
        os.system('cp '+mol_lmp+' '+temp_dir)
        os.chdir(temp_dir)

        full_mol_lmp_path = mol_lmp
        mol_lmp = os.path.split(mol_lmp)[-1]

        split_lmp_data_file(mol_lmp, 'LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')
        mol_data = data('LIG_converted.input', 'LIG_converted.lmp', 'LIG_converted.rotbond')

        cleanup_workspace()

        os.chdir(cwd)
        store_cache(full_mol_lmp_path, mol_data)
        
    return mol_data
    
#convert_molecule('input/CO3_minus2.lmp')
