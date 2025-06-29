import sys, os
import glob, subprocess
import time
from rdkit import Chem
from rdkit.Chem.Lipinski import RotatableBondSmarts
from GL_data import data
from xyz2mol import xyz2mol, read_xyz_file
from process_molecule_utils import get_rotatable_bond, build_new_rotbond, cleanup_workspace, check_cache, store_cache
import numpy as np
from ase.data import chemical_symbols, atomic_masses

def xyz2mol_get_rotatable_bond(xyzfile, mol2file, outfile):
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
    # we need to split manually written LAMMPS *.data data format file into input file and data file which is
    # compatible with the rest of our code (originally written for SwissParam Gromacs-LAMMPS conversion)

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
                    words = [words[0], 'multi/harmonic', words[1], words[2], words[3], words[4], words[5]]
                elif section == 'Improper Coeffs':
                    words = [words[0], 'harmonic', words[1], words[2]]
                else:
                    pass
                data_lines.append( ' '.join(words) )
        with open(data_filename,'w') as f:
            f.write( '\n'.join(data_lines) )
        return

    def _dump_rotbond_file(data_content, rotbond_filename):
        type_elem_dict = {}
        mass_dict = {round(atomic_masses[i]): chemical_symbols[i] for i in range(len(chemical_symbols)) if i!=0}
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
                    assert name.endswith('.data')
                    name = name[:-5]
                    if name == 'plus': charge = 1
                    elif name == 'minus': charge = -1
                    elif 'plus' in name: charge = int(name[4:])
                    elif 'minus' in name: charge = -int(name[5:])
                    else: raise Exception('unallowed *.data filename : '+data_file)
                xyz_lines.append( 'charge=' + str(charge) )
                for j in range(n_atoms):
                    atom_type = atoms_content[j][2]
                    elem = type_elem_dict[ atom_type ]
                    words = [elem, atoms_content[j][4], atoms_content[j][5], atoms_content[j][6]]
                    xyz_lines.append( ' '.join(words) )
        xyz_file = 'LIG.xyz'
        mol2_file = 'LIG.mol2'
        rotbond_file = 'LIG.rotbond'
        with open(xyz_file,'w') as f:
            f.write( '\n'.join(xyz_lines) )
        
        if os.path.isfile(mol2_file): os.system('rm ' + mol2_file)
        if os.path.isfile(rotbond_file): os.system('rm ' + rotbond_file)
        # by default, we use obabel for mol2 and rotbond file generation
        os.system('obabel ' + xyz_file + ' -O ' + mol2_file)
        success = os.path.isfile(mol2_file)
        if success:
            print('Successful with standard obabel mol2 and rotbond generation')
            get_rotatable_bond(mol2_file, rotbond_file)
            success = os.path.isfile(rotbond_file)
        # if obabel fails, we use xyz2mol for mol2 and rotbond file generation
        if not success:
            print('Failure with obabel mol2 or rotbond file generation, try to use xyz2mol+RDKit pathway')
            if os.path.isfile(mol2_file): os.system('rm ' + mol2_file)
            if os.path.isfile(rotbond_file): os.system('rm ' + rotbond_file)
            xyz2mol_get_rotatable_bond(xyz_file, mol2_file, rotbond_file)
            success = os.path.isfile(mol2_file)
            if not success:
                raise Exception('mol2 generation failed using both obabel and xyz2mol. Please check your ' + data_file + ' file for accuracy')
        build_new_rotbond(mol2_file, rotbond_file, converted_ligand_input, converted_ligand_data, converted_ligand_rotbond)
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
    
#convert_molecule('input/manual_PF6.data')
