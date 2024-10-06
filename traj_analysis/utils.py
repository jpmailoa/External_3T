import glob, os, re, sys
import json, copy, time
from itertools import combinations, cycle
from _ctypes import PyObj_FromPtr
from collections import defaultdict
from functools import partial
import ase
import ase.io as sio
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from networkx.algorithms import bipartite
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from io import StringIO
from joblib import Parallel, delayed
import rdkit
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolTransforms
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.drawOptions.annotationFontScale = 0.9 # default 0.5
PERIODIC_TABLE = Chem.GetPeriodicTable()
RDBONDORDER = {
    1: Chem.BondType.SINGLE,
    1.5: Chem.BondType.AROMATIC,
    "ar": Chem.BondType.AROMATIC,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}
# add string version of the key for each bond
RDBONDORDER.update({str(key): value for key, value in RDBONDORDER.items()})

old_stdout = sys.stdout


def _infer_bo_and_charges(mol):
    """Infer bond orders and formal charges from a molecule.

    Since most MD topology files don't explicitly retain information on bond
    orders or charges, it has to be guessed from the topology. This is done by
    looping over each atom and comparing its expected valence to the current
    valence to get the Number of Unpaired Electrons (NUE).
    If an atom has a negative NUE, it needs a positive formal charge (-NUE).
    If two neighbouring atoms have UEs, the bond between them most
    likely has to be increased by the value of the smallest NUE.
    If after this process, an atom still has UEs, it needs a negative formal
    charge of -NUE.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.RWMol
        The molecule is modified inplace and must have all hydrogens added

    Notes
    -----
    This algorithm is order dependant. For example, for a carboxylate group
    R-C(-O)-O the first oxygen read will receive a double bond and the other
    one will be charged. It will also affect more complex conjugated systems.
    """

    for atom in sorted(mol.GetAtoms(), reverse=True,
                       key=lambda a: _get_nb_unpaired_electrons(a)[0]):
        # get NUE for each possible valence
        nue = _get_nb_unpaired_electrons(atom)
        # if there's only one possible valence state and the corresponding
        # NUE is negative, it means we can only add a positive charge to
        # the atom
        if (len(nue) == 1) and (nue[0] < 0):
            atom.SetFormalCharge(-nue[0])
            mol.UpdatePropertyCache(strict=False)
        # go to next atom if above case or atom has no unpaired electron
        if (len(nue) == 1) and (nue[0] <= 0):
            continue
        else:
            neighbors = sorted(atom.GetNeighbors(), reverse=True,
                               key=lambda a: _get_nb_unpaired_electrons(a)[0])
            # check if one of the neighbors has a common NUE
            for i, na in enumerate(neighbors, start=1):
                # get NUE for the neighbor
                na_nue = _get_nb_unpaired_electrons(na)
                # smallest common NUE
                common_nue = min(
                    min([i for i in nue if i >= 0], default=0),
                    min([i for i in na_nue if i >= 0], default=0)
                )
                # a common NUE of 0 means we don't need to do anything
                if common_nue != 0:
                    # increase bond order
                    bond = mol.GetBondBetweenAtoms(
                        atom.GetIdx(), na.GetIdx())
                    order = common_nue + 1
                    bond.SetBondType(RDBONDORDER[order])
                    mol.UpdatePropertyCache(strict=False)
                    if i < len(neighbors):
                        # recalculate nue for atom
                        nue = _get_nb_unpaired_electrons(atom)

            # if the atom still has unpaired electrons
            nue = _get_nb_unpaired_electrons(atom)
            nue = min([i for i in nue if i >= 0], default=0)
            if nue > 0:
                # transform it to a negative charge
                atom.SetFormalCharge(-nue)
                atom.SetNumRadicalElectrons(0)
                mol.UpdatePropertyCache(strict=False)


def _get_nb_unpaired_electrons(atom):
    """Calculate the number of unpaired electrons (NUE) of an atom

    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
        The atom for which the NUE will be computed

    Returns
    -------
    nue : list
        The NUE for each possible valence of the atom
    """
    expected_vs = PERIODIC_TABLE.GetValenceList(atom.GetAtomicNum())
    current_v = atom.GetTotalValence() - atom.GetFormalCharge()
    return [v - current_v for v in expected_vs]


def _rebuild_conjugated_bonds(mol, max_iter=200):
    """Rebuild conjugated bonds without negatively charged atoms at the
    beginning and end of the conjugated system

    Depending on the order in which atoms are read during the conversion, the
    :func:`_infer_bo_and_charges` function might write conjugated systems with
    a double bond less and both edges of the system as anions instead of the
    usual alternating single and double bonds. This function corrects this
    behaviour by using an iterative procedure.
    The problematic molecules always follow the same pattern:
    ``anion[-*=*]n-anion`` instead of ``*=[*-*=]n*``, where ``n`` is the number
    of successive single and double bonds. The goal of the iterative procedure
    is to make ``n`` as small as possible by consecutively transforming
    ``anion-*=*`` into ``*=*-anion`` until it reaches the smallest pattern with
    ``n=1``. This last pattern is then transformed from ``anion-*=*-anion`` to
    ``*=*-*=*``.
    Since ``anion-*=*`` is the same as ``*=*-anion`` in terms of SMARTS, we can
    control that we don't transform the same triplet of atoms back and forth by
    adding their indices to a list.
    The molecule needs to be kekulized first to also cover systems
    with aromatic rings.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.RWMol
        The molecule to transform, modified inplace
    max_iter : int
        Maximum number of iterations performed by the function
    """
    mol.UpdatePropertyCache(strict=False)
    Chem.Kekulize(mol)
    pattern = Chem.MolFromSmarts("[*-;!O]-[*+0]=[*+0]")
    # number of unique matches with the pattern
    n_matches = len(set([match[0]
                         for match in mol.GetSubstructMatches(pattern)]))
    if n_matches == 0:
        # nothing to standardize
        return
    # check if there's an even number of anion-*=* patterns
    elif n_matches % 2 == 0:
        end_pattern = Chem.MolFromSmarts("[*-;!O]-[*+0]=[*+0]-[*-]")
    else:
        # as a last resort, the only way to standardize is to find a nitrogen
        # that can accept a double bond and a positive charge
        # or a carbonyl that will become an enolate
        end_pattern = Chem.MolFromSmarts(
            "[*-;!O]-[*+0]=[*+0]-[$([#7;X3;v3]),$([#6+0]=O)]")
    backtrack = []
    for _ in range(max_iter):
        # simplest case where n=1
        end_match = mol.GetSubstructMatch(end_pattern)
        if end_match:
            # index of each atom
            anion1, a1, a2, anion2 = end_match
            term_atom = mol.GetAtomWithIdx(anion2)
            # [*-]-*=*-C=O
            if term_atom.GetAtomicNum() == 6 and term_atom.GetFormalCharge() == 0:
                for neighbor in term_atom.GetNeighbors():
                    bond = mol.GetBondBetweenAtoms(anion2, neighbor.GetIdx())
                    if neighbor.GetAtomicNum() == 8 and bond.GetBondTypeAsDouble() == 2:
                        bond.SetBondType(Chem.BondType.SINGLE)
                        neighbor.SetFormalCharge(-1)
            else:
                # [*-]-*=*-N
                if term_atom.GetAtomicNum() == 7 and term_atom.GetFormalCharge() == 0:
                    end_charge = 1
                # [*-]-*=*-[*-]
                else:
                    end_charge = 0
                mol.GetAtomWithIdx(anion2).SetFormalCharge(end_charge)
            # common part of the conjugated systems: [*-]-*=*
            mol.GetAtomWithIdx(anion1).SetFormalCharge(0)
            mol.GetBondBetweenAtoms(anion1, a1).SetBondType(
                Chem.BondType.DOUBLE)
            mol.GetBondBetweenAtoms(a1, a2).SetBondType(Chem.BondType.SINGLE)
            mol.GetBondBetweenAtoms(a2, anion2).SetBondType(
                Chem.BondType.DOUBLE)
            mol.UpdatePropertyCache(strict=False)

        # shorten the anion-anion pattern from n to n-1
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            # check if we haven't already transformed this triplet
            for match in matches:
                # sort the indices for the comparison
                g = tuple(sorted(match))
                if g in backtrack:
                    # already transformed
                    continue
                else:
                    # take the first one that hasn't been tried
                    anion, a1, a2 = match
                    backtrack.append(g)
                    break
            else:
                anion, a1, a2 = matches[0]
            # charges
            mol.GetAtomWithIdx(anion).SetFormalCharge(0)
            mol.GetAtomWithIdx(a2).SetFormalCharge(-1)
            # bonds
            mol.GetBondBetweenAtoms(anion, a1).SetBondType(
                Chem.BondType.DOUBLE)
            mol.GetBondBetweenAtoms(a1, a2).SetBondType(Chem.BondType.SINGLE)
            mol.UpdatePropertyCache(strict=False)
            # start new iteration
            continue

        # no more changes to apply
        return

    # reached max_iter
    warnings.warn("The standardization could not be completed within a "
                  "reasonable number of iterations")


def ase2rdkit(atoms, charge=None, ignore_bondorder=False):
    #old_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer
    sio.write('-', atoms, format='xyz')
    xyz_block = buffer.getvalue()
    sys.stdout = old_stdout
    rdkitmol = Chem.rdmolfiles.MolFromXYZBlock(xyz_block)
    conn_mol = Chem.Mol(rdkitmol)
    conn_mol.AddConformer(conn_mol.GetConformer(0))
    rdMolTransforms.CanonicalizeConformer(conn_mol.GetConformer(0))
    rdDetermineBonds.DetermineConnectivity(conn_mol)
    if not isinstance(charge, type(None)):
        infer_bo_charge = False
    else:
        infer_bo_charge = True
    if infer_bo_charge and (3 not in atoms.numbers):
        conn_mol.UpdatePropertyCache(strict=False)
        _infer_bo_and_charges(conn_mol)
        #_rebuild_conjugated_bonds(conn_mol, max_iter=200)
        Chem.SanitizeMol(conn_mol)
        return conn_mol
    # except Li and PF6- (rdkit not support)
    if (3 not in atoms.numbers) and (str(atoms.symbols)!='PF6'):
        if not ignore_bondorder:
            rdDetermineBonds.DetermineBonds(conn_mol, charge=charge)
    return conn_mol


def parse_bonds(atoms, charge=None, ignore_bondorder=False):
    radius = 1.20
    cutoffs = radius * covalent_radii[atoms.numbers]
    cutoffs[(atoms.numbers == 3).nonzero()[0]] = 0
    nl = NeighborList(cutoffs=cutoffs, skin=0.0, self_interaction=False)
    nl.update(atoms)
    rdkitmol = ase2rdkit(atoms, charge, ignore_bondorder)

    bondpairs = []
    for a in range(len(atoms)):
        indices, offsets = nl.get_neighbors(a)
        _tmp = []
        for a2, offset in zip(indices, offsets):
            bond = rdkitmol.GetBondBetweenAtoms(a, int(a2))
            if hasattr(bond, 'GetBondType'):
                bond_type = bond.GetBondType().name
            else:
                bond_type = 'SINGLE_M'
            _tmp.append((a, int(a2), offset, bond_type))
        bondpairs.extend(_tmp)
    return bondpairs, rdkitmol


def parse_to_graph(atoms, bondpairs, node_idxs=None):
    G = nx.DiGraph()
    if isinstance(node_idxs, type(None)):
        G.add_nodes_from(np.arange(atoms.get_global_number_of_atoms()))
        for (a, a2, offset, bondtype) in bondpairs:
            G.add_edge(a, a2, bondtype=bondtype)
    else:
        G.add_nodes_from(node_idxs)
        for (a, a2, offset, bondtype) in bondpairs:
            G.add_edge(node_idxs[a], node_idxs[a2], bondtype=bondtype)
    UG = G.to_undirected()
    return UG


def ase2graph(atoms, charge=None, ignore_bondorder=False, node_idxs=None):
    bondpairs, rdkitmol = parse_bonds(atoms, charge, ignore_bondorder)
    UG = parse_to_graph(atoms, bondpairs, node_idxs)
    return UG, rdkitmol


def xyz2rdkit(xyz_f, charge=None, ignore_bondorder=False, node_idxs=None):
    atoms = sio.read(xyz_f, index='-1')
    UG, rdkitmol = ase2graph(atoms, charge, ignore_bondorder, node_idxs)
    return atoms, UG, rdkitmol


def get_improper_idx(rdkitmol):
    ANs = chemical_symbols
    improper_idx, improper_AN = [], []
    for atom in rdkitmol.GetAtoms():
        neighbors = [(x.GetIdx(), x.GetSymbol())
                         for x in atom.GetNeighbors()]
        neighbors.sort(key=lambda k: ANs.index(k[1]), reverse=True)
        if len(neighbors) == 3:
            neighbor_idxs = [x for x in neighbors 
                             if x[1] in ['H', 'C', 'O']]
            comb = list(combinations(neighbor_idxs, 3))
            idx, aN = atom.GetIdx(), atom.GetSymbol()
            for x in comb:
                atom_comb = ''.join([y[1] for y in x])
                if aN=='C' and atom_comb in ['OOO', 'OCH']: # C->OOO, C->OCH
                    improper_idx.append([idx]+[y[0] for y in x])
                    improper_AN.append([aN]+[y[1] for y in x])
    return np.array(improper_idx), np.array(improper_AN)


def calculate_improper_dihedral_angle(rdkitmol, return_idx_AN=False):
    improper_idx, improper_AN = get_improper_idx(rdkitmol)
    if len(improper_idx)==0: 
        if return_idx_AN:
            return 0, improper_idx, improper_AN
        else:
            return 0
    atom_pos = rdkitmol.GetConformer(0).GetPositions()
    v12 = atom_pos[improper_idx[:,0] ] - atom_pos[improper_idx[:,1] ]
    v32 = atom_pos[improper_idx[:,2] ] - atom_pos[improper_idx[:,1] ]
    v43 = atom_pos[improper_idx[:,3] ] - atom_pos[improper_idx[:,2] ]
    v123 = np.cross(v12,v32, axis=1)
    v234 = np.cross(-v32,v43, axis=1)
    temp1 = np.sum(v123*v234, axis=1)
    temp2 = np.linalg.norm(v123, axis=1) * np.linalg.norm(v234, axis=1)
    d_cos = temp1 / temp2
    #d_cos = np.clip(d_cos, a_min=-0.999999, a_max=0.999999)
    degree = np.arccos(d_cos)*180/np.pi
    if return_idx_AN:
        return degree, improper_idx, improper_AN
    else:
        return degree


def dict_update(raw, new):
    dict_update_iter(raw, new)
    dict_add(raw, new)


def dict_update_iter(raw, new):
    for key in raw:
        if key not in new.keys():
            continue
        if isinstance(raw[key], dict) and isinstance(new[key], dict):
            dict_update(raw[key], new[key])
        elif isinstance(raw[key], list) and isinstance(new[key], list):
            if key == 'categories':
                raw[key] += [item for item in new[key] if item not in raw[key]]
            elif key == 'images':
                img_lib = [item['id'] for item in raw[key]]
                raw[key] += [item for item in new[key] if item['id'] not in img_lib]
            else:
                raw[key] += new[key]
        else:
            raw[key] = new[key]


def dict_add(raw, new):
    update_dict = {}
    for key in new:
        if key not in raw.keys():
            update_dict[key] = new[key]
    raw.update(update_dict)


class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, NoIndent):
            return (self.FORMAT_SPEC.format(id(obj)))
        else:
            return super(MyEncoder, self).default(obj)

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)
        return json_repr


def get_rxn_statistics_df(rxn_history, config_id, save_fn=None):
    #sys.stdout = old_stdout
    idx = next(iter(rxn_history))
    mol_name = list(rxn_history[idx]['RXN_CNT']['init'])
    mol_name_wto_Li = [x for x in mol_name if x!='Li']
    N1, N2 = len(mol_name), len(mol_name_wto_Li)
    # 'rxn': break bonds or form new bond; 'rdc': geometry distortion 
    columns = ['idx'] + ['origin_'+x for x in mol_name] + \
                ['rxn_'+x for x in mol_name_wto_Li] + \
                ['rdc_'+x for x in mol_name_wto_Li] 
    data = np.zeros((len(rxn_history), 1+N1+2*N2)).astype(int)
    rxn_sets, rxn_bond_sets, rdc_sets = {}, {}, {}
    rxn_sets2, rdc_sets2 = {}, {}
    for i, (k, rxn) in enumerate(rxn_history.items()):
        data[i][0] = int(k)
        rxn_sets[int(k)], rxn_bond_sets[int(k)], rdc_sets[int(k)] = set(), set(), set()
        rxn_sets2[int(k)], rdc_sets2[int(k)] = set(), set()
        for x in rxn['RXNs']:
            src = sorted(x['src'])
            dst = sorted(y.split('-')[-1] for y in x['dst'])
            dst2 = sorted('-'.join(y.split('-')[-2:]) for y in x['dst'])
            if len(dst)==1 and len(src)==1 and '_reduction' in dst[0]:
                rdc_sets[int(k)].add('+'.join(src) + " => " + '+'.join(dst))
                rdc_sets2[int(k)].add('+'.join(src) + " => " + '+'.join(dst2))
            else:
                rxn_sets[int(k)].add('+'.join(src) + " => " + '+'.join(dst))
                rxn_sets2[int(k)].add('+'.join(src) + " => " + '+'.join(dst2))
                rxn_bond_sets[int(k)].add(tuple(x['changed_bonds']))

        for j, name in enumerate(mol_name+mol_name_wto_Li+mol_name_wto_Li):
            if j < N1:
                data[i][j+1] = rxn['RXN_CNT']['init'][name]
            elif N1 <= j < N1+N2:
                if name in rxn['RXN_CNT']['rxn_changed']:
                    data[i][j+1] = rxn['RXN_CNT']['rxn_changed'][name]
            elif N1+N2 <= j < N1+2*N2:
                if name in rxn['RXN_CNT']['rdc_changed']:
                    data[i][j+1] = rxn['RXN_CNT']['rdc_changed'][name]
    df = pd.DataFrame(data, columns=columns)
    df['rxn'] = list(rxn_sets.values())
    df['rdc'] = list(rdc_sets.values())
    df['rxn_wt_pid'] = list(rxn_sets2.values())
    df['rdc_wt_pid'] = list(rdc_sets2.values())
    df['rxn_changed_bonds'] = list(rxn_bond_sets.values())
    df.insert(0, 'config_id', config_id)
    df.sort_values(by=['config_id', 'idx'], inplace=True)
    if save_fn:
        df.to_csv(save_fn)
    return df


def combine_rxn_statistics_df(RXNs, save_fn='rxn_statistics_all.csv'):
    df = pd.DataFrame()
    for config_id, rxn_history in RXNs.items():
        _save_fn = save_fn[:-7]+'%s.csv'%config_id if isinstance(
            save_fn, type(None)) else None
        _df = get_rxn_statistics_df(rxn_history, config_id, save_fn=_save_fn)
        df = pd.concat([df, _df], ignore_index=True)
    df.fillna(0, inplace=True)
    df.sort_values(by=['config_id', 'idx'], inplace=True)
    df.reset_index(drop=True)
    if save_fn:
        df.to_csv(save_fn)
    return df


def get_total_charge(rdkitmol):
    charge = 0
    for atom in rdkitmol.GetAtoms():
        charge += atom.GetFormalCharge()
    return charge


def get_charge_dict(rxn_coll):
    assert hasattr(rxn_coll, 'TPL_rdkitmols')
    charge_dict = {}
    TPL_Names = list(rxn_coll.TPL_CNT.keys())
    for i, rdkitmol in enumerate(rxn_coll.TPL_rdkitmols):
        charge = get_total_charge(rdkitmol)
        if charge != 0:
            charge_dict[TPL_Names[i]] = charge
    return charge_dict


def parse_plt_label(name, ncharge):
    if '_minus' in name or '_plus' in name:
        name = name.split('_')[0]
    if ncharge == 0: return '', '', name
    pom = '+' if ncharge>=0 else '-'
    t = '' if ncharge in [1, -1] else '%d'%abs(ncharge)
    name = re.sub(r'(\d+)', r'$_{\1}$', name)
    name = r'%s$^{%s\rm{%s}}$'%(name, t, pom.replace(r"-", u"\u2212"))
    name = name.replace(r'$$', '')
    return pom, t, name


def plot_rxn_changed_mols(df, charge_dict={}, reverse=False):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams['mathtext.default'] = 'regular'
    a = 0.8
    base_dict = {'Li': 1, 'PF6': -1}
    _marker = ('o', '^', 's', 'D', '*', 'v', '>', '<')
    assert isinstance(charge_dict, dict)
    charge_dict.update(base_dict)
    df = df.copy()
    n_excess_Li, xlabel = 0, ''
    for col in df.columns:
        if 'origin_' in col and col[7:] in charge_dict:
            nc = charge_dict[col[7:]]
            n_excess_Li += df[col]*nc
            pom, t, l = parse_plt_label(col[7:], nc)
            pom = ' %s '%pom
            t = t+'*' if t!='' else t
            if pom == ' - ':
                xlabel += '%s%sn[%s]'%(pom, t, l)
            else:
                xlabel = '%s%sn[%s]'%(pom, t, l)+xlabel
    if reverse:
        xlabel = xlabel.replace(' + ', ' = ').replace(
                        ' - ', ' + ').replace(' = ', ' - ')
        n_excess_Li = -n_excess_Li
    xlabel_tmp = xlabel.split(' + ', 1)
    xlabel_tmp.append(xlabel_tmp.pop(0))
    xlabel = ''.join(xlabel_tmp)
    xlabel = xlabel.replace(' - ', ' \N{MINUS SIGN} ')
    xticks = n_excess_Li.values
    xl, xr = xticks.min(), xticks.max()
    N_ = xr - xl
    for i in range(2, 5, 1):
        if N_//i in [5, 6, 7, 8]:
            xticks = np.arange(xl, xr+i, i)
            break
    df['excess_Li'] = n_excess_Li
    df.drop(columns=['config_id', 'rxn', 'rdc', 'rxn_wt_pid', 
                     'rdc_wt_pid', 'rxn_changed_bonds'], inplace=True, errors='ignore')
    gb = df.groupby(['excess_Li']).agg(['mean', partial(np.std, ddof=0)])
    x = gb.index.values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    marker = cycle(_marker) 
    for col in df.columns:
        if 'rxn_' in col:
            l = col[4:]
            nc = charge_dict[l] if l in charge_dict else 0
            pom, t, nl = parse_plt_label(l, nc)
            ax1.errorbar(x, gb[col]['mean'], gb[col]['std'], ls='--', 
                 marker=next(marker), alpha=a, label=nl, capsize=3)
    handles, labels = ax1.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    ax1.legend(handles, labels)
    ax1.set_title('Reaction', fontsize=14)
    ax1.set_xlabel(xlabel, fontsize=13)
    ax1.set_xticks(xticks)
    ax1.set_ylabel('Avg. Molecule Count', fontsize=13)

    marker = cycle(_marker) 
    for col in df.columns:
        if 'rdc_' in col:
            l = col[4:]
            nc = charge_dict[l] if l in charge_dict else 0
            pom, t, nl = parse_plt_label(l, nc)
            ax2.errorbar(x, gb[col]['mean'], gb[col]['std'], ls='--', 
                 marker=next(marker), alpha=a, label=nl, capsize=3)
    handles, labels = ax2.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    ax2.legend(handles, labels)
    title = 'Reduction' if not reverse else 'Oxidation'
    ax2.set_title(title, fontsize=14)
    ax2.set_xlabel(xlabel, fontsize=13)
    ax2.set_xticks(xticks)
    ax2.set_ylabel('Avg. Molecule Count', fontsize=13)
    plt.tight_layout()
    plt.show()


def run_cleaned_rxn(df, col='rxn'):
    s = df[col].apply(lambda x: set([y for y in x]))
    flag = [[False for x in s.iloc[-1]] for i in range(len(df))]
    rxn_lib = set()
    first_run = True
    src_all = set(y for x in s.iloc[-1] for y in x.split(' => ')[0].split('+'))
    for i, rxn in enumerate(s.iloc[-1].copy()):
        v_rxn_idx = (set((rxn,)) & s).values.nonzero()[0][0]
        flag[-1][i] = False if v_rxn_idx == s.index.max() else True
        src = set(rxn.split(' => ')[0].split('+'))
        for j in range(len(s)-1):
            for _rxn in s[j].copy():
                if (j > v_rxn_idx) and (set(_rxn.split(
                    ' => ')[0].split('+')) & src) != set():
                    s[j].remove(_rxn)
                if first_run:
                    # _src = set(_rxn.split(' => ')[0].split('+'))
                    # if (_rxn in s[j]) and (src_all & _src) == set():
                    #     s[j].remove(_rxn)
                    if _rxn in rxn_lib:
                        if _rxn in s[j]:
                            s[j].remove(_rxn)
                    else:
                        rxn_lib.add(_rxn)
        first_run = False
    df[col+'_cleaned'] = s
    df[col+'_is_last_frame'] = flag
    return df


def run_rxn_diff_df(df):
    df = run_cleaned_rxn(df, col='rxn')
    df = run_cleaned_rxn(df, col='rdc')
    col_list = ['rxn', 'rdc', 'rxn_cleaned', 'rdc_cleaned']
    for cn in col_list[:]:
        tmp = pd.DataFrame()
        tmp['forward'] = df[cn].diff(periods=1).fillna("").apply(set)
        tmp['backward'] = df[cn].diff(periods=-1).fillna("").apply(set).apply(lambda x: set(
            [y.replace('=>','<=') for y in x])).shift(1).fillna("").apply(set)
        tmp['bilateral_diff'] = tmp.apply(lambda x:x[0] | x[1], axis=1)
        tmp[cn+'_src'] = df[cn].apply(lambda rxn: set([y for x in rxn \
                            for y in re.split(' => | <= ', x, 1)[0].split('+') if y != '']))
        tmp[cn+'_bi_diff_src'] = tmp['bilateral_diff'].apply(lambda rxn: set([y for x in rxn \
                            for y in re.split(' => | <= ', x, 1)[0].split('+') if y != '']))
        def get_fleeet_rxn(x):
            y = x[cn+'_bi_diff_src']-tmp[cn+'_src'].iloc[-1]
            x[cn+'_transient'] = set(m for m in x['bilateral_diff'] for n in y 
                                    if n in re.split(' => | <= ', m, 1)[0])
            x[cn+'_diff'] = set(m for m in x['forward'] if m not in x[cn+'_transient'])
            return x
        tmp = tmp.apply(get_fleeet_rxn, axis=1)
        df[cn+'_all_diff'] = tmp['forward']
        df[cn+'_all_rev_diff'] = tmp['backward']
        df[cn+'_all_bilateral_diff'] = tmp['bilateral_diff']
        df[cn+'_transient'] = tmp[cn+'_transient']
        df[cn+'_diff'] = tmp[cn+'_diff']
        col_list += [cn+'_diff', cn+'_transient']

    for cn in col_list[:]:
        df[cn+'_src'] = df[cn].apply(lambda rxn: set([y for x in rxn \
                            for y in re.split(' => | <= ', x, 1)[0].split('+') if y != '']))
        df[cn+'_dst'] = df[cn].apply(lambda rxn: set([y for x in rxn \
                            for y in re.split(' => | <= ', x, 1)[1].split('+') if y != '']))
        col_list += [cn+'_src', cn+'_dst']

    for cn in col_list[6:]:
        df[cn+'_cnt'] = df[cn].apply(lambda x: len(x))
        col_list.append(cn+'_cnt')
    return df, col_list


def combine_rxn_diff_df(df_all):
    if isinstance(df_all['rxn'][0], str):
        df_all['rxn'] = df_all['rxn'].apply(eval)
        df_all['rdc'] = df_all['rdc'].apply(eval)
        df_all['rxn_changed_bonds'] = df_all['rxn_changed_bonds'].apply(eval)
    config_ids = df_all.config_id.unique()
    df_new = pd.DataFrame()
    for config_id in config_ids:
        print(config_id, 'start processing...')
        df = df_all[df_all.config_id==config_id]
        df = df[df.idx>=0].copy().reset_index(drop=True)
        df, col_list = run_rxn_diff_df(df)
        df_new = pd.concat([df_new, df])
        print(config_id, 'done!')
    df_new.reset_index(drop=True, inplace=True)
    return df_new


def plot_rxn_cleaned_timestep_hist(df, bins=25, show_fake_rxn=False, xrange=(0, 250)):
    a = 0.7
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    X, L = [], []
    #remove data from last frame
    rxn_idx = [i for i, x, fs in zip(df.idx, df.rxn_cleaned_diff_src, df.rxn_is_last_frame) 
               if x!=set() for y, f in zip(x, fs) if not f]
    X.append(rxn_idx), L.append('rxn')
    if show_fake_rxn:
        #remove data from last frame
        fake_rxn_idx = [i for i, x in zip(df.idx, df.rxn_transient_src) \
                        if x!=set() for y in x]
        X.append(fake_rxn_idx), L.append('transient_rxn')
    ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, zorder=0, stacked=True)
    ax1.set_title('Reaction', fontsize=14)
    ax1.set_xlabel('Time step', fontsize=13)
    ax1.set_ylabel('Frequency', fontsize=13)

    X, L = [], []
    #remove data from last frame
    rdc_idx = [i for i, x, fs in zip(df.idx, df.rdc_cleaned_diff_src, df.rdc_is_last_frame) 
               if x!=set() for y, f in zip(x, fs) if not f]
    X.append(rdc_idx), L.append('rxn')
    if show_fake_rxn:
        #remove data from last frame
        fake_rdc_idx = [i for i, x in zip(df.idx, df.rdc_transient_src) \
                        if x!=set() for y in x]
        X.append(fake_rdc_idx), L.append('transient_rxn')
    ax2.hist(X, bins=bins, range=xrange, label=L, alpha=a, zorder=0, stacked=True)
    ax2.set_title('Reduction', fontsize=14)
    ax2.set_xlabel('Time step', fontsize=13)
    ax2.set_ylabel('Frequency', fontsize=13)
    if show_fake_rxn:
        ax1.legend()
        ax2.legend()
    plt.show()


def plot_rxn_cleaned_timestep_hist_by_mols(df, bins=25, show_fake_rxn=False, 
                                   xrange=(0, 250), charge_dict={}):
    a = 0.7
    base_dict = {'Li': 1, 'PF6': -1}
    assert isinstance(charge_dict, dict)
    charge_dict.update(base_dict)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    mol_name = [x[7:] for x in df.columns if ('origin_' in x) and (x[7:]!='Li')]
    def rxn_mol_cnt(x,  name):
        src = [y.rsplit('_', 1)[0] for y in x.iloc[0]]
        res = {'%s_%s_cnt'%(name,y): src.count(y) for y in mol_name}
        return res

    name = 'rxn_cleaned_diff_src'
    applied_df = df[['rxn_cleaned_diff_src']].apply(rxn_mol_cnt, args=(name,), 
                        axis='columns', result_type='expand')
    df = pd.concat([df, applied_df], axis='columns')
    X, L = [], []
    for mn in mol_name:
        #remove data from last frame
        mn_idx = [i for i, x, fs in zip(df.idx, df['%s_%s_cnt'%(name, mn)], 
            df['rxn_is_last_frame']) if x!=0 for y, f in zip(range(x), fs) if not f]
        nc = charge_dict[mn] if mn in charge_dict else 0
        pom, t, nl = parse_plt_label(mn, nc)
        X.append(mn_idx)
        L.append(nl)
    ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
    ax1.legend()
    ax1.set_title('Reaction', fontsize=14)
    ax1.set_xlabel('Time step', fontsize=13)
    ax1.set_ylabel('Frequency', fontsize=13)

    name = 'rdc_cleaned_diff_src'
    applied_df = df[['rdc_cleaned_diff_src']].apply(rxn_mol_cnt, args=(name,), 
                        axis='columns', result_type='expand')
    df = pd.concat([df, applied_df], axis='columns')
    X, L = [], []
    for mn in mol_name:
        #remove data from last frame
        mn_idx = [i for i, x, fs in zip(df.idx, df['%s_%s_cnt'%(name, mn)], 
            df['rdc_is_last_frame']) if x!=0 for y, f in zip(range(x), fs) if not f]
        nc = charge_dict[mn] if mn in charge_dict else 0
        pom, t, nl = parse_plt_label(mn, nc)
        X.append(mn_idx)
        L.append(nl)
    ax2.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
    ax2.legend()
    ax2.set_title('Reduction', fontsize=14)
    ax2.set_xlabel('Time step', fontsize=13)
    ax2.set_ylabel('Frequency', fontsize=13)
    plt.show()

    if show_fake_rxn:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        name = 'transient_rxn_cleaned_diff_src'
        applied_df = df[['rxn_cleaned_transient_src']].apply(rxn_mol_cnt, args=(name,), 
                            axis='columns', result_type='expand')
        df = pd.concat([df, applied_df], axis='columns')
        X, L = [], []
        for mn in mol_name:
            #remove data from last frame
            mn_idx = [i for i, x in zip(df.idx, df['%s_%s_cnt'%(name, mn)]) \
                      if x!=0 for y in range(x)]
            nc = charge_dict[mn] if mn in charge_dict else 0
            pom, t, nl = parse_plt_label(mn, nc)
            X.append(mn_idx)
            L.append(nl)
        ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
        ax1.legend()
        ax1.set_title('Transient Reaction', fontsize=14)
        ax1.set_xlabel('Time step', fontsize=13)
        ax1.set_ylabel('Frequency', fontsize=13)

        name = 'transient_rdc_cleaned_diff_src'
        applied_df = df[['rdc_cleaned_transient_src']].apply(rxn_mol_cnt, args=(name,), 
                            axis='columns', result_type='expand')
        df = pd.concat([df, applied_df], axis='columns')
        X, L = [], []
        for mn in mol_name:
            #remove data from last frame
            mn_idx = [i for i, x in zip(df.idx, df['%s_%s_cnt'%(name, mn)]) \
                      if x!=0 for y in range(x)]
            nc = charge_dict[mn] if mn in charge_dict else 0
            pom, t, nl = parse_plt_label(mn, nc)
            X.append(mn_idx)
            L.append(nl)
        ax2.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
        ax2.legend()
        ax2.set_title('Transient Reduction', fontsize=14)
        ax2.set_xlabel('Time step', fontsize=13)
        ax2.set_ylabel('Frequency', fontsize=13)
        plt.show()
    return df

