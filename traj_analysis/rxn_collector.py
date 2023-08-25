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

from .utils import ase2graph, calculate_improper_dihedral_angle
from .utils import MyEncoder, NoIndent, dict_update, combine_rxn_statistics_df

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


class RXNCollector(object):
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.config_id = work_dir.split('/')[-1]
        self.working_xyz_name = None
        self.reset_mol_tpl()
        self.preprocess()

    def reset_mol_tpl(self):
        config_json = glob.glob(self.work_dir + '/*.json')
        assert len(config_json) == 1
        config_json = config_json[0]
        blocks = json.load(open(config_json,'r'))
        if type(blocks) is dict:
            blocks = [blocks]
        block = blocks[0]
        assert 'molecule_xyz' in block
        assert 'lattice_poscar' in block

        POSCAR_file = self.work_dir + '/' + block['lattice_poscar']['file'].split('/')[-1]
        Li_atoms = sio.read(POSCAR_file, format='vasp')
        Li_atoms.info['name'] = 'Li'
        self.cell = Li_atoms.cell
        Li_N = Li_atoms.get_global_number_of_atoms()
        self.TPL_MOLs, self.TPL_CNT = [Li_atoms], {'Li': 1}
        self.TPL_NAMES_FULL_LIST = ['Li']
        self.TPL_IDXs = {'Li': np.arange(Li_N)}
        self.TPL_NAtoms_dict = defaultdict(list)
        self.TPL_NAtoms_dict[Li_atoms.get_global_number_of_atoms()].append(0)
        _idx = Li_atoms.get_global_number_of_atoms()
        for tpl_idx, mol_block in enumerate(block['molecule_xyz']):
            fn = mol_block['file'].split('/')[-1]
            TPL_NAME, suffix = fn.split('.')
            if suffix == 'xyz':
                mol_atoms = sio.read(os.path.dirname(self.work_dir) + 
                                     '/useful_items/' + fn, index='-1')
            elif suffix == 'lmp':
                mol_atoms = sio.read(os.path.dirname(self.work_dir) + 
                        '/useful_items/' + fn, format='lammps-data', index='-1')
                new_atom_nums = [np.where(np.abs(atomic_masses-x)<1e-5)[0][0] 
                                 for x in mol_atoms.get_masses()]
                mol_atoms.set_atomic_numbers(new_atom_nums)
            NMol = mol_block['count']
            N = mol_atoms.get_global_number_of_atoms()
            mol_atoms.info['name'] = TPL_NAME
            self.TPL_MOLs.append(mol_atoms)
            self.TPL_NAMES_FULL_LIST += [TPL_NAME+'_'+str(i) for i in range(NMol)]
            self.TPL_CNT[TPL_NAME] = mol_block['count']
            for i in range(NMol):
                self.TPL_IDXs[TPL_NAME+'_'+str(i)] = np.arange(_idx+i*N,_idx+(i+1)*N)
            self.TPL_NAtoms_dict[N].append(tpl_idx+1)
            _idx = _idx+NMol*N
        self.TPL_UGs, self.TPL_rdkitmols = [], []
        for atoms in self.TPL_MOLs:
            UG, rdkitmol = ase2graph(atoms)
            self.TPL_UGs.append(UG)
            self.TPL_rdkitmols.append(rdkitmol)
        self.RXN_History = {}

    def preprocess(self):
        ## Trajectory
        #FF_files = glob.glob(self.work_dir + '/FF_step*.xyz')
        #n = len(FF_files)
        #FF_files = [self.work_dir + '/FF_step'+str(i)+'.xyz' for i in range(1,n+1)]
        VASP_files = sorted(glob.glob(self.work_dir + '/VASP_step*.xyz'))
        n = len(VASP_files)
        VASP_files = [self.work_dir + '/VASP_step'+str(i)+'.xyz' for i in range(1,n+1)]

        contents = []
        #contents += [open(FF_file,'r').read() for FF_file in FF_files]
        contents += [open(VASP_file,'r').read() for VASP_file in VASP_files]

        contents = ''.join([content for content in contents])
        open(self.work_dir + '/all_step.xyz','w').write(contents)

    def read_xyz(self, fname='all_step.xyz'):
        '''if index == ":", it will return a list of the whole xyz frames'''
        if os.path.isfile(fname):
            self.working_xyz_name = fname
        else:
            self.working_xyz_name = self.work_dir + '/' + fname
        atom_frames = sio.read(self.working_xyz_name, index=':')
        if isinstance(atom_frames, ase.atoms.Atoms):
            atom_frames = [atom_frames]
        ## Assign the cells
        for atom_frame in atom_frames:
            atom_frame.cell = self.cell
            atom_frame.pbc = [True, True, True]
        self.atom_frames = atom_frames
        self.refer_atoms = atom_frames[0]
        UG, rdkitmol = ase2graph(self.refer_atoms)
        self.refer_UG = UG
        return atom_frames

    def get_frame(self, atom_frames, index=-1):
        self.target_frame_idx = index
        assert isinstance(index, int)
        atoms = self.atom_frames[index]
        sio.write(self.work_dir + '/working_frame.xyz', atoms, format='xyz')
        self.atoms = atoms
        return atoms

    def write_rxn_analysis_result(self):
        history = copy.deepcopy(self.RXN_History)
        for _i, _rxn in history.items():
            history[_i]['RXN_CNT'] = NoIndent(history[_i]['RXN_CNT'])
            for _j, rxn in enumerate(_rxn['RXNs']):
                history[_i]['RXNs'][_j] = NoIndent(rxn)
            for _j, mol_idx in _rxn['RXN_MOL_IDXs'].items():
                history[_i]['RXN_MOL_IDXs'][_j] = NoIndent(mol_idx)
            for k, v in _rxn['RXN_Graph'].items():
                if type(v) in [dict, tuple]:
                    history[_i]['RXN_Graph'][k] = NoIndent(v)
                elif type(v) == list:
                    for _k, _dict in enumerate(v):
                        history[_i]['RXN_Graph'][k][_k] = NoIndent(_dict)
                else:
                    history[_i]['RXN_Graph'][k] = v
        try:
            json_str = json.dumps(history, indent=2, cls=MyEncoder)
        except:
            print(self.target_frame_idx, history)
            json_str = json.dumps(history, indent=2, cls=MyEncoder)
        with open(self.dst_dir + '/rxn_and_nx_graph_data.json', 'w') as f:
            f.write(json_str)
        if len(history)==1 and -1 in history:
            with open(self.dst_dir + '/last_rxn_and_nx_graph_data.json', 'w') as f:
                f.write(json_str)

    def center_to_box(self, atoms):
        atoms_new = atoms.copy()
        Li_idxs = (atoms.numbers == 3).nonzero()[0]
        not_Li_idxs = (atoms.numbers != 3).nonzero()[0]
        atoms_new.wrap(pretty_translation=True)
        shift = stats.mode((atoms_new.positions - atoms.positions),
                           keepdims=True)[0][0]
        atoms_new.positions[not_Li_idxs] = atoms.positions[not_Li_idxs] + shift
        return atoms_new

    def get_subgraph(self, atoms, UG):
        self.subgraphs = list((UG.subgraph(c) for c in nx.connected_components(UG)))
        changed_sub_atoms, changed_sub_graphs, changed_sub_rdkitmol = [], [], []
        normal_sub_atoms = []
        for i, sg in enumerate(self.subgraphs):
            flag, sub_atoms, sub_UG, sub_rdkitmol = self._is_TPL_MOL(atoms, sg, i)
            if not flag:
                changed_sub_atoms.append(sub_atoms)
                changed_sub_graphs.append(sub_UG)
                changed_sub_rdkitmol.append(sub_rdkitmol)
            else:
                normal_sub_atoms.append(sub_atoms)
        return changed_sub_atoms, changed_sub_graphs, changed_sub_rdkitmol, normal_sub_atoms

    def get_sub_atoms(self, atoms, sub_idxs):
        _numbers = atoms.numbers[sub_idxs]
        _positions = atoms.positions[sub_idxs]
        _positions = _positions - _positions.mean(axis=0)
        return ase.Atoms(numbers=_numbers, positions=_positions)

    def _print_changed_subgraph(self, changed_subgraph):
        for i, sg in enumerate(changed_subgraph):
            if self.print_changed_subgraph > 0:
                print("subgraph {} has {} nodes".format(i, sg.number_of_nodes()))
                print("\tNodes:", sg.nodes())
                if self.print_changed_subgraph == 2:
                    print("\tEdges:", sg.edges())

    def _is_TPL_MOL(self, atoms, sg, idx):
        '''Identify whether a subgraph corresponds to an unchanged TPL mol'''
        edge_match = lambda e1,e2: e1['bondtype'] == e2['bondtype']
        n_nodes = sg.number_of_nodes()
        node_idxs = np.sort(np.array(sg.nodes()))
        sub_atoms = self.get_sub_atoms(atoms, node_idxs)
        sub_atoms.info['name'] = str(sub_atoms.symbols)
        sub_UG, sub_rdkitmol = ase2graph(sub_atoms, node_idxs=node_idxs)
        if n_nodes in self.TPL_NAtoms_dict.keys():
            for TPL_idx in self.TPL_NAtoms_dict[n_nodes]:
                TPL_UG = self.TPL_UGs[TPL_idx]
                GM = nx.isomorphism.GraphMatcher(sub_UG, TPL_UG, edge_match=edge_match)
                if GM.is_isomorphic():
                    d_tpl = calculate_improper_dihedral_angle(self.TPL_rdkitmols[TPL_idx])
                    d_subg = calculate_improper_dihedral_angle(sub_rdkitmol)
                    d_diff = np.abs(d_subg - d_tpl).max()
                    if d_diff<15 or sub_atoms.info['name']=='PF6':
                        sub_atoms.info['name'] = 'nsg%s_'%idx+self.TPL_MOLs[
                            TPL_idx].info['name'] +'-'+sub_atoms.info['name']
                        return True, sub_atoms, sub_UG, sub_rdkitmol
                    else:
                        sub_atoms.info['name'] = sub_atoms.info['name']+'_reduction'
                        return False, sub_atoms, sub_UG, sub_rdkitmol
        # isolated lithium atom
        elif n_nodes==1 and np.array(sg.nodes())[0] in self.TPL_IDXs['Li']:
            return True, sub_atoms, sub_UG, sub_rdkitmol
        return False, sub_atoms, sub_UG, sub_rdkitmol

    def _get_pre_RXNs(self, csa, csg, nsa, target_frame_idx):
        _RXNs = set() # {(src, dst),(src, dst)} like
        TPL_IDXs_key = list(self.TPL_IDXs.keys())
        if self.save_rxn_result: # and target_frame_idx==-1:
            for sa in nsa:
                sa_name = 'frame('+str(target_frame_idx)+')-'+str(sa.info['name'])
                sa_fn = self.dst_dir + '/%s_normal.xyz'%sa_name
                if 'Li' not in sa_name:
                    sio.write(sa_fn, sa, format='xyz')
        for sg_i in range(len(csg)):
            sg = csg[sg_i]
            sg_name = 'frame('+str(target_frame_idx)+')'+ \
                      '-sg'+str(sg_i)+'-'+str(csa[sg_i].info['name'])
            sg_fn = self.dst_dir + '/%s.xyz'%sg_name
            if self.save_rxn_result and not os.path.isfile(sg_fn):
                sio.write(sg_fn, csa[sg_i], format='xyz')
            sg_idx = np.array(sg.nodes())
            sg_idx.sort()
            for tpl_j in range(len(self.TPL_IDXs)):
                tpl_idx = self.TPL_IDXs[TPL_IDXs_key[tpl_j]]
                if sg_idx[-1] < tpl_idx[0]:
                    break
                elif sg_idx[0] > tpl_idx[-1]:
                    continue
                for node_idx in sg_idx:
                    if node_idx in tpl_idx:
                        src_name = self.TPL_NAMES_FULL_LIST[tpl_j]
                        _RXNs.add((src_name, sg_name))
        return _RXNs

    def _get_RXNs_and_RXNGraph(self, _RXNs, csg, target_frame_idx):
        RXN_graph, RXNs, RXN_MOL_IDXs = nx.DiGraph(), [], {}
        RXN_CNT = {'init': {}, 'rxn_changed': defaultdict(int), 
                               'rdc_changed': defaultdict(int)}
        RXN_CNT['init']['Li'] = len(self.TPL_IDXs['Li'])
        for k, v in self.TPL_CNT.items():
            if k != 'Li': RXN_CNT['init'][k] = v
        for (src_name, sg_name) in _RXNs:
            RXN_graph.add_nodes_from([
                        (src_name, {'bipartite': 0, 'type': 'src'}), 
                        (sg_name, {'bipartite': 1, 'type': 'dst'})])
            RXN_graph.add_edge(src_name, sg_name)
        UG = RXN_graph.to_undirected()
        subgraphs = list((UG.subgraph(c) for c in nx.connected_components(UG)))
        for sg in subgraphs:
            _rxn = {'src': [], 'dst': []}
            flag = 'rxn_changed'
            if len(sg.nodes)==2:
                rxn_str = ' | '.join([str(x) for x in sg.nodes])
                if '_reduction' in rxn_str:
                    flag = 'rdc_changed'
            for node in sg.nodes(data=True):
                _rxn[node[1]['type']].append(node[0])
                mol_name = node[0].rsplit('_', 1)[0]
                #mol_name = '_'.join(mol_name[:-1]) if len(mol_name)>=3 else mol_name[0]
                if mol_name in self.TPL_CNT.keys():
                    RXN_CNT[flag][mol_name] += 1
            RXNs.append(_rxn)
        RXN_CNT['rxn_changed'] = dict(RXN_CNT['rxn_changed'])
        RXN_CNT['rdc_changed'] = dict(RXN_CNT['rdc_changed'])
        m = 2 if target_frame_idx < 0 else 1
        RXN_MOL_IDXs = {k: np.array(csg[int(float(k.split('-')[m][2:]))].nodes
                        ).tolist() for rxn in RXNs for k in rxn['dst']}
        return RXN_CNT, RXN_graph, RXNs, RXN_MOL_IDXs

    def rxn_detector(self, atom_frames, target_frame_idx=-1):
        atoms = self.get_frame(atom_frames, target_frame_idx) # --> self.atoms
        UG, rdkitmol = ase2graph(atoms)
        RG1 = nx.difference(self.refer_UG, UG)
        RG2 = nx.difference(UG, self.refer_UG)
        csa, csg, csr, nsa = self.get_subgraph(atoms, UG)
        self.changed_sub_atoms = csa
        self.changed_sub_graphs = csg
        self.changed_sub_rdkitmol = csr
        self.normal_sub_atoms = nsa
        if self.print_changed_subgraph>0:
            self._print_changed_subgraph(csg)

        _RXNs = self._get_pre_RXNs(csa, csg, nsa, target_frame_idx)
        RXN_CNT, RXN_graph, RXNs, RXN_MOL_IDXs = \
            self._get_RXNs_and_RXNGraph(_RXNs, csg, target_frame_idx)
        graph_data = json_graph.node_link_data(RXN_graph) 

        # get break_bonds distance
        eles = atoms.get_chemical_symbols()
        for rxn in RXNs:
            rxn['changed_bonds'] = []
            for (na, nb) in (list(RG1.edges)+list(RG2.edges)):
                a_in_rxn, b_in_rxn = False, False
                for x in rxn['src']:
                    if na in self.TPL_IDXs[x]:
                        na_flag = '%s%d@%s'%(eles[na], na, x)
                        a_in_rxn = True
                    if nb in self.TPL_IDXs[x]:
                        nb_flag = '%s%d@%s'%(eles[nb], nb, x)
                        b_in_rxn = True
                if a_in_rxn and b_in_rxn:
                    dist0 = round(self.refer_atoms.get_distance(na, nb),4)
                    src_flag = (na_flag, nb_flag)

                a_in_rxn, b_in_rxn = False, False
                for x in rxn['dst']:
                    if na in RXN_MOL_IDXs[x]:
                        na_flag = '%s%d@%s'%(eles[na], na, x)
                        a_in_rxn = True
                    if nb in RXN_MOL_IDXs[x]:
                        nb_flag = '%s%d@%s'%(eles[nb], nb, x)
                        b_in_rxn = True
                if a_in_rxn and b_in_rxn:
                    dist1 = round(atoms.get_distance(na, nb),4)
                    dst_flag = (na_flag, nb_flag)
                    rxn['changed_bonds'].append((('%s%d'%(eles[na], na), '%s%d'%(eles[nb], nb)), 
                                                 (dist0, dist1), (src_flag, dst_flag)))

        RXN_result = {'RXN_CNT': RXN_CNT,
                      'RXNs': RXNs,
                      'RXN_MOL_IDXs': RXN_MOL_IDXs,
                      'RXN_Graph': graph_data}
        self.RXN_History[target_frame_idx] = RXN_result
        self.RXN_result = RXN_result
        self.RXN_graph = RXN_graph
        return RXN_result

    def draw_rxn_graph(self, G):
        #G = rxn_collector.RXN_graph
        color_dict = {'src': 'cyan', 'dst': 'lightgreen'}
        colors = [color_dict[node[1]['type']] for node in G.nodes(data=True)]
        src_nodes = [x for rxn in self.RXN_result['RXNs'] for x in rxn['src']]
        mapping = {node: node.replace('-', '\n') for node in G.nodes}
        H = nx.relabel_nodes(G, mapping)
        pos=nx.bipartite_layout(H, src_nodes)
        nx.draw_networkx(H, node_color=colors, pos=pos)
        plt.tight_layout()

    def _process(self, atom_frames, target_frame_idx=-1):
        start_time = time.time()
        RXN_result = self.rxn_detector(atom_frames, target_frame_idx)
        tmp_time = time.time()
        sys.stdout = old_stdout
        if self.print_time_cost: print('Analysis cost: %.4fs' %(tmp_time-start_time))
        return RXN_result

    def process(self, xyz_name='all_step.xyz', target_frame_idxs=[-1], 
                n_jobs=1, verbose=0, print_changed_subgraph=0, 
                save_rxn_result=True, print_time_cost=True):
        start_time = time.time()
        self.n_jobs = n_jobs
        self.print_changed_subgraph = print_changed_subgraph
        self.save_rxn_result = save_rxn_result
        self.print_time_cost = print_time_cost
        self.dst_dir = self.work_dir + '/rxn_analysis_result'
        if save_rxn_result and not os.path.isdir(self.dst_dir):
            os.makedirs(self.dst_dir)

        xyz_name = self.work_dir + '/' + xyz_name
        if not hasattr(self, 'atom_frames') or self.working_xyz_name!=xyz_name:
             self.read_xyz(fname=xyz_name) # --> self.atom_frames

        idx_type = type(target_frame_idxs)
        if idx_type == str: 
            assert target_frame_idxs == ':'
            target_frame_idxs = np.arange(len(self.atom_frames)).tolist()
        elif idx_type in [list, tuple, np.ndarray]: 
            target_frame_idxs = np.array(target_frame_idxs, dtype=int).tolist()
        else:
            assert idx_type == int
            target_frame_idxs = [target_frame_idxs]

        self._target_frame_idxs = target_frame_idxs
        tmp_time = time.time()
        sys.stdout = old_stdout
        if self.print_time_cost: print('Read file cost: %.4fs' %(tmp_time-start_time))
        if len(target_frame_idxs) >= 3*n_jobs: verbose=5
        #  prefer="threads",
        result_lib = Parallel(n_jobs=n_jobs, prefer="threads", require='sharedmem', 
                        verbose=verbose)(delayed(self._process)(self.atom_frames, idx) 
                                         for idx in target_frame_idxs)
        sys.stdout.flush()
        for i, idx in enumerate(target_frame_idxs):
            self.RXN_History[idx] = result_lib[i]
        if self.save_rxn_result:
            self.write_rxn_analysis_result()
        return self.RXN_History


class RXNCollector_Factory(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.RXN_Collectors = {}
        self.RXNs = {}
        self.logs = []

    def write_rxn_analysis_result(self, RXNs=None, 
                                  save_fn='rxn_and_nx_graph_data.json'):
        if isinstance(RXNs, type(None)): RXNs = self.RXNs
        RXNs = copy.deepcopy(RXNs)
        for sd, history in RXNs.items():
            for _i, _rxn in history.items():
                RXNs[sd][_i]['RXN_CNT'] = NoIndent(history[_i]['RXN_CNT'])
                for _j, rxn in enumerate(_rxn['RXNs']):
                    RXNs[sd][_i]['RXNs'][_j] = NoIndent(rxn)
                for _j, mol_idx in _rxn['RXN_MOL_IDXs'].items():
                    RXNs[sd][_i]['RXN_MOL_IDXs'][_j] = NoIndent(mol_idx)
                for k, v in _rxn['RXN_Graph'].items():
                    if type(v) in [dict, tuple]:
                        RXNs[sd][_i]['RXN_Graph'][k] = NoIndent(v)
                    elif type(v) == list:
                        for _k, _dict in enumerate(v):
                            RXNs[sd][_i]['RXN_Graph'][k][_k] = NoIndent(_dict)
                    else:
                        RXNs[sd][_i]['RXN_Graph'][k] = v
        json_str = json.dumps(RXNs, indent=2, cls=MyEncoder)
        with open(self.dst_dir + '/%s'%save_fn, 'w') as f:
            f.write(json_str)

    def _process(self, sub_dir, xyz_name='all_step.xyz', 
                 target_frame_idxs=-1, n_jobs=1, verbose=0):
        start_time = time.time()
        rxn_collector = RXNCollector(sub_dir)
        rxn_collector.process(xyz_name, target_frame_idxs, 
                              n_jobs=n_jobs,
                              verbose=verbose,
                              print_changed_subgraph=0, 
                              save_rxn_result=True, 
                              print_time_cost=False)
        RXNs = {sub_dir.split('/')[-1]: rxn_collector.RXN_History}
        tmp_time = time.time()
        _rxn = NoIndent(rxn_collector.RXN_result['RXN_CNT'])
        json_str = json.dumps(_rxn, indent=2, cls=MyEncoder)
        log_data = '%s done! cost: %.4fs \n     =>  %s \n' %(
                sub_dir, (tmp_time-start_time), json_str)
        self.logs.append(log_data)
        sys.stdout = old_stdout
        print(log_data)
        return rxn_collector

    def process(self, sub_dir_list=None, xyz_name='all_step.xyz', frame_idxs=[-1], 
                n_jobs=1, save_rxn_result_fn='rxn_and_nx_graph_data.json', 
                save_rxn_statistics_fn='rxn_statistics_all.csv'):
        self.dst_dir = self.root_path + '/total_rxn_analysis_result'
        if not os.path.isdir(self.dst_dir):
            os.makedirs(self.dst_dir)

        if isinstance(sub_dir_list, type(None)):
            sub_dir_list = sorted(glob.glob(self.root_path+'/0x*'))
            sub_dir_list = [x for x in sub_dir_list if '.tar.gz' not in x and 
                        os.path.isdir(x)]
        idx_type = type(frame_idxs)
        parallel_sd = False
        if idx_type == str: 
            assert frame_idxs == ':'
            parallel_sd = False
        elif idx_type in [list, tuple, np.ndarray]: 
            frame_idxs = np.array(frame_idxs, dtype=int).tolist()
            parallel_sd = True if len(frame_idxs) <= 10 else False
        else:
            print(idx_type)
            assert idx_type == int
            frame_idxs = [frame_idxs]
            parallel_sd = True
        if parallel_sd: #, prefer='processes'
            res_lib = Parallel(n_jobs=n_jobs, prefer='threads', verbose=5)(
                        delayed(self._process)(sub_dir, xyz_name, frame_idxs, 
                            n_jobs, verbose=0) for sub_dir in sub_dir_list)
            sys.stdout.flush()
            for i, sub_dir in enumerate(sub_dir_list):
                self.RXN_Collectors[sub_dir.split('/')[-1]] = res_lib[i]
                RXNs = {sub_dir.split('/')[-1]: res_lib[i].RXN_History}
                dict_update(self.RXNs, RXNs)
        else:
            print('Prefer to run in a folder first!')
            for sub_dir in sub_dir_list:
                sys.stdout = old_stdout
                print(sub_dir, 'processing ...')
                rxn_collector = self._process(sub_dir, xyz_name, 
                                     frame_idxs, n_jobs=n_jobs)
                self.RXN_Collectors[sub_dir.split('/')[-1]] = rxn_collector
                RXNs = {sub_dir.split('/')[-1]: rxn_collector.RXN_History}
                dict_update(self.RXNs, RXNs)
        _save_fn = None
        if save_rxn_statistics_fn: 
            _save_fn = self.dst_dir+'/'+save_rxn_statistics_fn
        self.df = combine_rxn_statistics_df(self.RXNs, save_fn=_save_fn)
        if save_rxn_result_fn:
            self.write_rxn_analysis_result(self.RXNs, save_rxn_result_fn)
