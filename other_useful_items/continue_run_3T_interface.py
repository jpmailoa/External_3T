import sys
sys.path.append('./utils/')
sys.path.append('./utils/Convert_Gromacs_LAMMPS/')
from run_utils import parse_config
from run_utils import create_model
from run_utils import create_optimizers
from run_utils import run_model
import torch
import os, math
import ase
import ase.io as sio

def main(config_json):
    blocks = parse_config(config_json)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = None
    line_count = 0
    for idx, block in enumerate(blocks):
        n_epoch = block['n_epoch']
        out_tag = block['out_tag']
        model = create_model(block, base_model=model).to(device)
        # If there is no file, it means this block failed at epoch 0
        if not os.path.isfile(out_tag+'_outE.txt'):
            continue_idx = idx
            break
        out_len = len(open(out_tag+'_outE.txt','r').readlines())
        # If this block succeeded before
        if out_len == n_epoch:
            continue_idx = idx + 1
            old_atom_pos = torch.Tensor(sio.read(out_tag+'.xyz',format='xyz').positions).to(device)
            model.reset_positions(old_atom_pos)
            line_count += math.ceil(out_len/block['print_freq'])
        # If we find the failed block
        else:
            continue_idx = idx
            break
    if continue_idx >= len(blocks):
        return

    # We will continue from block continue_idx
    # Prepare
    print('Continuing 3T from block',continue_idx,':',blocks[continue_idx]['out_tag'])
    os.system('rm workspace/*')
    if os.path.isfile('default.log'):
        logs = open('default.log','r').readlines()[0:line_count]
        with open('default.log','w') as f:
            for log in logs:
                f.write(log)
    # Continue
    for block in blocks[continue_idx:]:
        model = create_model(block, base_model=model).to(device)
        optimizers = create_optimizers(model, block)
        run_model(model, optimizers, block)

main('configs/LG100_EC_DMC_VC_v1.json')
