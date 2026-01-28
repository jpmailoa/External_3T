import sys
sys.path.append('./utils/')
sys.path.append('./utils/Convert_Gromacs_LAMMPS/')
from run_utils import parse_config
from run_utils import create_model
from run_utils import create_optimizers
from run_utils import run_model
import torch
import random
import os
import json

def main(config_json):
    blocks = parse_config(config_json)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = None
    for block in blocks:
        model = create_model(block, base_model=model).to(device)
        optimizers = create_optimizers(model, block)
        run_model(model, optimizers, block)
    return

def multiple_runs(run_num, tag, config_modify_func):
    for repeat in range(run_num):
        # Read template
        template = open('configs/'+tag+'_template.json','r').read()

        # Prepare config file with desired number and type of input molecules
        config_json, config_str = config_modify_func(template)
        
        # Execute
        main(config_json)

        # Store results
        result_dir = 'results/' + tag + '/'
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_dir = result_dir + config_json[:-5].split('_')[-1] + '/'
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        os.system('mv FF_step* ' + result_dir)
        os.system('mv VASP_step* ' + result_dir)
        os.system('mv default.log ' + result_dir)
        os.system('mv VASP_files ' + result_dir)
        os.system('cp ' + config_json + ' ' + result_dir)
        json_str = json.loads(config_str)
        poscar_file = json_str[0]['lattice_poscar']['file']
        os.system('cp ' + poscar_file + ' ' + result_dir)

    return
