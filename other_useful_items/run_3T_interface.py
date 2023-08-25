import sys
sys.path.append('./utils/')
sys.path.append('./utils/Convert_Gromacs_LAMMPS/')
from run_utils import parse_config
from run_utils import create_model
from run_utils import create_optimizers
from run_utils import run_model
import torch

def main(config_json):
    blocks = parse_config(config_json)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = None
    for block in blocks:
        model = create_model(block, base_model=model).to(device)
        optimizers = create_optimizers(model, block)
        run_model(model, optimizers, block)

main('configs/example_Li100_EC_DMC_VC_v8.json')
