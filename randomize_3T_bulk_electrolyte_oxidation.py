import random
import os, sys
from main_run_utils import multiple_runs

def config_modify_func(template):
    # Prepare random config file
    n_Li = str(random.randrange(1,5))
    n_VC = str(random.randrange(1,4))
    n_PF6 = str(1)
    n_CO3 = str(random.randrange(1,3))
    n_Ethylene = n_CO3
    n_oEC = str(random.randrange(1,3))
    n_EC = str( 10 )
    config_str = template.replace('$$Li$$', n_Li)
    config_str = config_str.replace('$$VC$$', n_VC)
    config_str = config_str.replace('$$PF6$$', n_PF6)
    config_str = config_str.replace('$$CO3$$', n_CO3)
    config_str = config_str.replace('$$Ethylene$$', n_Ethylene)
    config_str = config_str.replace('$$oEC$$', n_oEC)
    config_str = config_str.replace('$$EC$$', n_EC)
    config_dir = 'configs/'+tag
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)
    duplicate = True
    while duplicate:
        config_json = config_dir + '/' + tag + '_' + hex(random.randrange(0, sys.maxsize)) + '.json'
        if not os.path.isfile(config_json):
            duplicate = False
    with open(config_json,'w') as f:
        f.write(config_str)
    return config_json, config_str


tag = 'Electrolyte_Oxidation'
multiple_runs(10, tag, config_modify_func)