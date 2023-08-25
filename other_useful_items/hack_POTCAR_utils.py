import os

def generate_POTCAR(elems):
    convert_dict = {'Li':'Li_sv'}
    POTCAR_dir = 'templates/VASP/POTCAR/'
    with open('POTCAR','w') as f:
        for elem in elems:
            if elem in convert_dict: elem = convert_dict[elem]
            POTCAR_file = POTCAR_dir + elem + '/POTCAR'
            block = open(POTCAR_file,'r').read()
            f.write(block)
    return

generate_POTCAR(['H','Li','C','O'])
