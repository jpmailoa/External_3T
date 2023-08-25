import torch

def calc_E_F_forcefield(model):
    # In this version, we want to decouple tensor transformation model from force calculation.
    # Cut off autograd to ensure no undesired gradients are accumulated in the tensor transformation model.
    atom_pos = torch.Tensor( model.atom_pos.detach().cpu().numpy() ).to(model.device)
    atom_pos.requires_grad = True
    na = atom_pos.shape[0]
    all_dist = torch.linalg.norm(atom_pos.unsqueeze(0).expand(na,na,3) - \
                                 atom_pos.unsqueeze(1).expand(na,na,3),
                                 dim=2)
    E_bond = calculate_E_bond(model, all_dist)
    E_angle = calculate_E_angle(model, atom_pos, all_dist)
    E_dihedral = calculate_E_dihedral(model, atom_pos)
    E_improper = calculate_E_improper(model, atom_pos)

    # For LJ and coulomb, we will want to include impact of PBC.
    # This ensures that the molecule does not mistakenly sink into the lattice's sides.
    mirror_pos = torch.zeros(3,3,3,na,3).to(model.device)
    cell = model.cell
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                mirror_pos[i+1,j+1,k+1] = atom_pos + i*cell[0] + j*cell[1] + k*cell[2]
    mirror_pos = mirror_pos.view(27*na,3)
    all_dist = torch.linalg.norm(mirror_pos.unsqueeze(0).expand(27*na,27*na,3) - \
                                 mirror_pos.unsqueeze(1).expand(27*na,27*na,3),
                                 dim=2)
    E_LJ = calculate_E_LJ(model, all_dist)
    E_coulomb = calculate_E_coulomb(model, all_dist)
            
    # right now this energy is kcal/mol
    E_total = E_bond + E_angle + E_dihedral + E_improper + E_LJ + E_coulomb

##    print('Energy bond\t: ',E_bond)
##    print('Energy angle\t: ',E_angle)
##    print('Energy dihedral\t: ',E_dihedral)
##    print('Energy improper\t: ',E_improper)
##    print('Energy LJ\t: ',E_LJ)
##    print('Energy Coulomb\t: ',E_coulomb)
##    print('Energy total\t: ',E_total)

    F_atoms = - torch.autograd.grad(E_total, atom_pos)[0]
    F_atoms = F_atoms.reshape(-1,3)

    del atom_pos, mirror_pos, all_dist
    del E_bond, E_angle, E_dihedral, E_improper, E_LJ, E_coulomb
    torch.cuda.empty_cache()

    E_total = E_total.detach().cpu().numpy()
    F_atoms = F_atoms.detach().cpu().numpy()
    return E_total, F_atoms

def calculate_E_bond(model, all_dist):
    d_bond = all_dist[ model.bond_idx[:,0],model.bond_idx[:,1] ]

    # Define output
    E_bond = 0
    
    # Harmonic
    idx = model.bond_harmonic_idx
    coeffs = model.bond_harmonic_coeffs
    if not(coeffs.shape[0] == 0):
        E_bond_harmonic = coeffs[:,0] * ((d_bond[idx] - coeffs[:,1])**2)
        E_bond += E_bond_harmonic.sum()
    
    return E_bond

def _d2r(angle_deg):
    torch_pi = torch.acos(torch.zeros(1))*2
    angle_rad = angle_deg / 180.0 * torch_pi.to(angle_deg.device)
    return angle_rad

def calculate_E_angle(model, atom_pos, all_dist):
    v1 = atom_pos[ model.angle_idx[:,0] ] - atom_pos[ model.angle_idx[:,1] ]
    v2 = atom_pos[ model.angle_idx[:,2] ] - atom_pos[ model.angle_idx[:,1] ]
    temp1 = torch.sum(v1*v2, dim=1)
    temp2 = torch.linalg.norm(v1, dim=1) * torch.linalg.norm(v2, dim=1)
    d_cos = temp1 / temp2
    d_cos = torch.clamp(d_cos, min=-0.999999, max=0.999999)
    angle = torch.acos( d_cos )
    dist = all_dist[ model.angle_idx[:,0], model.angle_idx[:,2] ]

    # Define output
    E_angle = 0
    
    # Harmonic
    idx = model.angle_harmonic_idx
    coeffs = model.angle_harmonic_coeffs
    if not (coeffs.shape[0] == 0):
        ref_angle = _d2r(coeffs[:,1])
        E_angle_harmonic = coeffs[:,0] * ((angle[idx] - ref_angle)**2)
        E_angle += E_angle_harmonic.sum()

    # Charmm
    idx = model.angle_charmm_idx
    coeffs = model.angle_charmm_coeffs
    if not (coeffs.shape[0] == 0):
        ref_angle = _d2r(coeffs[:,1])
        E_angle_charmm = coeffs[:,0] * ((angle[idx] - ref_angle)**2) +\
                         coeffs[:,2] * ((dist[idx] - coeffs[:,3])**2)
        E_angle += E_angle_charmm.sum()        

    return E_angle

def calculate_E_dihedral(model, atom_pos):
    v12 = atom_pos[ model.dihedral_idx[:,0] ] - atom_pos[ model.dihedral_idx[:,1] ]
    v32 = atom_pos[ model.dihedral_idx[:,2] ] - atom_pos[ model.dihedral_idx[:,1] ]
    v43 = atom_pos[ model.dihedral_idx[:,3] ] - atom_pos[ model.dihedral_idx[:,2] ]
    v123 = torch.cross(v12,v32, dim=1)
    v234 = torch.cross(-v32,v43, dim=1)
    temp1 = torch.sum(v123*v234, dim=1)
    temp2 = torch.linalg.norm(v123, dim=1) * torch.linalg.norm(v234, dim=1)
    d_cos = temp1 / temp2
    d_cos = torch.clamp(d_cos, min=-0.999999, max=0.999999)

    # Define output
    E_dihedral = 0
    
    # Multi/harmonic
    idx = model.dihedral_multiharm_idx
    coeffs = model.dihedral_multiharm_coeffs
    if not(coeffs.shape[0] == 0):
        temp_d_cos = d_cos[idx]
        E_dihedral_multiharm = coeffs[:,0] +\
                               coeffs[:,1] * temp_d_cos +\
                               coeffs[:,2] * torch.pow(temp_d_cos,2) +\
                               coeffs[:,3] * torch.pow(temp_d_cos,3) +\
                               coeffs[:,4] * torch.pow(temp_d_cos,4)
        E_dihedral += E_dihedral_multiharm.sum()
    
    # Charmm
    idx = model.dihedral_charmm_idx
    coeffs = model.dihedral_charmm_coeffs
    if not(coeffs.shape[0] == 0):
        d_acos = torch.acos(d_cos[idx])
        ref_angle = _d2r(coeffs[:,2])
        E_dihedral_charmm = coeffs[:,0] * (1 + torch.cos(coeffs[:,1]*d_acos - ref_angle)) * coeffs[:,3]

        # There is nan grad problem associated with weight of 0 in CHARMM force field coeffs[:,3]
        # Somehow Gromacs-LAMMPS conversion produces these 0-contribution force field.
        # Ignoring the contribution eliminates this problem
        if (coeffs[:,3]**2).sum() == 0:
            E_dihedral = E_dihedral
        else:
            E_dihedral += E_dihedral_charmm.sum()

    return E_dihedral

def calculate_E_improper(model, atom_pos):
    if (model.improper_idx.shape[0] == 0):
        return 0
    v12 = atom_pos[ model.improper_idx[:,0] ] - atom_pos[ model.improper_idx[:,1] ]
    v32 = atom_pos[ model.improper_idx[:,2] ] - atom_pos[ model.improper_idx[:,1] ]
    v43 = atom_pos[ model.improper_idx[:,3] ] - atom_pos[ model.improper_idx[:,2] ]
    v123 = torch.cross(v12,v32, dim=1)
    v234 = torch.cross(-v32,v43, dim=1)
    temp1 = torch.sum(v123*v234, dim=1)
    temp2 = torch.linalg.norm(v123, dim=1) * torch.linalg.norm(v234, dim=1)
    d_cos = temp1 / temp2
    d_cos = torch.clamp(d_cos, min=-0.999999, max=0.999999)
    
    # Define output
    E_improper = 0
    
    # Harmonic
    idx = model.improper_harmonic_idx
    coeffs = model.improper_harmonic_coeffs
    if not(coeffs.shape[0] == 0):
        d_acos = torch.acos(d_cos[idx])
        ref_angle = _d2r(coeffs[:,1])
        E_improper_harmonic = coeffs[:,0] * ((d_acos - ref_angle)**2)
        E_improper += E_improper_harmonic.sum()

    return E_improper

def calculate_E_LJ(model, all_dist):
    na = model.atom_pos.shape[0]
    # these indices get all pairs with 0 < rij < 9.0, but both (i,j) and (j,i) are included
    indices = torch.nonzero( (all_dist-4.5)**2 < 4.5**2, as_tuple=False).to(torch.device('cpu'))
    # with the following modification, only 1 of each reciprocal pairs in primary box are included
    indices = indices[torch.nonzero(indices[:,0]<indices[:,1], as_tuple=False)[:,0],:]
    box_i = indices[:,0].div(na, rounding_mode='floor').to(model.device)
    box_j = indices[:,1].div(na, rounding_mode='floor').to(model.device)
    idx_i = indices[:,0] % na
    idx_j = indices[:,1] % na
    type_i = model.atom_type[ idx_i ]
    type_j = model.atom_type[ idx_j ]
    eps = model.epsilon[ type_i, type_j ]
    sigma = model.sigma[ type_i, type_j ]
    r = all_dist[ indices[:,0], indices[:,1] ]
    frac = (sigma/r)**6
    mask = model.sb_mask[ idx_i, idx_j ]
    ones = torch.ones( mask.shape[0] ).to(model.device)
    mask = torch.where( box_i == box_j, mask, ones )
    nonzero = torch.where(mask!=0)[0]
    eps = eps[nonzero]
    frac = frac[nonzero]
    mask = mask[nonzero]
    E_LJ = 4*eps*( frac**2 - frac ) * mask
    return E_LJ.sum() / 27.0    # divide by 27 because we include all 27 mirror PBC boxes

def calculate_E_coulomb(model, all_dist):
    na = model.atom_pos.shape[0]
    indices = torch.triu_indices(27*na,27*na,1).transpose(0,1).to(model.device)
    box_i = indices[:,0].div(na, rounding_mode='floor').to(model.device)
    box_j = indices[:,1].div(na, rounding_mode='floor').to(model.device)
    idx_i = indices[:,0] % na
    idx_j = indices[:,1] % na
    r = all_dist[ indices[:,0], indices[:,1] ]
    charge_i = model.atom_charge[ idx_i ]
    charge_j = model.atom_charge[ idx_j ]
    mask = model.sb_mask[ idx_i, idx_j ]
    ones = torch.ones( mask.shape[0] ).to(model.device)
    mask = torch.where( box_i == box_j, mask, ones )
    nonzero = torch.where(mask!=0)[0]
    charge_i = charge_i[nonzero]
    charge_j = charge_j[nonzero]
    r = r[nonzero]
    mask = mask[nonzero]
    E_coulomb = model.coulomb_coeff * charge_i * charge_j * mask / r
    return E_coulomb.sum() / 27.0   # divide by 27 because we include all 27 mirror PBC boxes

def wrap_atoms(pos_raw, cell):
    cell_transpose = cell.transpose(0,1)
    cell_transpose_inv = cell_transpose.inverse()
    cell_dot = torch.zeros(3,3).to(pos_raw.device)
    for m in range(3):
        for n in range(3):
            cell_dot[m, n] = cell[m].dot(cell[n])
    cell_dot_inv = cell_dot.inverse()
    rel_pos = pos_raw.matmul(cell_transpose).matmul(cell_dot_inv)
    rel_wrap = rel_pos - rel_pos.floor()
    pos_wrap = rel_wrap.matmul(cell_dot).matmul(cell_transpose_inv)
    return pos_wrap
