import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from pyhelpers.store import save_fig

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


def plot_rxn_new_timestep_hist_by_mols2(df, bins=25, show_fake_rxn=False, 
                xrange=(0, 250), charge_dict={}, highlight_rxn=[], remove_rxn=[]):
    a = 0.7
    base_dict = {'Li': 1, 'PF6': -1, 'oEC_radical_minus1_minus2': -1}
    assert isinstance(charge_dict, dict)
    charge_dict.update(base_dict)

    fig, ax1= plt.subplots(1, 1, figsize=(14, 5))
    mol_name = [x[7:] for x in df.columns if ('origin_' in x) and (x[7:]!='Li')]
    def rxn_mol_cnt(x,  name):
        src = [y.rsplit('_', 1)[0] for y in x.iloc[0]]
        res = {'%s_%s_cnt'%(name,y): src.count(y) for y in mol_name}
        return res

    name = 'rxn_new_src'
    applied_df = df[['rxn_new_src']].apply(rxn_mol_cnt, args=(name,), 
                        axis='columns', result_type='expand')
    df = pd.concat([df, applied_df], axis='columns')
    X, L = [], []
    for mn in mol_name:
        #remove data from last frame
        mn_idx = [i for i, x in zip(df.idx, df['%s_%s_cnt'%(name, mn)]) \
                  if (x!=0 and i not in remove_rxn) for y in range(x)]
        nc = charge_dict[mn] if mn in charge_dict else 0
        pom, t, nl = parse_plt_label(mn, nc)
        X.append(mn_idx)
        if len(mn_idx)==0: nl = '_nolegend_'
        L.append(nl)

        # Dump source data  for Nat. Comm. editorial requirement
        if len(mn_idx) != 0:
            out_rxn_steps = np.array(mn_idx, dtype=int).reshape([-1,1])
            rxn_mol_name = str.split(mn, '_')[0]
            np.savetxt("AIMD_reaction_stepNumber_"+rxn_mol_name+".csv", out_rxn_steps, delimiter=",", fmt="%i")

    _, _, xx = ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True)
    colors = [x.patches[0].get_facecolor() for x in xx]

    X, L, C = [], [], []
    for ii, mn in enumerate(mol_name):
        #remove data from last frame
        mn_idx = [i for i, x in zip(df.idx, df['%s_%s_cnt'%(name, mn)]) \
                  if (x!=0 and i in highlight_rxn) for y in range(x)]
        nc = charge_dict[mn] if mn in charge_dict else 0
        pom, t, nl = parse_plt_label(mn, nc)
        X.append(mn_idx)
        nl = '*'+nl
        if len(mn_idx)==0: nl = '_nolegend_'
        L.append(nl)
        C.append(colors[ii])
    ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True, 
             color=C, edgecolor='black', hatch='//')

    ax1.legend(ncol=1, columnspacing=0.8, fontsize=16, loc='upper left')
    #ax1.set_title('New Reaction', fontsize=14)
    ax1.set_xlabel('DFT call', fontsize=18)
    xticks = [0,200,400,600,800,1000,1200]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=16)
    ax1.set_ylabel('Frequency', fontsize=18)
    yticks = [0,2,4,6,8]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks, fontsize=16)

    plt.axis([0,1200,0,8])
    plt.tight_layout()
    plt.plot([1167,1167],[0,8],'k--', linewidth=2)
    plt.savefig('Fig_Supp_SingleTraj_Breakdown_AIMD.tiff', dpi=600)

    # Create EMF vector graphic for Nat. Comm. editorial requirement
    save_fig('Fig_Supp_SingleTraj_Breakdown_AIMD_vFinal.svg', dpi=600, conv_svg_to_emf=True, verbose=True)

    plt.show()
    return df
    

df_new = pd.read_csv('merged_full_rxn_statistics_all_AIMD.csv', index_col=0)
for col in df_new.columns:
    if isinstance(df_new[col][0], str):
        df_new[col] = df_new[col].apply(eval)

charge_dict = {'PF6': -1, 'CO3_minus2': -2, 'oEC_radical_minus1_minus2': -1, 'Li': 1}
df_new_2 = plot_rxn_new_timestep_hist_by_mols2(df_new, show_fake_rxn=True, bins=48,
                                               charge_dict=charge_dict, xrange=(0, 1200), 
                                               #highlight_rxn=[91, 257, 504, 654, 863, 1137], 
                                               remove_rxn=[113, 174, 208, 232, 400, 624, 646])
