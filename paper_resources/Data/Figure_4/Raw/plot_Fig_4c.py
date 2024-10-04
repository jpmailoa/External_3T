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

def plot_rxn_cleaned_timestep_hist_by_mols(df, bins=25, show_transient_rxn=False, 
                                   xrange=(0, 250), charge_dict={}):
    a = 0.7
    base_dict = {'Li': 1, 'PF6': -1}
    assert isinstance(charge_dict, dict)
    charge_dict.update(base_dict)

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 4))
    mol_name = [x[7:] for x in df.columns if ('origin_' in x) and (x[7:]!='Li')]
    def rxn_mol_cnt(x,  name):
        src = [y.rsplit('_', 1)[0] for y in x.iloc[0]]
        res = {'%s_%s_cnt'%(name,y): src.count(y) for y in mol_name}
        return res

    # Manually control color to comply with Nat.Comm. editorial request
    colors = ['tab:blue','tab:orange','tab:green','tab:grey','tab:purple','tab:brown','tab:pink']
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

        # Dump source data  for Nat. Comm. editorial requirement
        if len(mn_idx) != 0:
            out_rxn_steps = np.array(mn_idx).reshape([-1,1])
            rxn_mol_name = mn.split('_')[0]
            np.savetxt("Fig_4c_stepNumber_"+rxn_mol_name+".csv", out_rxn_steps, delimiter=",", fmt="%i")

    ax1.hist(X, bins=bins, range=xrange, label=L, alpha=a, stacked=True, color=colors)
    ax1.legend(fontsize=16)
    ax1.set_xlabel('DFT call', fontsize=18)
    xticks = [0,50,100,150,200,250]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=16)
    ax1.set_ylabel('Frequency', fontsize=18)
    yticks = [0,10,20,30,40]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks, fontsize=16)

    plt.tight_layout()
    plt.savefig('Fig_4c.tiff', dpi=600)
    plt.savefig('Fig_4c_vFinal.pdf', dpi=600)

    # Create EMF vector graphic for Nat. Comm. editorial requirement
    save_fig('Fig_4c_vFinal.svg', dpi=600, conv_svg_to_emf=True, verbose=True)

    plt.show()
    return df

df_new = pd.read_csv('full_rxn_statistics_all_oxidation.csv', index_col=0)
for col in df_new.columns:
    if isinstance(df_new[col][0], str):
        df_new[col] = df_new[col].apply(eval)

charge_dict = {'PF6': -1, 'CO3_minus2': -2, 'oEC_radical_minus1_minus2': -1, 'Li': 1}
df_new_2 = plot_rxn_cleaned_timestep_hist_by_mols(df_new, show_transient_rxn=True, charge_dict=charge_dict)
