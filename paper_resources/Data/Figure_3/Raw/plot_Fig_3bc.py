import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from functools import partial
from itertools import cycle

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
    #df.drop(columns=['config_id', 'rxn', 'rdc'], inplace=True)
    df.drop(columns=['config_id', 'rxn', 'rdc', 'rxn_wt_pid', 'rdc_wt_pid', 'rxn_changed_bonds'], inplace=True)
    gb = df.groupby(['excess_Li']).agg(['mean', partial(np.std, ddof=0)])
    x = gb.index.values

    # Dump source data  for Nat. Comm. editorial requirement
    count_dict = {}
    for n in n_excess_Li:
        if n in count_dict: count_dict[n] += 1
        else: count_dict[n] = 1
    print('Sample count grouped by excess charge:', count_dict)
    for excess in range(len(count_dict)):
        select_df = df.loc[df['excess_Li']==excess]
        for col in df.columns:
            if 'rxn_' in col:
                l = col[4:].split('_')[0]
                out = np.array(select_df[col].values).reshape([-1,1])
                np.savetxt("Fig_3b_excess_"+str(excess)+"_"+l+".csv", out, delimiter=",", fmt="%i")
            if 'rdc_' in col:
                l = col[4:].split('_')[0]
                out = np.array(select_df[col].values).reshape([-1,1])
                np.savetxt("Fig_3c_excess_"+str(excess)+"_"+l+".csv", out, delimiter=",", fmt="%i")
    
    # Manually control color to comply with Nat.Comm. editorial request
    colors = ['tab:blue','tab:orange','tab:green','tab:grey']
    colors.reverse()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    marker = cycle(_marker) 
    for col in df.columns:
        if 'rxn_' in col:
            l = col[4:]
            nc = charge_dict[l] if l in charge_dict else 0
            pom, t, nl = parse_plt_label(l, nc)
            # Special color change for PF6 to comply with editorial request
            ax1.errorbar(x, gb[col]['mean'], gb[col]['std'], ls='--', 
                         marker=next(marker), alpha=a, label=nl, capsize=3, color=colors.pop())
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=16, loc='upper left')
    handles = [h[0] for h in handles]
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=16)
    ax1.set_ylabel('Avg. mol. count', fontsize=18)
    yticks = [0,1,2,3]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks, fontsize=16)
    
    # Manually control color to comply with Nat.Comm. editorial request
    colors = ['tab:blue','tab:orange','tab:green','tab:grey']
    colors.reverse()
    marker = cycle(_marker) 
    for col in df.columns:
        if 'rdc_' in col:
            l = col[4:]
            nc = charge_dict[l] if l in charge_dict else 0
            pom, t, nl = parse_plt_label(l, nc)
            # Special color change for PF6 to comply with editorial request
            ax2.errorbar(x, gb[col]['mean'], gb[col]['std'], ls='--', 
                         marker=next(marker), alpha=a, label=nl, capsize=3, color=colors.pop())
    handles, labels = ax2.get_legend_handles_labels()
    # remove the errorbars
    handles = [h[0] for h in handles]
    # ax2.legend(handles, labels, fontsize=16, loc='upper center')
    title = 'Reduction' if not reverse else 'Oxidation'
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig('Fig_3bc.tiff', dpi=600)
    plt.savefig('Fig_3bc_vFinal.pdf', dpi=600)
    
    # Create EMF vector graphic for Nat. Comm. editorial requirement
    save_fig('Fig_3bc_vFinal.svg', dpi=600, conv_svg_to_emf=True, verbose=True)

    plt.show()

df = pd.read_csv('df_merge.csv', index_col=0)
plot_rxn_changed_mols(df)
