import os
import importlib
import numpy as np
import pandas as pd
import utils.please
from utils.config import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.figure_style import my_paper
from statsmodels.stats.anova import AnovaRM
from scipy import stats
import random

plt.rcParams['font.family'] = my_paper['font_family']
# plt.rcParams['pdf.use14corefonts'] = True
# plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering if used

def learning_figure(measure='MD', fig_size=[8.2, 6], show_plot=False):
    '''
    Performance improvement over days with BNs separate.
    '''

    df = pd.read_csv(os.path.join(ANALYSIS_PATH, 'efc2_all.csv'))

    # make summary dataframe:
    df.replace(-1, np.nan, inplace=True)
    ANA = pd.DataFrame()
    ANA = df.groupby(['day','sn','trained','BN'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    num_blocks = ANA['BN'].unique().shape[0]

    # average across subjects:
    grouped = ANA.groupby(['day', 'trained', 'BN']).agg(
        MD_mean=('MD', 'mean'),
        MD_sem=('MD', 'sem'),
        RT_mean=('RT', 'mean'),
        RT_sem=('RT', 'sem'),
        ET_mean=('ET', 'mean'),
        ET_sem=('ET', 'sem'),
        len_avg=('MD', 'count')
    ).reset_index()
    grouped['adjusted_BN'] = (grouped['day']-1)*num_blocks + grouped['BN']

    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(fig_size[0]*cm, fig_size[1]*cm))
    
    # ========= Untrained =========
    subset = grouped[(grouped['trained']==0)]
    sns.lineplot(data=subset, x='adjusted_BN', y=measure+'_mean', hue='day',
                        marker='none', markersize=my_paper['marker_size'], linewidth=my_paper['line_width'], 
                        palette=[my_paper['color_untrained']], markeredgecolor=my_paper['color_untrained'], ax=ax)
    # plt.errorbar(subset['adjusted_BN'], subset[measure+'_mean'], yerr=subset[measure+'_sem'],
    #             fmt='none', ecolor=my_paper['color_untrained'], capsize=0, elinewidth=my_paper['error_width'])
    for day in subset['day'].unique():
        ax.fill_between(subset[subset['day']==day]['adjusted_BN'], 
                subset[subset['day']==day][measure+'_mean'] - subset[subset['day']==day][measure+'_sem'], 
                subset[subset['day']==day][measure+'_mean'] + subset[subset['day']==day][measure+'_sem'], 
                color=my_paper['color_untrained'], alpha=0.3, edgecolor='none')

    # ========= Trained =========
    subset = grouped[(grouped['trained']==1)]
    sns.lineplot(data=subset, x='adjusted_BN', y=measure+'_mean', hue='day',
                        marker='none', markersize=my_paper['marker_size'], linewidth=my_paper['line_width'], 
                        palette=[my_paper['colors_blue'][3]], markeredgecolor=my_paper['colors_blue'][3], ax=ax)
    # plt.errorbar(subset['adjusted_BN'], subset[measure+'_mean'], yerr=subset[measure+'_sem'],
    #             fmt='none', ecolor=my_paper['colors_blue'][3], capsize=0, elinewidth=my_paper['error_width'])
    for day in subset['day'].unique():
        ax.fill_between(subset[subset['day']==day]['adjusted_BN'], 
                subset[subset['day']==day][measure+'_mean'] - subset[subset['day']==day][measure+'_sem'], 
                subset[subset['day']==day][measure+'_mean'] + subset[subset['day']==day][measure+'_sem'], 
                color=my_paper['colors_blue'][3], alpha=0.3, edgecolor='none')

    if measure=='MD':
        ax.set_ylim([0, 2])
        ax.set_yticks(ticks=[0, 1, 2])
        ax.set_ylabel('mean deviation [N]')
    elif measure=='RT':
        ax.set_ylim([300, 700])
        ax.set_yticks(ticks=[300, 500, 700])
        ax.set_ylabel('reaction time [ms]')
    elif measure=='ET':
        ax.set_ylim([0, 4000])
        ax.set_yticks(ticks=[0, 1000, 2000, 3000, 4000], labels=['0','1','2','3','4'])
        ax.set_ylabel('execution time [s]')

    ax.set_xticks(ticks=[4.5, 12.5, 20.5, 28.5, 36.5], labels=['pre-test', 'd2', 'd3', 'd4', 'post-test'])
    ax.set_xlabel('day')
    ax.legend().set_visible(False)

    # Make it pretty:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    ax.spines["left"].set_bounds(ax.get_ylim()[0], ax.get_ylim()[-1])
    ax.spines["bottom"].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])

    ax.tick_params(axis='x', direction='in', length=2, width=my_paper['axis_width'])
    ax.tick_params(axis='y', direction='in', length=2, width=my_paper['axis_width'])

    fig.savefig(os.path.join(FIGURE_PATH,'efc2_learning_'+measure+'.pdf'), format='pdf', bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # STATS:
    print(f'========= {measure} STATS =========')
    ANA = df.groupby(['day','sn','trained'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    # t-test untrained vs. trained:
    print('untrained vs. trained:')
    # day 1:
    trained_day1 = ANA[(ANA['day']==1) & (ANA['trained']==1)][measure]
    untrained_day1 = ANA[(ANA['day']==1) & (ANA['trained']==0)][measure]
    res = stats.ttest_rel(trained_day1, untrained_day1)
    print(f'day1: t_{len(trained_day1)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')
    # day 5:
    trained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==1)][measure]
    untrained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==0)][measure]
    res = stats.ttest_rel(trained_day5, untrained_day5)
    print(f'day1: t_{len(trained_day5)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')

    # t-tets day 5, last block:
    ANA = df.groupby(['day','sn','trained','BN'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    trained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==1) & (ANA['BN']==8)][measure]
    untrained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==0) & (ANA['BN']==8)][measure]
    res = stats.ttest_rel(trained_day5, untrained_day5)
    print(f'day5, last block: t_{len(trained_day5)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')
    
    # ANOVA trained across days:
    print()
    print(f'rm-ANOVA trained:')
    ANA = df.groupby(['day','sn','trained','BN'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    subset = ANA[(ANA['trained']==1)]
    anova = AnovaRM(data=subset, depvar=measure, subject='sn', within=['day', 'BN']).fit()
    table = anova.anova_table
    table['Pr > F'] = table['Pr > F'].apply(lambda p: f'{p:.6f}')
    table['F Value'] = table['F Value'].apply(lambda f: f'{f:.6f}')
    print(table)

    # t-test untrained day1 vs. day5:
    print()
    print('untrained day1 vs. day5:')
    ANA = df.groupby(['day','sn','trained'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    day1 = ANA[(ANA['day']==1) & (ANA['trained']==0)][measure]
    day5 = ANA[(ANA['day']==5) & (ANA['trained']==0)][measure]
    res = stats.ttest_rel(day1, day5)
    print(f't_{len(day1)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')

def plot_trial_example(sn=None, chord=None, trial=None, fs=500, t_minus=None, t_max=None):
    df = pd.read_csv(os.path.join(ANALYSIS_PATH, f'efc2_all.csv'))
    chords = df['chordID'].unique()

    if sn is None:
        sn = random.choice([i for i in range(100,114) if i!=105])
    if chord is None:
        chord = random.choice(chords)

    # load the subject data:
    df = df[df['sn']==sn].reset_index(drop=True)
    idx = df[(df['chordID']==chord) & (df['trial_correct']==1)].index
    df = df.iloc[idx]
    mov = pd.read_pickle(os.path.join(ANALYSIS_PATH, f'efc2_{sn}_mov.pkl'))
    mov = mov.iloc[idx]

    if trial == 'average': # get the average:
        pass
        # day 1:
        # tmp_mov = mov[(mov['day']==1)]
        # for i in range(tmp_mov.shape[0]):

    elif trial is None: # choose random trials from each day:
        day1 = df[(df['day']==1)].index
        print(day1)
        day5 = df[(df['day']==5)].index
        # choose 5 random trials:
        trials_day1 = random.sample(day1, k=5)
        trials_day5 = random.sample(day5, k=5)

        # figure setup:
        cm = 1/2.54
        subplot_width = 3*cm  # cm
        subplot_height = 2*cm  # cm
        n_rows = 5
        n_cols = 5
        total_fig_width = n_cols * subplot_width
        total_fig_height = n_rows * subplot_height

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_fig_width, total_fig_height))

        for i in trials_day1+trials_day5:
            fGain = [df['fGain1'].iloc[i], df['fGain2'].iloc[i], df['fGain3'].iloc[i], df['fGain4'].iloc[i], df['fGain5'].iloc[i]]
            global_gain = df['forceGain'].iloc[i]
            fs = 500
            baseline_threshold = df['baselineTopThresh'].iloc[i]
            t, force = utils.please.get_trial_force(mov, fGain, global_gain, baseline_threshold, fs, t_minus, t_max)

            # plot:
            ax = axes.flat[i]  # Choose the correct axis
            ax.plot(t, force)  # Example plot
            # ax.legend()
            ax.set_title(f"Trial {i+1}")
        
        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()


    return df, mov
    