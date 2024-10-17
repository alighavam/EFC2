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
import matplotlib.patches as patches

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
        ax.set_ylabel('mean deviation [N]', fontsize=my_paper['label_fontsize'])
    elif measure=='RT':
        ax.set_ylim([300, 700])
        ax.set_yticks(ticks=[300, 500, 700])
        ax.set_ylabel('reaction time [ms]', fontsize=my_paper['label_fontsize'])
    elif measure=='ET':
        ax.set_ylim([0, 4000])
        ax.set_yticks(ticks=[0, 1000, 2000, 3000, 4000], labels=['0','1','2','3','4'])
        ax.set_ylabel('execution time [s]', fontsize=my_paper['label_fontsize'])

    ax.set_xticks(ticks=[4.5, 12.5, 20.5, 28.5, 36.5], labels=['pre-test', 'd2', 'd3', 'd4', 'post-test'])
    ax.set_xlabel('day', fontsize=my_paper['label_fontsize'])
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

    ax.tick_params(axis='both', labelsize=my_paper['tick_fontsize'])

    fig.savefig(os.path.join(FIGURE_PATH,'efc2_learning_'+measure+'.pdf'), format='pdf', bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # STATS:
    print(f'========= {measure} STATS =========')
    ANA = df.groupby(['day','sn','trained'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    # t-test untrained vs. trained:
    print('"untrained vs. trained:"')
    # day 1:
    trained_day1 = ANA[(ANA['day']==1) & (ANA['trained']==1)][measure]
    untrained_day1 = ANA[(ANA['day']==1) & (ANA['trained']==0)][measure]
    res = stats.ttest_rel(trained_day1, untrained_day1)
    print(f'    day1: t_{len(trained_day1)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')
    # day 5:
    trained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==1)][measure]
    untrained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==0)][measure]
    res = stats.ttest_rel(trained_day5, untrained_day5)
    print(f'    day5: t_{len(trained_day5)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')

    # t-tets day 5, last block:
    ANA = df.groupby(['day','sn','trained','BN'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    trained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==1) & (ANA['BN']==8)][measure]
    untrained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==0) & (ANA['BN']==8)][measure]
    res = stats.ttest_rel(trained_day5, untrained_day5)
    print(f'    day5, last block: t_{len(trained_day5)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')

    # day5 untrained vs day2 trained:
    ANA = df.groupby(['day','sn','trained'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    trained_day2 = ANA[(ANA['day']==2) & (ANA['trained']==1)][measure]
    untrained_day5 = ANA[(ANA['day']==5) & (ANA['trained']==0)][measure]
    res = stats.ttest_rel(trained_day2, untrained_day5, alternative='two-sided')
    print(f'    day 5 untrained vs. day 2 trained: t_{len(trained_day2)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')

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

def specific_vs_general(measure='MD', fig_size=[8.2, 6], show_plot=False, plot_type='box'):
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

    # percent improvements:
    ANA = df.groupby(['day','sn','trained','BN'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    # average day1, block 7 and 8:
    t1 = ANA[(ANA['day'] == 1) & (ANA['trained'] == 1) & (ANA['BN'].isin([7,8]))].groupby(['day', 'trained', 'sn']).mean()[measure].values
    t5 = ANA[(ANA['day'] == 5) & (ANA['trained'] == 1) & (ANA['BN'].isin([1,2]))].groupby(['day', 'trained', 'sn']).mean()[measure].values
    u1 = ANA[(ANA['day'] == 1) & (ANA['trained'] == 0) & (ANA['BN'].isin([7,8]))].groupby(['day', 'trained', 'sn']).mean()[measure].values
    u5 = ANA[(ANA['day'] == 5) & (ANA['trained'] == 0) & (ANA['BN'].isin([1,2]))].groupby(['day', 'trained', 'sn']).mean()[measure].values

    # ======================================== BOX PLOT ======================================== #
    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(fig_size[0]*cm, fig_size[1]*cm))

    if plot_type == 'box':
        # Add major gridlines in the y-axis
        ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.boxplot([t1,t5], positions=[0.8, 1.2], widths=0.2,
                boxprops=dict(color=my_paper['colors_blue'][3]),
                whiskerprops=dict(color=my_paper['colors_blue'][3]),
                medianprops=dict(color=my_paper['colors_blue'][3]))
        
        ax.boxplot([u1,u5], positions=[4.8, 5.2], widths=0.2, 
                boxprops=dict(color=my_paper['color_untrained']),
                whiskerprops=dict(color=my_paper['color_untrained']),
                medianprops=dict(color=my_paper['color_untrained']))

        x = np.repeat([4.8, 5.2],len(t1)) + np.random.normal(0, 0.05, 2*len(t1))
        y = np.concatenate([u1,u5])
        for i in range(len(t1)):
            ax.plot([x[i],x[i+len(t1)]], [y[i],y[i+len(t1)]], color='#FAC9B8', linewidth=0.5, zorder=1)
        x = np.repeat([0.8, 1.2],len(t1)) + np.random.normal(0, 0.05, 2*len(t1))
        y = np.concatenate([t1,t5])
        for i in range(len(t1)):
            ax.plot([x[i],x[i+len(t1)]], [y[i],y[i+len(t1)]], color='#8AC6D0', linewidth=0.5, zorder=1)

    if plot_type == 'scatter':
        # Add major gridlines in the y-axis
        ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
        x = np.repeat([0.8, 1.2],len(t1)) + np.random.normal(0, 0.05, 2*len(t1))
        y = np.concatenate([u1,u5])
        for i in range(len(t1)):
            ax.plot([x[i],x[i+len(t1)]], [y[i],y[i+len(t1)]], color='#FAC9B8', linewidth=0.5, zorder=1)
        ax.scatter(x, y, color=my_paper['color_untrained'], s=10, zorder=2)

        x = np.repeat([0.8, 4.8],len(t1)) + np.random.normal(0, 0.05, 2*len(t1))
        y = np.concatenate([t1,t5])
        for i in range(len(t1)):
            ax.plot([x[i],x[i+len(t1)]], [y[i],y[i+len(t1)]], color='#8AC6D0', linewidth=0.5, zorder=1)
        ax.scatter(x, y, color=my_paper['colors_blue'][3], s=10, zorder=2)
    
    if measure=='MD':
        ax.set_ylim([0, 3])
        ax.set_yticks(ticks=[0, 1, 2, 3])
        ax.set_ylabel('mean deviation [N]', fontsize=my_paper['label_fontsize'])
    elif measure=='RT':
        ax.set_ylim([300, 700])
        ax.set_yticks(ticks=[300, 500, 700])
        ax.set_ylabel('reaction time [ms]', fontsize=my_paper['label_fontsize'])
    elif measure=='ET':
        ax.set_ylim([0, 4000])
        ax.set_yticks(ticks=[0, 1000, 2000, 3000, 4000], labels=['0','1','2','3','4'])
        ax.set_ylabel('execution time [s]', fontsize=my_paper['label_fontsize'])

    ax.set_xticks(ticks=[1,5], labels=['pre-test', 'post-test'])
    ax.set_xlabel('day', fontsize=my_paper['label_fontsize'])
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

    ax.tick_params(axis='both', labelsize=my_paper['tick_fontsize'])

    # fig.savefig(os.path.join(FIGURE_PATH,'efc2_imprv_'+measure+'.pdf'), format='pdf', bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
def imprv(measure='MD', fig_size=[8.2, 6], show_plot=False):
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

    # percent improvements:
    ANA = df.groupby(['day','sn','trained','BN'])[['is_test','group','RT','ET','MD']].mean().reset_index()
    # average day1, block 7 and 8:
    t1 = ANA[(ANA['day'] == 1) & (ANA['trained'] == 1) & (ANA['BN'].isin([7,8]))].groupby(['day', 'trained', 'sn']).mean()[measure].values
    t5 = ANA[(ANA['day'] == 5) & (ANA['trained'] == 1) & (ANA['BN'].isin([1,2]))].groupby(['day', 'trained', 'sn']).mean()[measure].values
    u1 = ANA[(ANA['day'] == 1) & (ANA['trained'] == 0) & (ANA['BN'].isin([7,8]))].groupby(['day', 'trained', 'sn']).mean()[measure].values
    u5 = ANA[(ANA['day'] == 5) & (ANA['trained'] == 0) & (ANA['BN'].isin([1,2]))].groupby(['day', 'trained', 'sn']).mean()[measure].values

    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(fig_size[0]*cm, fig_size[1]*cm))
    trained_imprv = (t1-t5)
    untrained_imprv = (u1-u5)
    # Add major gridlines in the y-axis
    # ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.axhline(y=0, color=(0.8,0.8,0.8), linestyle='--', linewidth=0.5)
    ax.boxplot([trained_imprv], positions=[1], widths=0.15,
            boxprops=dict(color=my_paper['colors_blue'][3]),
            whiskerprops=dict(color=my_paper['colors_blue'][3]),
            medianprops=dict(color=my_paper['colors_blue'][3]))
    ax.boxplot([untrained_imprv], positions=[2], widths=0.15,
            boxprops=dict(color=my_paper['color_untrained']),
            whiskerprops=dict(color=my_paper['color_untrained']),
            medianprops=dict(color=my_paper['color_untrained']))
    jitter = 0.05  # Jitter to avoid overlap
    ax.scatter(np.ones_like(trained_imprv) * 1 + np.random.uniform(-jitter, jitter, size=trained_imprv.shape),
               trained_imprv, color=my_paper['colors_blue'][3], zorder=2, s=4, alpha=0.5)
    ax.scatter(np.ones_like(untrained_imprv) * 2 + np.random.uniform(-jitter, jitter, size=untrained_imprv.shape),
               untrained_imprv, color=my_paper['color_untrained'], zorder=2, s=4, alpha=0.5)
    
    if measure=='MD':
        ax.set_ylim([-1.2, 1.2])
        ax.set_yticks(ticks=[-1.2,0,1.2])
        ax.set_ylabel('mean deviation improvement [N]', fontsize=my_paper['label_fontsize'])
    elif measure=='RT':
        ax.set_ylim([-500, 500])
        ax.set_yticks(ticks=[-500, 0, 500])
        ax.set_ylabel('reaction time improvement [ms]', fontsize=my_paper['label_fontsize'])
    elif measure=='ET':
        ax.set_ylim([-3000, 3000])
        ax.set_yticks(ticks=[-3000, 0, 3000], labels=['-3','0','3'])
        ax.set_ylabel('execution time improvement [s]', fontsize=my_paper['label_fontsize'])

    ax.set_xlim([0.8, 2.2])
    ax.set_xticks(ticks=[1,2], labels=['trained', 'untrained'])
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
    ax.tick_params(axis='both', labelsize=my_paper['tick_fontsize'])

    fig.savefig(os.path.join(FIGURE_PATH,'efc2_imprv_'+measure+'.pdf'), format='pdf', bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # STATS:
    print(f'========= {measure} STATS =========')
    relative = untrained_imprv / trained_imprv
    print(f'relative improvement: {np.mean(relative)*100:.3f} +/- {np.std(relative)/np.sqrt(len(relative))*100:.3f}')

    # t-test improvement vs 0:
    print()
    print('trained improvement vs. 0:')
    res = stats.ttest_1samp(trained_imprv, 0)
    print(f'    trained: t_{len(trained_imprv)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')
    print('untrained improvement vs. 0:')
    res = stats.ttest_1samp(untrained_imprv, 0)
    print(f'   untrained: t_{len(untrained_imprv)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')
    
def plot_trial_example(sn=None, chord=None, trial=None, fs=500, t_minus=None, t_max=None, fig_size=[6, 4], days=[1,5], num_trials=3, export_fig=False, xlim=None):
    df = pd.read_csv(os.path.join(ANALYSIS_PATH, f'efc2_all.csv'))
    chords = df['chordID'].unique()

    if sn is None:
        sn = random.choice([i for i in range(100,114) if i!=105])
    if chord is None:
        chord = random.choice(chords)

    # load the subject data:
    df = df[df['sn']==sn].reset_index(drop=True)
    idx = df[(df['chordID']==chord) & (df['trial_correct']==1)].index
    df = df.iloc[idx].reset_index(drop=True)
    trained = df['trained'].iloc[0]
    mov = pd.read_pickle(os.path.join(ANALYSIS_PATH, f'efc2_{sn}_mov.pkl'))
    mov = mov.iloc[idx].reset_index(drop=True)

    print(f'subject {sn}, chord {chord}, trained {trained}\n')
    if trial == 'average': # get the average of trials:
        # loop on days:
        for i, day in enumerate(days):
            idx = df[(df['day']==day)].index.to_list()
            # idx = idx[:10]
            print(f'average of day {day}')
            ET = np.mean(df['ET'].iloc[idx])
            MD = np.mean(df['MD'].iloc[idx])
            print(f'mean ET: {ET}')
            print(f'mean MD: {MD}')

            # get the forces of the day and average them:
            forces = []
            max_length = 0
            for trial_idx in idx:
                fGain = [df['fGain1'].iloc[trial_idx], df['fGain2'].iloc[trial_idx], df['fGain3'].iloc[trial_idx], df['fGain4'].iloc[trial_idx], df['fGain5'].iloc[trial_idx]]
                global_gain = df['forceGain'].iloc[trial_idx]
                fs = 500
                baseline_threshold = df['baselineTopThresh'].iloc[trial_idx]
                t, force = utils.please.get_trial_force(mov['mov'].iloc[trial_idx], fGain, global_gain, baseline_threshold, fs, t_minus, t_max)
                force = utils.please.moving_average(force, 10)
                forces.append(force)
                if force.shape[0] > max_length:
                    max_length = force.shape[0]

            # Zero-pad the forces to the length of the longest trial
            padded_forces = []
            for force in forces:
                pad_width = max_length - force.shape[0]
                padded_force = np.pad(force, ((0, pad_width), (0, 0)), mode='edge')
                padded_forces.append(padded_force)
            
            # Compute the average force
            average_force = np.mean(padded_forces, axis=0)

            # Compute the SEM of the force
            sem_force = np.std(padded_forces, axis=0) / np.sqrt(len(padded_forces))

            # Plot the average force
            cm = 1/2.54
            fig, ax = plt.subplots(figsize=(fig_size[0]*cm, fig_size[1]*cm))
            ax.axhline(y=2, color='#84D788', linestyle='--', label='y2')  # Horizontal dashed line at y=0
            ax.axhline(y=-2, color='#84D788', linestyle='--', label='y-2')  # Horizontal dashed line at y=0
            # Create a patch (a rectangle in this case)
            rect = patches.Rectangle((-1, -1.2), 6, 2.4, linewidth=2, edgecolor='none', facecolor='#EBEBEB', label='zone')
            ax.add_patch(rect)
            
            # choose appropriate t limit for a nice looking figure:
            t = np.linspace(0, max_length / fs, max_length)
            t_lim = utils.please.find_closest_index(t, ET/1000)
            if xlim is not None:
                t_lim = t_lim = utils.please.find_closest_index(t, xlim[i])
            t = t[:t_lim]
            average_force = average_force[:t_lim, :]
            sem_force = sem_force[:t_lim, :]
            for finger in range(average_force.shape[1]):
                ax.plot(t, average_force[:, finger], label=f'f {finger+1}', color=my_paper['colors_colorblind'][finger], linewidth=0.7)
                ax.fill_between(t, average_force[:, finger] - sem_force[:, finger], average_force[:, finger] + sem_force[:, finger], color=my_paper['colors_colorblind'][finger], alpha=0.3)

            ax.set_ylim([-6, 6])
            ax.set_xlim([0, 5])
            ax.set_yticks(ticks=[-5,0,5])
            ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5])
            ax.set_ylabel('force [N]', fontsize=my_paper['label_fontsize'])
            ax.set_xlabel('time [s]', fontsize=my_paper['label_fontsize'])
            # trial day: 
            ax.set_title(f'day {day}', fontsize=6)

            # Make it pretty:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

            ax.spines["left"].set_bounds(ax.get_ylim()[0], ax.get_ylim()[-1])
            ax.spines["bottom"].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])

            ax.tick_params(axis='x', direction='in', length=2, width=my_paper['axis_width'])
            ax.tick_params(axis='y', direction='in', length=2, width=my_paper['axis_width'])
            ax.tick_params(axis='both', labelsize=my_paper['tick_fontsize'])

            # Show the plot
            plt.show()

            if export_fig:
                fig.savefig(os.path.join(FIGURE_PATH, f'efc2_example_avg_{sn}_{chord}_{day}.pdf'), format='pdf', bbox_inches='tight')

    elif trial is None: # choose random trials from each day:
        day1 = df[(df['day']==days[0])].index.to_list()
        day5 = df[(df['day']==days[-1])].index.to_list()
        # choose k random trials:
        trials_day1 = random.sample(day1, k=num_trials)
        trials_day5 = day5[-num_trials:]
        trained = df['trained'].iloc[0]
        print(f'subject {sn}, chord {chord}, trained {trained}\n')
        print(f'day{days[0]}: ', trials_day1, f' day{days[-1]}: ', trials_day5, '\n')
        ET = df['ET'].iloc[trials_day1+trials_day5]
        MD = df['MD'].iloc[trials_day1+trials_day5]
        print('ET:', ET.values)
        print('MD:', MD.values)

        # figure setup:
        cm = 1/2.54
        for i, trial_idx in enumerate(trials_day1+trials_day5):
            fGain = [df['fGain1'].iloc[trial_idx], df['fGain2'].iloc[trial_idx], df['fGain3'].iloc[trial_idx], df['fGain4'].iloc[trial_idx], df['fGain5'].iloc[trial_idx]]
            global_gain = df['forceGain'].iloc[trial_idx]
            fs = 500
            baseline_threshold = df['baselineTopThresh'].iloc[trial_idx]
            t, force = utils.please.get_trial_force(mov['mov'].iloc[trial_idx], fGain, global_gain, baseline_threshold, fs, t_minus, t_max)
            force = utils.please.moving_average(force, 10)

            # plot:
            fig, ax = plt.subplots(figsize=(fig_size[0]*cm, fig_size[1]*cm))
            ax.axhline(y=2, color='#84D788', linestyle='--', label='y2')  # Horizontal dashed line at y=0
            ax.axhline(y=-2, color='#84D788', linestyle='--', label='y-2')  # Horizontal dashed line at y=0

            for i in range(force.shape[1]):
                plt.plot(t, force[:, i], color=color_palette[i], label=f'f {i+1}')

            # Create a patch (a rectangle in this case)
            rect = patches.Rectangle((-1, -1.2), 6, 2.4, linewidth=2, edgecolor='none', facecolor='#F0F0F0', label='zone')
            ax.add_patch(rect)

            # ax.legend(['f1', 'f2', 'f3', 'f4', 'f5'], loc='upper right', fontsize=my_paper['leg_fontsize'])
            ax.set_ylim([-6, 6])
            ax.set_xlim([t[0], t[-1]])
            ax.set_yticks(ticks=[-5,0,5])
            ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5])
            ax.set_ylabel('force [N]', fontsize=my_paper['label_fontsize'])
            ax.set_xlabel('time [s]', fontsize=my_paper['label_fontsize'])
            ax.set_title(f'trial {trial_idx}', fontsize=6)

            # Make it pretty:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

            ax.spines["left"].set_bounds(ax.get_ylim()[0], ax.get_ylim()[-1])
            ax.spines["bottom"].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])

            ax.tick_params(axis='x', direction='in', length=2, width=my_paper['axis_width'])
            ax.tick_params(axis='y', direction='in', length=2, width=my_paper['axis_width'])
            ax.tick_params(axis='both', labelsize=my_paper['tick_fontsize'])

            # Show the plot
            plt.show()

    else: # if a list of trails is provided:
        trained = df['trained'].iloc[0]
        print(f'subject {sn}, chord {chord}, trained {trained}\n')
        print('trials:', trial, '\n')
        ET = df['ET'].iloc[trial]
        MD = df['MD'].iloc[trial]
        print('ET:', ET.values)
        print('MD:', MD.values)
        # tlim1 = 0
        # tlim2 = 5
        # if t_minus is not None:
        #     tlim1 = -t_minus/1000
        # if t_max is not None:
        #     tlim2 = t_max/1000

        # figure setup:
        cm = 1/2.54
        for i, trial_idx in enumerate(trial):
            fGain = [df['fGain1'].iloc[trial_idx], df['fGain2'].iloc[trial_idx], df['fGain3'].iloc[trial_idx], df['fGain4'].iloc[trial_idx], df['fGain5'].iloc[trial_idx]]
            global_gain = df['forceGain'].iloc[trial_idx]
            fs = 500
            baseline_threshold = df['baselineTopThresh'].iloc[trial_idx]
            t, force = utils.please.get_trial_force(mov['mov'].iloc[trial_idx], fGain, global_gain, baseline_threshold, fs, t_minus, t_max)
            force = utils.please.moving_average(force, 15)

            # plot:
            fig, ax = plt.subplots(figsize=(fig_size[0]*cm, fig_size[1]*cm))
            ax.axhline(y=2, color='#84D788', linestyle='--', label='y2')  # Horizontal dashed line at y=0
            ax.axhline(y=-2, color='#84D788', linestyle='--', label='y-2')  # Horizontal dashed line at y=0

            for finger in range(force.shape[1]):
                plt.plot(t, force[:,finger], label=f'f {i+1}', color=my_paper['colors_colorblind'][finger])
            
            # Create a patch (a rectangle in this case)
            rect = patches.Rectangle((-1, -1.2), 6, 2.4, linewidth=2, edgecolor='none', facecolor='#F0F0F0', label='zone')
            ax.add_patch(rect)

            # ax.legend(['','','f1', 'f2', 'f3', 'f4', 'f5'], loc='upper right', fontsize=my_paper['leg_fontsize'])
            ax.set_ylim([-6, 6])
            ax.set_xlim([t[0], t[-1]])
            ax.set_yticks(ticks=[-5,0,5])
            ax.set_xticks(ticks=[0, 1, 2, 3, 4])
            ax.set_ylabel('force [N]', fontsize=my_paper['label_fontsize'])
            ax.set_xlabel('time [s]', fontsize=my_paper['label_fontsize'])
            # trial day: 
            tr_day = df['day'].iloc[trial_idx]
            ax.set_title(f'trial {trial_idx}, day {tr_day}', fontsize=6)

            # Make it pretty:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

            ax.spines["left"].set_bounds(ax.get_ylim()[0], ax.get_ylim()[-1])
            ax.spines["bottom"].set_bounds(ax.get_xticks()[0], ax.get_xticks()[-1])

            ax.tick_params(axis='x', direction='in', length=2, width=my_paper['axis_width'])
            ax.tick_params(axis='y', direction='in', length=2, width=my_paper['axis_width'])
            ax.tick_params(axis='both', labelsize=my_paper['tick_fontsize'])

            # Show the plot
            plt.show()

            if export_fig:
                fig.savefig(os.path.join(FIGURE_PATH, f'efc2_example_{chord}_{trial_idx}.pdf'), format='pdf', bbox_inches='tight')

    return df, mov
    