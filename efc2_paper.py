import os
import numpy as np
import pandas as pd
import utils
from utils.config import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.figure_style import my_paper
from statsmodels.stats.anova import AnovaRM
from scipy import stats

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
    plt.rcParams['font.family'] = my_paper['font_family']
    
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
        ax.set_ylim([0, 4200])
        ax.set_yticks(ticks=[0, 2100, 4200], labels=['0','2.1','4.2'])
        ax.set_ylabel('execution time [s]')

    ax.set_xticks(ticks=[4.5, 12.5, 20.5, 28.5, 36.5], labels=['pre-test', '2', '3', '4', 'post-test'])
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

    fig.savefig(os.path.join(FIGURE_PATH,'efc2_learning_'+measure+'.svg'), format='svg')
    
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

    
def learning_figure_avg(measure='MD', fig_size=[4, 6], show_plot=False):
    '''
    Performance improvement over days with BNs averaged.
    '''
    
    df = pd.read_csv(os.path.join(ANALYSIS_PATH, 'efc2_all.csv'))

    # make summary dataframe:
    df.replace(-1, np.nan, inplace=True)
    ANA = pd.DataFrame()
    ANA = df.groupby(['day','sn','trained'])[['is_test','group','RT','ET','MD']].mean().reset_index()

    # average across subjects:
    grouped = ANA.groupby(['day', 'trained']).agg(
        MD_mean=('MD', 'mean'),
        MD_sem=('MD', 'sem'),
        RT_mean=('RT', 'mean'),
        RT_sem=('RT', 'sem'),
        ET_mean=('ET', 'mean'),
        ET_sem=('ET', 'sem'),
        len_avg=('MD', 'count')
    ).reset_index()

    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(fig_size[0]*cm, fig_size[1]*cm))
    plt.rcParams['font.family'] = my_paper['font_family']
    
    # ========= Untrained =========
    subset = grouped[(grouped['trained']==0)]
    sns.lineplot(data=subset, x='day', y=measure+'_mean',
                 marker='o', markersize=my_paper['marker_size'], linewidth=my_paper['line_width'], 
                 color=my_paper['color_untrained'], markeredgecolor=my_paper['color_untrained'], ax=ax)
    plt.errorbar(subset['day'], subset[measure+'_mean'], yerr=subset[measure+'_sem'],
                 fmt='none', ecolor=my_paper['color_untrained'], capsize=0, elinewidth=my_paper['error_width'])

    # ========= Trained =========
    subset = grouped[(grouped['trained']==1)]
    sns.lineplot(data=subset, x='day', y=measure+'_mean',
                 marker='o', markersize=my_paper['marker_size'], linewidth=my_paper['line_width'], 
                 color=my_paper['colors_blue'][3], markeredgecolor=my_paper['colors_blue'][3], ax=ax)
    plt.errorbar(subset['day'], subset[measure+'_mean'], yerr=subset[measure+'_sem'],
                 fmt='none', ecolor=my_paper['colors_blue'][3], capsize=0, elinewidth=my_paper['error_width'])
    
    if measure=='MD':
        ax.set_ylim([0, 2.2])
        ax.set_yticks(ticks=[0, 1, 2])
        ax.set_ylabel('mean deviation [N]')
    elif measure=='RT':
        ax.set_ylim([200, 900])
        ax.set_yticks(ticks=[200, 400, 600, 800])
        ax.set_ylabel('reaction time [ms]')
    elif measure=='ET':
        ax.set_ylim([0, 5000])
        ax.set_yticks(ticks=[0, 2100, 4200], labels=['0','2.1','4.2'])
        ax.set_ylabel('execution time [s]')

    ax.set_xticks(ticks=[1,2,3,4,5], labels=['pre-test', '2', '3', '4', 'post-test'])
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

    fig.savefig(os.path.join(FIGURE_PATH,'efc2_learning_'+measure+'_avg.svg'), format='svg')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # STATS:

def learning_repetition_figure(ana, measure='MD', fig_size=[8.2, 6], show_plot=False):
    

    # REPETITIONS SEPARATE:
    print('========= '+measure+' STATS: repetitions separate =========')

    # REPETITIONS AVERAGED:
    print('========= '+measure+' STATS: repetitions averaged =========')
    grouped = ana.groupby(['day', 'trained', 'sn']).mean(['MD','RT','ET']).reset_index()

    # Percent improvement from day 1 to day 5:
    print(f'Percent improvement from day 1 to day 5:')
    imprv_trained = grouped[(grouped['day']==1) & (grouped['trained']==1)][measure] - grouped[(grouped['day']==5) & (grouped['trained']==1)][measure]
    imprv_untrained = grouped[(grouped['day']==1) & (grouped['trained']==0)][measure] - grouped[(grouped['day']==5) & (grouped['trained']==0)][measure]
    print()

    # ANOVA day1 vs. day5:
    tmp_df = grouped[(grouped['day']==1) | (grouped['day']==5)]
    anova = AnovaRM(data=tmp_df, depvar=measure, subject='sn', within=['day', 'trained']).fit()
    anova_table = anova.anova_table
    anova_table['Pr > F'] = anova_table['Pr > F'].apply(lambda p: f'{p:.6f}')
    anova_table['F Value'] = anova_table['F Value'].apply(lambda f: f'{f:.6f}')
    # Print the formatted table
    print(anova_table)
    print()

    # t-tests:
    print('day 1, trained vs. untrained:')
    day1_trained = grouped[(grouped['day']==1) & (grouped['trained']==1)][measure]
    day1_untrained = grouped[(grouped['day']==1) & (grouped['trained']==0)][measure]
    res = stats.ttest_rel(day1_trained, day1_untrained)
    print(f't_{len(day1_trained)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')

    print('day 5, trained vs. untrained:')
    day5_trained = grouped[(grouped['day']==5) & (grouped['trained']==1)][measure]
    day5_untrained = grouped[(grouped['day']==5) & (grouped['trained']==0)][measure]
    res = stats.ttest_rel(day5_trained, day5_untrained)
    print(f't_{len(day5_trained)-1} = {res.statistic:.3f}, p = {res.pvalue:.6f}')