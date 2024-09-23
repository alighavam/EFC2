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
                        marker='o', markersize=my_paper['marker_size'], linewidth=my_paper['line_width'], 
                        palette=[my_paper['color_untrained']], markeredgecolor=my_paper['color_untrained'], ax=ax)
    plt.errorbar(subset['adjusted_BN'], subset[measure+'_mean'], yerr=subset[measure+'_sem'],
                fmt='none', ecolor=my_paper['color_untrained'], capsize=0, elinewidth=my_paper['error_width'])

    # ========= Trained =========
    subset = grouped[(grouped['trained']==1)]
    sns.lineplot(data=subset, x='adjusted_BN', y=measure+'_mean', hue='day',
                        marker='o', markersize=my_paper['marker_size'], linewidth=my_paper['line_width'], 
                        palette=[my_paper['colors_blue'][3]], markeredgecolor=my_paper['colors_blue'][3], ax=ax)
    plt.errorbar(subset['adjusted_BN'], subset[measure+'_mean'], yerr=subset[measure+'_sem'],
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

    fig.savefig(os.path.join(FIGURE_PATH,'efc2_learning_'+measure+'.eps'), format='eps')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # STATS:
    

def learning_figure_avg(measure='MD', fig_size=[8.2, 6], show_plot=False):
    '''
    Creates and saves the figure for the learning across days. 

    1- Repetitions are averaged.
    2- The measures are averaged across participants. 
    3- Bars indicate the standard error of the mean.
    '''
    
    ana = pd.read_csv(os.path.join(ANALYSIS_PATH,'efc2_day.csv'))
    
    # average across subjects:
    grouped = ana.groupby(['day', 'trained']).agg(
        MD_mean=('MD', 'mean'),
        MD_sem=('MD', 'sem'),
        RT_mean=('RT', 'mean'),
        RT_sem=('RT', 'sem'),
        ET_mean=('ET', 'mean'),
        ET_sem=('ET', 'sem')
    ).reset_index()
    
    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(fig_size[0]*cm, fig_size[1]*cm))
    plt.rcParams['font.family'] = my_paper['font_family']
    
    # ========= Untrained =========
    subset = grouped[(grouped['trained']==0)]
    sns.lineplot(data=subset, x='adjusted_repetition', y=measure+'_mean', hue='day',
                        marker='o', markersize=my_paper['marker_size'], linewidth=my_paper['line_width'], 
                        palette=[my_paper['color_untrained']], markeredgecolor=my_paper['color_untrained'], ax=ax)
    plt.errorbar(subset['adjusted_repetition'], subset[measure+'_mean'], yerr=subset[measure+'_sem'],
                fmt='none', ecolor=my_paper['color_untrained'], capsize=0, elinewidth=my_paper['error_width'])

    # ========= Trained =========
    subset = grouped[(grouped['trained']==1)]
    sns.lineplot(data=subset, x='adjusted_repetition', y=measure+'_mean', hue='day',
                        marker='o', markersize=my_paper['marker_size'], linewidth=my_paper['line_width'], 
                        palette=[my_paper['colors_blue'][3]], markeredgecolor=my_paper['colors_blue'][3], ax=ax)
    plt.errorbar(subset['adjusted_repetition'], subset[measure+'_mean'], yerr=subset[measure+'_sem'],
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

    ax.set_xticks(ticks=[3, 8, 13, 18, 23], labels=['pre-test', '2', '3', '4', 'post-test'])
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

    fig.savefig(os.path.join(FIGURE_PATH,'efc2_learning_'+measure+'.eps'), format='eps')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


    # statistical tests:
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

def learning_repetition_figure(ana, measure='MD', fig_size=[8.2, 6], show_plot=False):
    

    # REPETITIONS SEPARATE:
    print('========= '+measure+' STATS: repetitions separate =========')
