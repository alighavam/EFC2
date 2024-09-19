import os
import numpy as np
import pandas as pd
import utils
from utils.config import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.figure_style import my_paper

def learning_figure(ana, measure='MD', fig_size=[8.2, 6], show_plot=False):
    '''
    Creates and saves the figure for the learning across days. 

    1- Repetitions are plotted as separate points.
    2- The measures are averaged across participants. 
    3- Bars indicate the standard error of the mean.
    '''
    
    # average across subjects:
    grouped = ana.groupby(['day', 'trained', 'adjusted_repetition']).agg(
        MD_mean=('MD', 'mean'),
        MD_sem=('MD', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        RT_mean=('RT', 'mean'),
        RT_sem=('RT', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        ET_mean=('ET', 'mean'),
        ET_sem=('ET', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
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
