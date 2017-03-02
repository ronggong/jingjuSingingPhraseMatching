'''
 * Copyright (C) 2017  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuSingingPhraseMatching
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''

"""
Code to plot the results
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ylables = ['Percentage (%)', 'Percentage (%)']

# 0 line dan, 1 line laosheng

hmm =[[ 25.9119080372,	15.7303370787,	28.0898876404,	34.8314606742,	47.191011236,	61.797752809],
 [ 48.8940223178,	37.5,	50,	63.6363636364,	73.8636363636,	81.8181818182]]

# HSMM
hsmm = [[38.2985903764,	23.595505618,	47.191011236,	57.3033707865,	65.1685393258,	71.9101123596],
[64.0407508196,	55.6818181818,	67.0454545455,	72.7272727273,	79.5454545455,	82.9545454545]]

# post-processor
post = [[27.1066275139,	16.8539325843,	29.2134831461,	34.8314606742,	50.5617977528,	64.0449438202],
[49.0806286984,	37.5,	52.2727272727,	63.6363636364,	73.8636363636,	81.8181818182]]


colors = ['g','k','b','r']
markers = ['*','^','x','D']
linestyles = [':','--','-.','-']

labels = ['Baseline-HMMs', 'HSMMs', 'Post-processor']

df0 = pd.DataFrame(np.matrix([hmm[0],hsmm[0],post[0]]).T, columns=labels,
                  index=pd.Index(['MRR',
                                'Top-1 hit',
                                'Top-3 hit',
                                'Top-5 hit',
                                'Top-10 hit',
                                'Top-20 hit'],
                  name='Index'))
df1 = pd.DataFrame(np.matrix([hmm[1],hsmm[1],post[1]]).T, columns=labels,
                   index=pd.Index(['MRR',
                                   'Top-1 hit',
                                   'Top-3 hit',
                                   'Top-5 hit',
                                   'Top-10 hit',
                                   'Top-20 hit'],
                  name='Index'))

def overlapped_bar(df, show=False, width=0.9, alpha=1.0,
                   title='', xlabel='', ylabel='', **plot_kwargs):
    """Like a stacked bar chart except bars on top of each other with transparency"""
    xlabel = xlabel or df.index.name
    N = len(df)
    M = len(df.columns)
    indices = np.arange(N)
    colors = ['steelblue', 'firebrick', 'darksage', 'goldenrod', 'gray'] * int(M / 5. + 1)
    for i, label, color in zip(range(M), df.columns, colors):
        kwargs = plot_kwargs
        kwargs.update({'color': color, 'label': label})
        plt.bar(indices, df[label], width=width, alpha=alpha if i else 1, **kwargs)
        plt.xticks(indices + .5 * width,
                   ['{}'.format(idx) for idx in df.index.values])
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    if show:
        plt.show()
    return plt.gcf()

ind = np.arange(len(hmm[0]))
width = 0.25
ms = 10
fontsize = 15

# for color,marker,label in zip(colors,markers,labels):
#     plt.plot(ind, df[label], color=color, marker = marker, markersize = ms,label=label)
#     plt.xticks(ind,['{}'.format(idx) for idx in df.index.values])
# # plt.text(np.argmax(f)-0.6,np.max(f)+3,str(np.max(f)),fontsize=fontsize)
# plt.legend(loc='best')
# plt.ylabel(ylables[idx_measure],fontsize=fontsize)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

f, axarr = plt.subplots(2, figsize = (8,4), sharex=True)

for color,marker,label,linestyle in zip(colors,markers,labels,linestyles):
    axarr[0].plot(ind, df0[label], color=color, marker = marker, markersize = ms,label=label,linestyle=linestyle)
    axarr[0].set_ylabel(ylables[0],fontsize=fontsize)
    axarr[0].xaxis.grid(True)


    # plt.xticks(ind,['{}'.format(idx) for idx in df.index.values])

for color,marker,label,linestyle in zip(colors,markers,labels,linestyles):
    axarr[1].plot(ind, df1[label], color=color, marker = marker, markersize = ms,label=label,linestyle=linestyle)
    axarr[1].set_ylabel(ylables[1],fontsize=fontsize)
    axarr[1].xaxis.grid(True)

    plt.xticks(ind,['{}'.format(idx) for idx in df0.index.values])
    axarr[1].set_xlabel('Evaluation metrics',fontsize=fontsize)


axarr[0].legend(loc='upper center', bbox_to_anchor=(0.48, 1.3),
          fancybox=True, shadow=True, ncol=3)

plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
# plt.ylabel(ylables[0], fontsize=fontsize)
plt.tight_layout()

plt.show()

