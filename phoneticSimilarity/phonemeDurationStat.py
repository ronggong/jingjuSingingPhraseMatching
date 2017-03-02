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

from general.trainTestSeparation import getRecordingNamesSimi
from general.textgridParser import syllableTextgridExtraction
import matplotlib.pyplot as plt
from scipy.misc import factorial
from scipy.optimize import curve_fit
from scipy.stats import gamma,expon
from general.filePath import *
from general.parameters import *
from general.phonemeMap import dic_pho_map
import json
import numpy as np
import os


def phoDurCollection(recordings):
    '''
    collect durations of pho into dictionary
    :param recordings:
    :return:
    '''
    dict_duration_pho = {}
    for recording in recordings:

        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)

        for pho in nestedPhonemeLists:
            for p in pho[1]:
                dur_pho = p[1] - p[0]
                sampa_pho = dic_pho_map[p[2]]

                if sampa_pho not in dict_duration_pho.keys():
                    dict_duration_pho[sampa_pho] = [dur_pho]
                else:
                    dict_duration_pho[sampa_pho].append(dur_pho)
    return dict_duration_pho

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def durPhoDistribution(array_durPho,sampa_pho,plot=False):
    '''
    pho durations histogram
    :param array_durPho:
    :return:
    '''

    # plt.figure(figsize=(10, 6))

    # integer bin edges
    offset_bin = 0.005
    bins = np.arange(0, max(array_durPho)+2, 2*offset_bin) - offset_bin

    # histogram
    entries, bin_edges, patches = plt.hist(array_durPho, bins=bins, normed=True, fc=(0, 0, 1, 0.7),label='pho: '+sampa_pho+' duration histogram')

    # centroid duration
    bin_centres = bin_edges-offset_bin
    bin_centres = bin_centres[:-1]
    centroid = np.sum(bin_centres*entries)/np.sum(entries)

    ##-- fit with poisson distribution
    # bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
    #
    # parameters, cov_matrix = curve_fit(poisson, bin_middles, entries)
    #
    # x = np.linspace(0, max(array_durPho), 1000)
    # x = np.arange(0,max(array_durPho),hopsize_t)
    #
    # p = poisson(x, *parameters)

    ##-- fit with gamma distribution

    # discard some outlier durations by applying 2 standard deviations interval
    mean_array_durPho=np.mean(array_durPho)
    std_array_durPho=np.std(array_durPho)
    index_keep = np.where(array_durPho<mean_array_durPho+2*std_array_durPho)
    array_durPho_keep = array_durPho[index_keep]

    # discard some duration in histogram to make the fitting reasonable
    if class_name == 'laosheng':
        if sampa_pho == 'in':
            array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<2.5)]
        elif sampa_pho == '@n':
            array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<3)]
        elif sampa_pho == 'eI^':
            array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<1.5)]
        elif sampa_pho == 'EnEn':
            array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<2.0)]
        elif sampa_pho == 'UN':
            array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<2.5)]

    # step is the hopsize_t, corresponding to each frame
    # maximum length is the 4 times of the effective length
    x = np.arange(0, 8*max(array_durPho_keep),hopsize_t_phoneticSimilarity)

    param   = gamma.fit(array_durPho_keep,floc = 0)
    y       = gamma.pdf(x, *param)
    # y = expon.pdf(x)

    if plot:
        # possion fitting curve
        # plt.plot(x,p,'r',linewidth=2,label='Poisson distribution fitting curve')

        # gamma fitting curve
        # plt.plot(x, y, 'r-', lw=2, alpha=0.6, label='gamma pdf')
        plt.axvline(centroid, linewidth = 3, color = 'r', label = 'centroid frequency')
        plt.legend(fontsize=18)
        plt.xlabel('Pho duration distribution ',fontsize=18)
        plt.ylabel('Probability',fontsize=18)
        plt.axis('tight')
        plt.tight_layout()
        plt.show()

    y /= np.sum(y)

    return y.tolist(),centroid

if __name__ == '__main__':

    rp = os.path.dirname(__file__)

    for cn in ['danAll', 'laosheng']:

        recordings_train = getRecordingNamesSimi('TRAIN',cn)
        dict_duration_pho = phoDurCollection(recordings_train)

        dict_centroid_dur = {}
        dict_dur_dist = {}
        for pho in dict_duration_pho:
            durDist,centroid_dur = durPhoDistribution(np.array(dict_duration_pho[pho]),pho,plot=False)
            dict_centroid_dur[pho]  = centroid_dur
            dict_dur_dist[pho]      = durDist # the first proba is always 0

        # dump duration centroid
        with open(os.path.join(rp, 'lyricsRecognizer' ,'dict_centroid_dur'+cn+'.json'),'wb') as outfile:
            json.dump(dict_centroid_dur,outfile)

        # the gamma occupancy duration distribution is never used
        # with open('dict_dur_dist_'+class_name+'.json','wb') as outfile:
        #     json.dump(dict_dur_dist,outfile)

