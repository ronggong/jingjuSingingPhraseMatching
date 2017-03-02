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

import json
import unicodecsv as csv
import numpy as np
from os import path, makedirs
from general.eval import *
from general.filePath import class_name

from scipy.stats import norm

currentPath = path.dirname(__file__)

def phoDurDist(x,dur_mean,proportion_std = 0.35):
    '''
    build gaussian distribution which has mean = dur_mean, std = dur_mean*proportion_std
    :param dur_mean: estimated from the centroid duration
    :param proportion_std:
    :return:
    '''
    prob = norm.logpdf(x,dur_mean,dur_mean*proportion_std)
    return prob

def sumLogDurProbs(list_state_dur_path_centroid_pho_durs,pstd):
    '''
    post processer duration probabilities

    :param list_state_dur_path_centroid_pho_durs: N_path*2, [[state_duration_path,centroid_pho_durations],...]
    :param pstd: proporation std to mean duration
    :return:
    '''

    list_sum_log_dur_probs = []
    for ii_sdp_cpd, sdp_cpd in enumerate(list_state_dur_path_centroid_pho_durs):

        state_dur_path = sdp_cpd[0]
        centroid_pho_durs = sdp_cpd[1]

        if len(state_dur_path) != len(centroid_pho_durs):
            sum_log_dur_prob = -float('inf')
        else:

            dur_probs = []
            for ii_dur_mean,dur_mean in enumerate(centroid_pho_durs):
                dur_decoded     = state_dur_path[ii_dur_mean][1]
                pho_dur_prob    = phoDurDist(dur_decoded,dur_mean,proportion_std = pstd)
                dur_probs.append(pho_dur_prob)

            sum_log_dur_prob = np.sum(dur_probs)/len(centroid_pho_durs)

        list_sum_log_dur_probs.append(sum_log_dur_prob)
    return list_sum_log_dur_probs

def evalOnSinglefile(filename,
                     lyrics,
                     posteri_probs,
                     sum_log_dur_probas,
                     phrases,
                     query_lyrics,
                     coef_post_processor,
                     results_path,
                     hsmm=False,
                     writecsv=False):

    '''
    hsmm=False: merge posteri_probabilities of viterbi decoding with duration probability post-processor modelling
    hsmm=True:  not merge probability

    writecsv: write the result of each query in a csv file

    :param filename:
    :param lyrics:
    :param posteri_probs:
    :param sum_log_dur_probas: not used in hsmm=True case
    :param phrases:
    :param query_lyrics:
    :param coef_post_processor: not used in hsmm=True case
    :param hsmm:
    :param writecsv:
    :return: ranking order of the ground truth score
    '''

    # assign posteri probabilities
    if not hsmm:
        posteri_probs_post    = (np.array(posteri_probs)+coef_post_processor*np.array(sum_log_dur_probas))
    else:
        posteri_probs_post    = np.array(posteri_probs)

    # sort matching lyrics and phrases according to posteri probabilites sorted indices
    ppinds          = posteri_probs_post.argsort()
    posteri_probs_post    = posteri_probs_post[ppinds[::-1]]
    lyrics_sorted   = [lyrics[i] for i in ppinds[::-1]]
    phrases_sorted  = [phrases[i] for i in ppinds[::-1]]

    # find the the rank of the ground-truth score phrase
    list_sdist = []
    for l in lyrics_sorted:
        sdist = stringDist(query_lyrics,l)
        list_sdist.append(sdist)

    order = list_sdist.index(max(list_sdist))

    # write csv file
    if writecsv:
        # create results_path if it doesn't exist
        if not path.isdir(results_path):
            makedirs(results_path)

        with open(path.join(results_path,filename+'.csv'),'wb') as csvfile:
            w = csv.writer(csvfile)
            w.writerow([query_lyrics,str(order),lyrics_sorted[order]])
            for ii in range(len(lyrics_sorted)):
                w.writerow([phrases_sorted[ii],lyrics_sorted[ii],posteri_probs_post[ii],list_sdist[ii]])

    order += 1

    return order


def calculateMetrics(list_rank,
                     string_lyricsRecognizer,
                     coef_post_processor,
                     proportion_std,
                     am):
    """
    Calculate matching evaluation metrics
    If HSMMs is evaluated, set coef_post_processor=0

    :param list_rank:
    :param string_lyricsRecognizer:
    :param coef_post_processor:
    :param proportion_std:
    :param am:
    :return:
    """
    mrr = MRR(list_rank)
    top1hit = topXhit(1,list_rank)
    top3hit = topXhit(3,list_rank)
    top5hit = topXhit(5,list_rank)
    top10hit = topXhit(10,list_rank)
    top20hit = topXhit(20,list_rank)
    top100hit = topXhit(100,list_rank)

    # write results into csv
    path_eval = path.join(currentPath,
                          '..',
                          'eval',
                          class_name+'_'+am+'_'+string_lyricsRecognizer+'_'+str(coef_post_processor)+'_'+str(proportion_std)+'.csv')

    with open(path_eval,'wb') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['MRR',mrr])
        w.writerow(['top 1 hit',top1hit])
        w.writerow(['top 3 hit',top3hit])
        w.writerow(['top 5 hit',top5hit])
        w.writerow(['top 10 hit',top10hit])
        w.writerow(['top 20 hit',top20hit])
        w.writerow(['top 100 hit',top100hit])

def lessNRank(query_phrase_names, list_rank, N=3, writecsv=False):
    '''
    find the phrase name and groundtruth ranking <= N
    :param query_phrase_names:
    :param list_rank:
    :param N:
    :param writecsv:
    :return:
    '''

    list_lessNRank = []
    for ii in xrange(len(query_phrase_names)):
        phrase_name = query_phrase_names[ii]
        ranking = list_rank[ii]
        if ranking <= N:
            list_lessNRank.append([phrase_name,ranking])

    if writecsv and len(list_lessNRank):
        path_less3Lyrics = path.join(currentPath,'..','errorAnalysis/less'+str(N)+'.csv')
        with open(path_less3Lyrics,'wb') as csvfile:
            w = csv.writer(csvfile)
            for lN in list_lessNRank:
                w.writerow(lN)
    return list_lessNRank

def compareMelodicSimiResults(path_largerPyin,list_lessNRank_phrase_name):
    '''
    compare with the results of melodic similarity
    find the intersection set, melodic similarity ranking > N, phonetic similarity ranking < N
    :param path_largerPyin: path of the melodic similarity csv
    :param list_lessNRank_phrase_name: ranking less than N phrase name by phonetic similarity
    :return: intersection set of the phrase name
    '''
    phrase_names_largerN = []
    with open(path_largerPyin,'r') as openfile:
        csv_reader = csv.reader(openfile,delimiter=',')
        for row in csv_reader:
            phrase_names_largerN.append(row[0])

    return set.intersection(set(phrase_names_largerN),set(list_lessNRank_phrase_name))

def resultAnalysisProcess(method, proportionality_std=0, path_json_dict_query_phrases='dummy', am='gmm', cnn_file_name=''):

    # path for storing matching ranking result csv
    results_path = path.join(currentPath, '..', 'results/'+method+'_'+class_name)

    with open(path_json_dict_query_phrases,'r') as outfile:
        dict_query_phrases = json.load(outfile)


    if method == 'lyricsRecognizerHMM':
        # best cpp = 1.0, pstd = 0.7 for dan
        # best cpp = 1.0, pstd = 1.5 for laosheng

        # pstd = 0 means no post-processor duration modelling
        if class_name == 'danAll':
            pstd_options = [0, 0.7]
        elif class_name == 'laosheng':
            pstd_options = [0, 1.5]

        for cpp in [1.0]:
            # for pstd in np.linspace(0.0,2.0,21):
            for pstd in pstd_options:

                list_rank = []
                for key in dict_query_phrases:

                    dqp                                     = dict_query_phrases[key]
                    query_phrase_name                       = dqp['query_phrase_name']
                    lyrics_net                              = dqp['lyrics_net']
                    posteri_probas                          = dqp['posteri_probas']
                    phrases                                 = dqp['phrases']
                    line_lyrics                             = dqp['line_lyrics']
                    list_state_dur_path_centroid_pho_durs   = dqp['list_state_dur_path_centroid_pho_durs']
                    list_sum_log_dur_probs                  = sumLogDurProbs(list_state_dur_path_centroid_pho_durs, pstd)

                    if pstd == 0.0:
                        # without any duration modelling
                        order = evalOnSinglefile(filename               =query_phrase_name,
                                                   lyrics               =lyrics_net,
                                                   posteri_probs        =posteri_probas,
                                                   sum_log_dur_probas   =[],
                                                   phrases              =phrases,
                                                   query_lyrics         =line_lyrics,
                                                   coef_post_processor  =0,
                                                   hsmm                 =True,
                                                   results_path         =results_path,
                                                   writecsv             =True
                                                 )
                    else:
                        # post-processor duration modelling
                        order = evalOnSinglefile(filename               =query_phrase_name,
                                                 lyrics                 =lyrics_net,
                                                 posteri_probs          =posteri_probas,
                                                 sum_log_dur_probas     =list_sum_log_dur_probs,
                                                 phrases                =phrases,
                                                 query_lyrics           =line_lyrics,
                                                 coef_post_processor    =cpp,
                                                 results_path           =results_path,
                                                 hsmm                   =False,
                                                 writecsv               =False
                                                 )
                    list_rank.append(order)
                calculateMetrics(list_rank, method, cpp, pstd, am + cnn_file_name)

    elif method == 'lyricsRecognizerHSMM':
        list_rank = []
        for key in dict_query_phrases:
            dqp = dict_query_phrases[key]

            query_phrase_name   = dqp['query_phrase_name']
            lyrics_net          = dqp['lyrics_net']
            posteri_probas      = dqp['posteri_probas']
            phrases             = dqp['phrases']
            line_lyrics         = dqp['line_lyrics']

            order = evalOnSinglefile(filename=query_phrase_name,
                                       lyrics=lyrics_net,
                                       posteri_probs=posteri_probas,
                                       sum_log_dur_probas=[],
                                       phrases=phrases,
                                       query_lyrics=line_lyrics,
                                       coef_post_processor=0,
                                       results_path=results_path,
                                       hsmm=True,
                                       writecsv=False)

            list_rank.append(order)
            calculateMetrics(list_rank, method, 0, proportionality_std, am)

        """
        Not be used
        list_less10Rank = lessNRank(dict_query_phrases.keys(), list_rank,N=3, writecsv=True)

        path_largerPyin = path.join(currentPath,'../..','melodicSimilarity','errorAnalysis/larger3pyinRoletypeWeight.csv')

        print compareMelodicSimiResults(path_largerPyin,[llr[0] for llr in list_less10Rank])
        """