import json
import unicodecsv as csv
from os import path
import numpy as np
from general.eval import *
from general.filePath import class_name
from general.trainTestSeparation import getRecordings

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
        # print ii_sdp_cpd
        # print len(state_dur_path)
        # print len(centroid_pho_durs)

        if len(state_dur_path) != len(centroid_pho_durs):
            sum_log_dur_prob = -float('inf')
        else:

            dur_probs = []
            for ii_dur_mean,dur_mean in enumerate(centroid_pho_durs):
                dur_decoded     = state_dur_path[ii_dur_mean][1]
                pho_dur_prob    = phoDurDist(dur_decoded,dur_mean,proportion_std = pstd)
                # if pho_dur_prob > 1:
                #     print dur_decoded, dur_mean, pho_dur_prob
                dur_probs.append(pho_dur_prob)
            # print dur_probs
            # print [np.log(dp) for dp in dur_probs]

            sum_log_dur_prob = np.sum(dur_probs)/len(centroid_pho_durs)

            # print 'sum log dur prob:',sum_log_dur_prob
            # print len(centroid_pho_durs), len(state_dur_path)
            # print centroid_pho_durs
            # print state_dur_path
        list_sum_log_dur_probs.append(sum_log_dur_prob)
    return list_sum_log_dur_probs

def evalOnSinglefilePostProcessor(filename,
                                  lyrics,
                                  posteri_probs,
                                  sum_log_dur_probas,
                                  phrases,
                                  query_lyrics,
                                  coef_post_processor,
                                    results_path,
                                  writecsv=False):

    '''
    merge posteri_probabilities of viterbi decoding with duration probability post-processor modelling

    :param filename:
    :param lyrics:
    :param posteri_probs:
    :param sum_log_dur_probas:
    :param phrases:
    :param query_lyrics:
    :param coef_post_processor:
    :param writecsv:
    :return: ranking order of the ground truth score
    '''

    # print query_lyrics
    # print posteri_probs
    # print lyrics

    # sort
    probas_merge    = (np.array(posteri_probs)+coef_post_processor*np.array(sum_log_dur_probas))
    # probas_merge    = np.array(posteri_probs)
    ppinds          = probas_merge.argsort()
    # posteri_probs_sorted = posteri_probs[ppinds[::-1]]
    probas_merge    = probas_merge[ppinds[::-1]]
    lyrics_sorted   = [lyrics[i] for i in ppinds[::-1]]
    phrases_sorted  = [phrases[i] for i in ppinds[::-1]]

    list_sdist = []
    for l in lyrics_sorted:
        sdist = stringDist(query_lyrics,l)
        list_sdist.append(sdist)

    order = list_sdist.index(max(list_sdist))

    if writecsv:
        with open(path.join(results_path,filename+'.csv'),'wb') as csvfile:
            w = csv.writer(csvfile)
            w.writerow([query_lyrics,str(order),lyrics_sorted[order]])
            for ii in range(len(lyrics_sorted)):
                w.writerow([phrases_sorted[ii],lyrics_sorted[ii],probas_merge[ii],list_sdist[ii]])

    order += 1

    return order

def evalOnSinglefileHsmm(filename,
                         lyrics,
                         posteri_probs,
                         phrases,
                         query_lyrics,
                         results_path,
                         writecsv=False):

    '''
    merge posteri_probabilities of viterbi decoding with duration probability post-processor modelling

    :param filename:
    :param lyrics:
    :param posteri_probs:
    :param phrases:
    :param query_lyrics:
    :param writecsv:
    :return: ranking order of the ground truth score
    '''

    # print query_lyrics
    # print posteri_probs
    # print lyrics

    # sort
    posteri_probs_np = np.array(posteri_probs)
    ppinds          =  posteri_probs_np.argsort()
    # posteri_probs_sorted = posteri_probs[ppinds[::-1]]
    posteri_probs_np    = posteri_probs_np[ppinds[::-1]]
    lyrics_sorted   = [lyrics[i] for i in ppinds[::-1]]
    phrases_sorted  = [phrases[i] for i in ppinds[::-1]]

    list_sdist = []
    for l in lyrics_sorted:
        sdist = stringDist(query_lyrics,l)
        list_sdist.append(sdist)

    order = list_sdist.index(max(list_sdist))

    if writecsv:
        with open(path.join(results_path,filename+'.csv'),'wb') as csvfile:
            w = csv.writer(csvfile)
            w.writerow([query_lyrics,str(order),lyrics_sorted[order]])
            for ii in range(len(lyrics_sorted)):
                w.writerow([phrases_sorted[ii],lyrics_sorted[ii],posteri_probs_np[ii],list_sdist[ii]])

    order += 1

    return order

def calculateMetrics(list_rank,
                     string_lyricsRecognizer,
                     coef_post_processor,
                     proportion_std):
    mrr = MRR(list_rank)
    top1hit = topXhit(1,list_rank)
    top3hit = topXhit(3,list_rank)
    top5hit = topXhit(5,list_rank)
    top10hit = topXhit(10,list_rank)
    top20hit = topXhit(20,list_rank)
    top100hit = topXhit(100,list_rank)

    path_eval = path.join(currentPath,
                          '..',
                          'eval',
                          class_name+'_'+string_lyricsRecognizer+'_'+str(coef_post_processor)+'_'+str(proportion_std)+'.csv')

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

def resultAnalysisProcess(method,proportionality_std):
    # for cpp in [0.5,1,2,5,10]:

    if method == 'lyricsRecognizerHmm':
        path_json_dict_query_phrases = 'results/dict_query_phrases_' \
                                       + method + '_' \
                                       + class_name + '.json'
        results_path = path.join(currentPath, '..', 'results/lyricsRecognizerHmmDan')

    else:
        path_json_dict_query_phrases = 'results/dict_query_phrases_' \
                                       + method + '_' \
                                       + class_name + '_' \
                                       + str(proportionality_std) + '.json'
        results_path = path.join(currentPath, '..', 'results/lyricsRecognizerHsmm')

    with open(path_json_dict_query_phrases,'r') as outfile:
        dict_query_phrases = json.load(outfile)

    # filenames = getRecordings(results_path)

    # best cpp = 1.0, pstd = 1.25 hmm
    # best proportionality std = 0.5 until now
    if method == 'lyricsRecognizerHMM':

        cpp = 1.0
        for pstd in [0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]:
            list_rank = []
            for key in dict_query_phrases:

                dqp                                     = dict_query_phrases[key]

                query_phrase_name                       = dqp['query_phrase_name']
                lyrics_net                              = dqp['lyrics_net']
                posteri_probas                          = dqp['posteri_probas']
                phrases                                 = dqp['phrases']
                line_lyrics                             = dqp['line_lyrics']

                # print lyrics_net[79]
                # print phrases[79]


                list_state_dur_path_centroid_pho_durs    = dqp['list_state_dur_path_centroid_pho_durs']
                list_sum_log_dur_probs                   = sumLogDurProbs(list_state_dur_path_centroid_pho_durs, pstd)
                order = evalOnSinglefilePostProcessor(query_phrase_name,
                                                      lyrics_net,
                                                      posteri_probas,
                                                      list_sum_log_dur_probs,
                                                      phrases,
                                                      line_lyrics,
                                                      coef_post_processor=cpp,
                                                      results_path=results_path,
                                                      writecsv=False)
                list_rank.append(order)
            calculateMetrics(list_rank, method, cpp, pstd)


        else:
            list_rank = []
            for key in dict_query_phrases:
                dqp = dict_query_phrases[key]

                query_phrase_name = dqp['query_phrase_name']
                lyrics_net = dqp['lyrics_net']
                posteri_probas = dqp['posteri_probas']
                phrases = dqp['phrases']
                line_lyrics = dqp['line_lyrics']

                order = evalOnSinglefileHsmm(query_phrase_name,
                                         lyrics_net,
                                         posteri_probas,
                                         phrases,
                                         line_lyrics,
                                        results_path=results_path,
                                         writecsv=False)
                list_rank.append(order)
                calculateMetrics(list_rank, method, 0, 0)
        """
        list_less10Rank = lessNRank(dict_query_phrases.keys(), list_rank,N=3, writecsv=True)

        path_largerPyin = path.join(currentPath,'../..','melodicSimilarity','errorAnalysis/larger3pyinRoletypeWeight.csv')

        print compareMelodicSimiResults(path_largerPyin,[llr[0] for llr in list_less10Rank])
        """