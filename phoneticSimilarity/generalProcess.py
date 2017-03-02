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
This .py contains the processing pipline for two methods: HMM and HSMM.

post-processor duration modelling is not included here instead in lyricsRecognizer/resultAnalysis.py
because it's a kind of post-processing after the state sequence decoding has been done

generalProcess funtion gets input of incoming query audio files, segment them into phrase
phraseProcess function takes over these segmented phrases and do matching

There are some obsolete method here, 'obsMatrix', 'candidateSynthesizeData'
These are the previous experimentation of different matching methods, which didn't work

The output ranking results is stored in path_json_dict_query_phrases
"""

from os import path
import sys
import json
import pickle
import numpy as np

currentPath = path.dirname(path.abspath(__file__))
lyricsRecognizerPath = path.join(currentPath, 'lyricsRecognizer')
sys.path.append(lyricsRecognizerPath)

# from general.trainTestSeparation import getRecordingNamesSimi
from general.textgridParser import syllableTextgridExtraction
from general.parameters import list_N_frames,hopsize_phoneticSimilarity
from targetAudioProcessing import gmmPhoModelLoad,processFeature,obsMatrixPho
from general.filePath import *

from ParallelLRHMM import ParallelLRHMM
from makeNet import makeNet

import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from ParallelLRHSMM import ParallelLRHSMM
from makeHSMMNet import makeHSMMNet

from scoreDictFilter import runDictScoreRankNFilter
from scipy.io import wavfile

# from scoreManip import scoreMSynthesize,scoreMSynthesizePho,mfccSynthesizeFromData,mfccSynthesizeFromGMM,plotMFCC
# from tailsMFCCTrain import loadMFCCTrain
# from general.dtwSankalp import dtwNd
# from targetAudioProcessing import gmmModelLoad,obsMPlot,obsMPlotPho,obsMatrix
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
# from operator import itemgetter
# from resultAnalysis import evalOnSinglefilePostProcessor,sumLogDurProbs,calculateMetrics

# load score json, pre-computed
with open(path.join(currentPath, '../melodicSimilarity/scores.json'),'r') as f:
    dict_score = json.load(f)

dist_measures   = ['euclideanDist','sankalpNdDTW121']
dm              = dist_measures[1]

# load textgrid, wav paths, load melodic similarity results
if class_name == 'danAll':
    textgridDataDir = textgrid_path_dan
    wavDataDir = wav_path_dan
    path_melodic_similarity_results = path.join(currentPath, '..', 'melodicSimilarity', 'results',
                                                'danAll_900_0.7_pyin_roleTypeWeight')

elif class_name == 'laosheng':
    textgridDataDir = textgrid_path_laosheng
    wavDataDir = wav_path_laosheng
    path_melodic_similarity_results = path.join(currentPath,'..','melodicSimilarity','results','900_0.7_pyin')

def phraseProcess(filename,
                  i,
                  line_list,
                  wavData,
                  gmmModel,
                  kerasModel,
                  sampleRate,
                  method,
                  proportionality_std=0):
    # sub module to process each phrase

    line = line_list[0]
    start_frame = int(round(line[0] * sampleRate))
    end_frame = int(round(line[1] * sampleRate))
    line_lyrics = line[2]

    wav_line = wavData[start_frame:end_frame]
    wavfile.write('temp.wav', sampleRate, wav_line)

    # choose feature type as mfcc or mfccBands for GMM or DNN
    if am == 'gmm':
        mfcc_target = processFeature('temp.wav', feature_type='mfcc')
    elif am == 'cnn':
        mfcc_target = processFeature('temp.wav', feature_type='mfccBands2D')

    N_frame         = mfcc_target.shape[0]
    duration_target = (N_frame * hopsize_phoneticSimilarity) / float(sampleRate)

    # only examine on the first N ranking results of melodic similarity
    query_phrase_name   = filename + '_' + str(i)
    dict_score_100      = runDictScoreRankNFilter(dict_score,
                                                 path_melodic_similarity_results,
                                                 query_phrase_name,
                                                 N=100)
    # print 'dict_score_100 len:',len(dict_score_100)

    if method == 'obsMatrix':
        # not used
        obsM = obsMatrixPho(mfcc_target, gmmModel)  # pho gmm
        obsM = np.exp(obsM)
        obsM = obsM / np.sum(obsM, axis=0)
        print obsM
    elif method == 'candidateSynthesizeData':
        # not used
        # choose the template which has the nearest length of the target mfcc
        index_template = np.argmin(np.abs((np.array(list_N_frames) - N_frame)))
        output = open('syllable_mfcc_templates/dic_mfcc_synthesized_' + str(list_N_frames[index_template]) + '.pkl', 'r')
        dic_mfcc_synthesized = pickle.load(output)
        output.close()

    elif method == 'lyricsRecognizerHMM':

        # build matching network from the score dataset
        phrases, \
        lyrics_net, \
        mat_trans_comb, \
        state_pho_comb, \
        index_start, \
        index_end, \
        list_centroid_pho_dur = makeNet(dict_score_100)

        hmm = ParallelLRHMM(lyrics_net,
                            mat_trans_comb,
                            state_pho_comb,
                            index_start,
                            index_end)
        if am == 'gmm':
            hmm._gmmModel(gmmModels_path)

        # viterbi decoding
        paths_hmm, posteri_probas = hmm._viterbiLog(observations=mfcc_target, am=am, kerasModel=kerasModel)
        # best_match_lyrics = hmm._getBestMatchLyrics(path_hmm)

        # aggregate output for result anaylsis
        list_state_dur_path_centroid_pho_durs = []
        for ii_path in xrange(len(paths_hmm)):
            path_ii = paths_hmm[ii_path]
            state_dur_path = hmm._pathStateDur(path_ii)

            centroid_pho_durs = list_centroid_pho_dur[ii_path]
            centroid_pho_durs = np.array(centroid_pho_durs) / np.sum(centroid_pho_durs)
            centroid_pho_durs *= duration_target

            list_state_dur_path_centroid_pho_durs.append([state_dur_path, centroid_pho_durs.tolist()])

        dict_query_phrase = \
            {'list_state_dur_path_centroid_pho_durs': list_state_dur_path_centroid_pho_durs,
             'query_phrase_name': query_phrase_name,
             'lyrics_net': lyrics_net,
             'posteri_probas': posteri_probas.tolist(),
             'phrases': phrases,
             'line_lyrics': line_lyrics
             }

    elif method == 'lyricsRecognizerHSMM':
        # build matching network from the score dataset
        phrases, \
        lyrics_net, \
        mat_trans_comb, \
        state_pho_comb, \
        index_start, \
        index_end, \
        list_centroid_pho_dur = makeHSMMNet(dict_score_100)

        # calculate the mean duration of each phoneme (state) in the network
        # this mean will be to generate gaussian duration distribution for each state.
        mean_dur_state = []
        for cpd in list_centroid_pho_dur:
            cpd = np.array(cpd) / np.sum(cpd)
            cpd *= duration_target
            mean_dur_state += cpd.tolist()
        # print mean_dur_state

        hsmm = ParallelLRHSMM(lyrics_net,
                              mat_trans_comb,
                              state_pho_comb,
                              index_start,
                              index_end,
                              mean_dur_state,
                              proportionality_std)
        if am == 'gmm':
            hsmm._gmmModel(gmmModels_path)

        # viterbi decoding
        paths_hmm, posteri_probas = hsmm._viterbiHSMM(observations=mfcc_target, am=am, kerasModel=kerasModel)

        # aggregate results for analysis
        dict_query_phrase = \
            {'query_phrase_name': query_phrase_name,
             'lyrics_net': lyrics_net,
             'posteri_probas': posteri_probas.tolist(),
             'phrases': phrases,
             'line_lyrics': line_lyrics}

    return dict_query_phrase, query_phrase_name

def generalProcess(method,
                   proportionality_std=0,
                   path_json_dict_query_phrases='dummy',
                   am='gmm',
                   files=()):

    ##-- method conditions
    if method == 'obsMatrix':
        # not used
        gmmModel = gmmPhoModelLoad()
    else:
        gmmModel = ''
        pass

    if am=='cnn':
        kerasModel = ParallelLRHMM.kerasModel(kerasModels_path)
    else:
        kerasModel = ''

    dict_query_phrases = {}
    # files = [filename for filename in getRecordingNamesSimi('TEST',class_name)]

    for filename in files:
        nestedPhonemeLists, _, _ = syllableTextgridExtraction(textgridDataDir, filename, 'line', 'details')
        sampleRate, wavData = wavfile.read(path.join(wavDataDir,filename+'.wav'))
        for i, line_list in enumerate(nestedPhonemeLists):
            print filename, i

            # these phrases are not in score dataset
            if (filename == 'lseh-Zi_na_ri-Hong_yang_dong-qm' and i in [4,5]) or \
                (filename == 'lsxp-Huai_nan_wang-Huai_he_ying02-qm' and i in [0,1,2,3]):
                continue

            if filename == 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm' and i in [3]:
                continue

            dict_query_phrase, query_phrase_name = phraseProcess(filename,
                                                      i,
                                                      line_list,
                                                      wavData,
                                                      gmmModel,
                                                      kerasModel,
                                                      sampleRate,
                                                      method,
                                                      proportionality_std)

            dict_query_phrases[query_phrase_name] = dict_query_phrase

    with open(path_json_dict_query_phrases,'wb') as outfile:
        json.dump(dict_query_phrases,outfile)


    '''
    # below is reserved for experiment of obsMatrix and candidateSynthesizeData methods
            list_simi = []
            # best_dist = float('Inf')
            # best_M    = np.array([])
            # best_scoreM = np.array([])
            for key in dict_score:
                print key

                if method == 'obsMatrix':
                    # scoreM = scoreMSynthesize(dict_score[key],N_frame)    # syllable gmm
                    scoreM = scoreMSynthesizePho(dict_score[key],N_frame)   # pho gmm
                    # scoreM = scoreM[1:,:]
                    # scoreMaskedM = obsM*scoreM
                    # udist = abs(np.sum(scoreMaskedM))

                    # print scoreM
                    obsMPlotPho(obsM)
                    obsMPlotPho(scoreM)
                    udist,plen = dtwNd(scoreM.transpose(),obsM.transpose())
                    print udist,plen
                    udist = udist/plen
                else:
                    if method == 'candidateSynthesizeData':
                        # mfcc synthesized from data, dim_mfcc = 12
                        mfcc_synthesized = dic_mfcc_synthesized[key]
                    else:
                        # mfcc synthesized from GMM, dim_mfcc = 36
                        mfcc_synthesized = mfccSynthesizeFromGMM(dict_score[key],mfcc_target.shape[1],N_frame)

                    udist,_ = fastdtw(mfcc_target,mfcc_synthesized,dist=euclidean)    # not good result
                    # udist = dtwNd(mfcc_target,mfcc_synthesized)     # bad result
                    print udist
                    # plotMFCC(mfcc_synthesized)
                    # plotMFCC(mfcc_target)

                # print udist
                lyrics = dict_score[key]['lyrics']
                sdist = stringDist(line_lyrics,lyrics)
                list_simi.append([key,lyrics,udist,sdist])

                # if udist < best_dist:
                #     best_dist=udist
                #     best_M = scoreMaskedM
                #     best_scoreM = scoreM
                # list_simi.append([key,lyrics,sdist])

            list_simi = sorted(list_simi,key=itemgetter(2))
            list_sdist = [ls[3] for ls in list_simi]
            # list_sdist = [ls[2] for ls in list_simi]
            order = list_sdist.index(max(list_sdist))
            #
            # obsMPlot(obsM)
            # obsMPlot(best_M)
            # print best_dist
            # print list_simi[order][2]

            with open('results/obsMatrixPho_cosineDist/'+filename+str(i)+'.csv','wb') as csvfile:
                w = csv.writer(csvfile)
                w.writerow([line_lyrics,str(order),list_simi[order][1]])
                for row_simi in list_simi:
                    w.writerow(row_simi)

            order += 1
            list_rank.append(order)


    print list_rank

    mrr = MRR(list_rank)
    top1hit = topXhit(1,list_rank)
    top3hit = topXhit(3,list_rank)
    top5hit = topXhit(5,list_rank)
    top10hit = topXhit(10,list_rank)
    top20hit = topXhit(20,list_rank)

    with open('eval/'+class_name+'_'+dm+'_'+method+'_'+'mfcc_noscaling_pho'+'.csv','wb') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['MRR',mrr])
        w.writerow(['top 1 hit',top1hit])
        w.writerow(['top 3 hit',top3hit])
        w.writerow(['top 5 hit',top5hit])
        w.writerow(['top 10 hit',top10hit])
        w.writerow(['top 20 hit',top20hit])
    '''

    #  # run analysis and save
    # for cpp in xrange(1,31,1):
    #     runResultsAnalysis(cpp)

