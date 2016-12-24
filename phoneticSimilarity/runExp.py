from os import path
import sys

currentPath = path.dirname(__file__)
parentPath = path.join(currentPath, 'lyricsRecognizer')
sys.path.append(parentPath)

from general.trainTestSeparation import getRecordingNamesSimi
from general.textgridParser import syllableTextgridExtraction
from general.filePath import *
from general.eval import *
from general.parameters import list_N_frames
from general.dtwSankalp import dtwNd
from targetAudioProcessing import gmmModelLoad,gmmPhoModelLoad,processMFCC,obsMPlot,obsMPlotPho,obsMatrix,obsMatrixPho
from scoreManip import scoreMSynthesize,scoreMSynthesizePho,mfccSynthesizeFromData,mfccSynthesizeFromGMM,plotMFCC
from tailsMFCCTrain import loadMFCCTrain
from ParallelLRHMM import ParallelLRHMM
from makeNet import makeNet
from scipy.io import wavfile
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from operator import itemgetter
import json
import pickle
import unicodecsv as csv
import numpy as np


with open('../melodicSimilarity/scores.json','r') as f:
    dict_score = json.load(f)

class_name = 'laosheng'
dist_measures = ['euclideanDist','sankalpNdDTW121']
dm = dist_measures[1]

methods = ['candidateSynthesizeData','candidateSynthesizeGMM','obsMatrix','lyricsRecognizer']
method = methods[3]

if class_name == 'dan':
    textgridDataDir = textgrid_path_dan
    wavDataDir = wav_path_dan
elif class_name == 'laosheng':
    textgridDataDir = textgrid_path_laosheng
    wavDataDir = wav_path_laosheng

if method == 'obsMatrix':
    # gmmModel = gmmModelLoad()
    gmmModel = gmmPhoModelLoad()
elif method == 'lyricsRecognizer':
    phrases,lyrics_net,mat_trans_comb,state_pho_comb,index_start,index_end = makeNet(dict_score)
    # print np.where(mat_trans_comb==1.0)
    # print index_start
else:
    pass
    # dic_syllable_feature_train = loadMFCCTrain('dic_syllable_feature_train_'+class_name+'.pkl')

list_rank = []
files = [filename for filename in getRecordingNamesSimi('TEST',class_name)]
for filename in files:
    nestedPhonemeLists, _, _ = syllableTextgridExtraction(textgridDataDir, filename, 'line', 'details')
    sampleRate, wavData = wavfile.read(path.join(wavDataDir,filename+'.wav'))
    print filename
    for i, line_list in enumerate(nestedPhonemeLists):

    # these phrases are not in score corpus
        if (filename == 'lseh-Zi_na_ri-Hong_yang_dong-qm' and i in [4,5]) or \
            (filename == 'lsxp-Huai_nan_wang-Huai_he_ying02-qm' and i in [0,1,2,3]):
            continue

        line = line_list[0]
        start_frame = int(round(line[0]*sampleRate))
        end_frame = int(round(line[1]*sampleRate))
        line_lyrics = line[2]

        wav_line = wavData[start_frame:end_frame]
        wavfile.write('temp.wav',sampleRate,wav_line)

        mfcc_target = processMFCC('temp.wav')
        N_frame = mfcc_target.shape[0]

        if method == 'obsMatrix':
            # obsM = obsMatrix(mfcc_target,gmmModel)    # syllable gmm
            obsM = obsMatrixPho(mfcc_target,gmmModel)   # pho gmm
            obsM = np.exp(obsM)
            # obsM = obsM[1:,:]
            obsM =  obsM/np.sum(obsM,axis=0)
            print obsM
        elif method == 'candidateSynthesizeData':
            # choose the template which has the nearest length of the target mfcc
            index_template = np.argmin(np.abs((np.array(list_N_frames)-N_frame)))
            output = open('syllable_mfcc_templates/dic_mfcc_synthesized_'+str(list_N_frames[index_template])+'.pkl', 'r')
            dic_mfcc_synthesized = pickle.load(output)
            output.close()
        elif method == 'lyricsRecognizer':
            hmm = ParallelLRHMM(lyrics_net,mat_trans_comb,state_pho_comb,index_start,index_end)
            hmm._gmmModel(gmmModels_path)
            path_hmm,posteri_probas = hmm._viterbiLog(observations=mfcc_target)
            best_match_lyrics = hmm._getBestMatchLyrics(path_hmm)
            print best_match_lyrics
            print path_hmm
            dict_out_lyricsReco = {'query_lyrics':line_lyrics,
                                   'posteri_probas':posteri_probas.tolist(),
                                   'best_match_lyrics':best_match_lyrics,
                                   'lyrics':lyrics_net,
                                   'phrases':phrases,
                                   'path':path_hmm.tolist()}
            with open('results/lyricsRecognizer/'+filename+'_'+str(i)+'.json','w') as outfile:
                json.dump(dict_out_lyricsReco,outfile)
            # hmm._pathPlot([],[],path_hmm)
            continue


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
