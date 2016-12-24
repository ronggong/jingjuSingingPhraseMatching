# -*- coding: utf-8 -*-

'''
the debug printf is written in dtw.c or cdtw.pyx
'''
from os import path,makedirs
from operator import itemgetter
import json

from scipy.io import wavfile
import unicodecsv as csv
import numpy as np

from general.trainTestSeparation import getRecordingNamesSimi
from general.textgridParser import syllableTextgridExtraction
from general.filePath import *
from general.dtwSankalp import dtw1d_generic,dtw1d_std,plotDTW
from general.utilsFunctions import pitchtrackInterp
from melodicSimi import pitchProcessing_audio,pitchProcessingPyin,euclideanDist,plotTwoPitch,plotPitch
from general.eval import stringDist,MRR,topXhit


with open('scores.json','r') as f:
    dict_score = json.load(f)

class_name = 'laosheng'
dist_measures = ['euclideanDist','dtw111']
dm = dist_measures[1]

if class_name == 'dan':
    textgridDataDir = textgrid_path_dan
    wavDataDir = wav_path_dan
elif class_name == 'laosheng':
    textgridDataDir = textgrid_path_laosheng
    wavDataDir = wav_path_laosheng

def runProcess(thres_high_freq=1000,thres_pitch_confidence=0.85):
    list_rank = {}
    query_pitchtracks = {}
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

            ##-- query pitchtrack
            name_query_pitchtrack = filename+'_'+str(i)
            pitchInCents, _ = pitchProcessing_audio('temp.wav',
                                                    thres_high_freq=thres_high_freq,
                                                    thres_pitch_confidence=thres_pitch_confidence)
            # pitchInCents = pitchProcessingPyin('temp.wav',sampleRate)

            pitchInCents -= np.mean(pitchInCents)
            pitchInCents = pitchtrackInterp(pitchInCents)
            query_pitchtracks[name_query_pitchtrack] = pitchInCents

            ##-- query each score candidate
            list_simi = []
            dtw_path = {}
            # best_udist = float('Inf')
            for key in dict_score:
                lyrics = dict_score[key]['lyrics']
                pitchtrack_cents = dict_score[key]['pitchtrack_cents']

                print filename,i,key
                # if filename == 'lseh-Wei_guo_jia-Hong_yang_dong02-qm' and \
                #     i == 0 and key == 'daeh-ZuoRiGong-TaiZhenWaiChuan.xml_1':
                #     plotTwoPitch(pitchInCents,pitchtrack_cents)

                if dm == 'dtw111':
                    # udist,path_align = dtw1d_generic(pitchInCents,pitchtrack_cents)
                    udist,path_align = dtw1d_std(pitchInCents,pitchtrack_cents)

                    # udist = udist/plen
                    dtw_path[key] = [path_align[0].tolist(),path_align[1].tolist()]

                elif dm == 'euclideanDist':
                    udist = euclideanDist(np.array(pitchInCents),np.array(pitchtrack_cents))
                print filename,'done'
                sdist = stringDist(line_lyrics,lyrics)
                list_simi.append([key,lyrics,udist,sdist])
                # list_simi.append([key,lyrics,sdist])

            # plotTwoPitch(pitchInCents,best_pitchtrack_candiates)
            # plotDTW(best_path,best_cost_arr)

            list_simi = sorted(list_simi,key=itemgetter(2))
            list_sdist = [ls[3] for ls in list_simi]
            # list_sdist = [ls[2] for ls in list_simi]
            order = list_sdist.index(max(list_sdist))

            ##-- dump the results and path
            directory_results = 'results/'+str(thres_high_freq)+'_'+str(thres_pitch_confidence)
            if not path.isdir(directory_results):
                makedirs(directory_results)

            with open(join(directory_results,name_query_pitchtrack+'.csv'),'wb') as csvfile:
                w = csv.writer(csvfile)
                w.writerow([line_lyrics,str(order),list_simi[order][1]])
                for row_simi in list_simi:
                    w.writerow(row_simi)

            directory_resultsPath = 'resultsPath/'+str(thres_high_freq)+'_'+str(thres_pitch_confidence)
            if not path.isdir(directory_resultsPath):
                makedirs(directory_resultsPath)

            with open(join(directory_resultsPath,'path_align_'+name_query_pitchtrack+'.json'),'w') as outfile:
                json.dump(dtw_path,outfile)

            order += 1

            # the second element of list rank row is the distance
            # between the query pitch track and the ground truth pitch track
            list_rank[name_query_pitchtrack] = [order,list_simi[order-1][2]]

    with open('query_pitchtracks_'
                      +str(thres_high_freq)+'_'
                      +str(thres_pitch_confidence)
                      +'.json','w') as outfile:
        json.dump(query_pitchtracks,outfile)

    with open('list_rank_'
                      +str(thres_high_freq)+'_'
                      +str(thres_pitch_confidence)
                      +'.json','w') as outfile:
        json.dump(list_rank,outfile)

    print list_rank

    list_rank = [lr[0] for lr in list_rank.values()]
    mrr = MRR(list_rank)
    top1hit = topXhit(1,list_rank)
    top3hit = topXhit(3,list_rank)
    top5hit = topXhit(5,list_rank)
    top10hit = topXhit(10,list_rank)
    top20hit = topXhit(20,list_rank)
    top100hit = topXhit(100,list_rank)

    with open('eval/'+class_name+'_'
                      +dm+'_distNoNormalize_'
                      +str(thres_high_freq)+'_'
                      +str(thres_pitch_confidence)
                      +'.csv','wb') as csvfile:

        w = csv.writer(csvfile)

        w.writerow(['MRR',mrr])
        w.writerow(['top 1 hit',top1hit])
        w.writerow(['top 3 hit',top3hit])
        w.writerow(['top 5 hit',top5hit])
        w.writerow(['top 10 hit',top10hit])
        w.writerow(['top 20 hit',top20hit])
        w.writerow(['top 100 hit',top100hit])

# best parameter is 900, 0.7
for params in [[900,0.7]]:
    runProcess(thres_high_freq=params[0],thres_pitch_confidence=params[1])
