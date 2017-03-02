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

from general.pinyinMap import *
from general.phonemeMap import *
from general.parameters import list_N_frames
from general.utilsFunctions import hz2cents
from general.filePath import class_name
from tailsMFCCTrain import loadMFCCTrain
from targetAudioProcessing import gmmModelLoad
import matplotlib.pyplot as plt
import numpy as np
import pinyin
import json
import pickle
from sklearn import preprocessing

def scoreMSynthesize(dict_score_info,N_frame):
    '''
    synthesize score Matrix
    :param syl_durations: a list of the syllable durations
    :return: Matrix
    '''
    syl_finals,syl_durations,_ = retrieveSylInfo(dict_score_info)
    # print syl_finals,syl_durations
    syl_durations = (np.array(syl_durations)/np.sum(syl_durations))*N_frame
    scoreM = np.zeros((len(finals),N_frame))
    counter_frame=0
    for ii in range(len(syl_finals)-1):
        index_final = finals.index(syl_finals[ii])
        scoreM[index_final,counter_frame:counter_frame+syl_durations[ii]] = 1.0
        counter_frame += syl_durations[ii]
    index_final = finals.index(syl_finals[-1])
    scoreM[index_final,counter_frame:] = 1

    return scoreM

def scoreMSynthesizePho(dict_score_info,N_frame):
    '''
    synthesize score Matrix
    :param syl_durations: a list of the syllable durations
    :return: Matrix
    '''
    syl_finals,_,_ = retrieveSylInfo(dict_score_info)
    # print syl_finals,syl_durations

    list_pho = []
    for sf in syl_finals:
        pho_final = dic_final_2_sampa[sf]
        for pho in pho_final:
            pho_map = dic_pho_map[pho]
            if pho_map == u'H':
                pho_map = u'y'
            elif pho_map == u'9':
                pho_map = u'S'
            elif pho_map == u'yn':
                pho_map = u'in'
            list_pho.append(pho_map)

    pho_dict = dic_pho_map.values()
    pho_durations = (np.array([1.0]*len(list_pho))/len(list_pho))*N_frame
    scoreM = np.zeros((len(pho_dict),N_frame))
    counter_frame=0
    for ii in range(len(list_pho)-1):
        index_pho = pho_dict.index(list_pho[ii])
        scoreM[index_pho,counter_frame:counter_frame+pho_durations[ii]] = 1.0
        counter_frame += pho_durations[ii]
    index_final = finals.index(syl_finals[-1])
    scoreM[index_final,counter_frame:] = 1

    return scoreM

def mfccSynthesizeFromGMM(dict_score_info,dim_mfcc,N_frame):
    '''
    sample the mfcc array from GMM model
    :param dic_score_info:
    :param N_frame:
    :return:
    '''
    gmmModels = gmmModelLoad()
    syl_finals,syl_durations,_ = retrieveSylInfo(dict_score_info)
    # print syl_finals,syl_durations
    syl_durations = (np.array(syl_durations)/np.sum(syl_durations))*N_frame
    mfcc_synthesized = np.zeros((N_frame,dim_mfcc))
    counter_frame = 0
    for ii in range(len(syl_finals)-1):
        final_ii = syl_finals[ii]
        dur = int(float(syl_durations[ii]))
        if final_ii == 'v':
            final_ii = 'u'
        elif final_ii == 've':
            final_ii = 'ue'
        X,y = gmmModels[final_ii].sample(n_samples=dur)
        mfcc_synthesized[counter_frame:counter_frame+dur,:] = X
        counter_frame += dur
    X,y = gmmModels[syl_finals[-1]].sample(n_samples=mfcc_synthesized.shape[0]-counter_frame)
    mfcc_synthesized[counter_frame:,:] = X
    return mfcc_synthesized

def mfccSynthesizeFromData(dict_score_info,dic_syllable_feature,N_frame):
    '''
    synthesize mfcc feature matrix for a singing candidate phrase
    :param dict_score_info:
    :param N_frame:
    :return:
    '''
    syl_finals,syl_durations,syl_cents = retrieveSylInfo(dict_score_info)
    syl_durations = (np.array(syl_durations)/np.sum(syl_durations))*N_frame
    mfcc_synthesized = np.array([])
    for ii in range(len(syl_finals)):
        final_ii = syl_finals[ii]
        if final_ii == 'v':
            final_ii = 'u'
        elif final_ii == 've':
            final_ii = 'ue'

        list_final_ii = dic_syllable_feature[final_ii]
        if len(list_final_ii):
            list_dur = np.array([dic_final_element_ii['N_frame'] for dic_final_element_ii in list_final_ii])
            index_final_chosen = np.argmin((np.abs(list_dur-syl_durations[ii])))
            mfcc = list_final_ii[index_final_chosen]['mfcc']
            if not len(mfcc_synthesized):
                mfcc_synthesized = mfcc
            else:
                mfcc_synthesized = np.vstack((mfcc_synthesized,mfcc))
        else:
            print('no candidate final found for final_ii')
    # mfcc_synthesized = preprocessing.StandardScaler().fit_transform(mfcc_synthesized)

    return mfcc_synthesized

def retrieveSylInfo(dict_score_info):
    '''
    collect pinyin and pinyin durations from score
    :param dict_score_info:
    :return:
    '''
    syl_finals = []
    syl_durations =[]
    syl_cents = []
    for dict_note in dict_score_info['notes']:
        lyric = dict_note['lyric']
        dur = dict_note['quarterLength']
        cents = hz2cents(dict_note['freq'])
        # print lyric
        # print dict_note['quarterLength']
        if lyric and dur:
            py = pinyin.get(lyric, format="strip", delimiter=" ")
            py_split = py.split()
            if len(py_split) > 1:
                if py_split[0] in non_pinyin:
                    py = py_split[1]
                else:
                    py = py_split[0]
            # print py
            final = dic_pinyin_2_initial_final_map[py]['final']
            syl_finals.append(final)
            syl_durations.append(dur)
            syl_cents.append(cents)
        elif len(syl_durations):
            syl_durations[-1] += dur
    return syl_finals,syl_durations,syl_cents

def plotMFCC(mfcc):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(mfcc)
    plt.show()

def processSyllableMFCCTemplates(N_frame):
    '''
    generate mfcc synthesized templates according to N_frame
    N_frame: template syllable frame length
    :return:
    '''
    dic_mfcc_synthesized = {}
    for key in dict_score:
        print key
        mfcc_synthesized = mfccSynthesizeFromData(dict_score[key],dic_syllable_feature_train,N_frame)
        dic_mfcc_synthesized[key] = mfcc_synthesized

    output = open('syllable_mfcc_templates/dic_mfcc_synthesized_'+str(N_frame)+'.pkl', 'wb')
    pickle.dump(dic_mfcc_synthesized, output)
    output.close()

if __name__ == '__main__':
    dict_score = json.load(open('../melodicSimilarity/scores.json'))
    dic_syllable_feature_train = loadMFCCTrain('dic_syllable_feature_train_'+class_name+'.pkl')


    for N_frame in list_N_frames:
        processSyllableMFCCTemplates(N_frame)


