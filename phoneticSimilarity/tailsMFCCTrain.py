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

# this file is reserved for other usages

import os,sys

currentPath = os.path.dirname(__file__)
utilsPath = os.path.join(currentPath, '..','melodicSimilarity')
sys.path.append(utilsPath)

from acousticModelTraining import getFeature
from general.filePath import *
from general.pinyinMap import *
from general.parameters import *
from general.textgridParser import syllableTextgridExtraction
from general.trainTestSeparation import getRecordingNamesSimi
import essentia.standard as ess
import numpy as np
from scipy.io import wavfile
import pickle
from melodicSimi import pitchProcessing_audio,discardPitch

def dumpFeature(class_name,recordings,syllableTierName,phonemeTierName):
    '''
    dump the MFCC for each final
    :param recordings:
    :return:
    '''

    if class_name == 'dan':
        textgrid_path = textgrid_path_dan
        wav_path    = wav_path_dan
    elif class_name == 'laosheng':
        textgrid_path = textgrid_path_laosheng
        wav_path = wav_path_laosheng

    ##-- dictionary feature
    dic_final_feature = {}

    for final in finals:
        dic_final_feature[final] = []

    for recording in recordings:
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)

        # audio
        wav_full_filename   = os.path.join(wav_path,recording+'.wav')
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_full_filename, sampleRate = fs)()

        # MFCC feature
        mfcc = getFeature(audio)

        # mfcc = preprocessing.StandardScaler().fit_transform(mfcc)

        for ii,syl in enumerate(nestedPhonemeLists):
            print 'calculating ', recording, ' and syl ', str(ii), ' of ', str(len(nestedPhonemeLists))

            # average pitch
            start_frame = int(round(syl[0][0]*fs))
            end_frame = int(round(syl[0][1]*fs))
            wavfile.write('temp.wav',fs,audio[start_frame:end_frame])

            pitchInCents = pitchProcessing_audio('temp.wav')

            # map from annotated xsampa to readable notation
            key = dic_pinyin_2_initial_final_map[syl[0][2]]['final']
            mfcc_syl = np.array([])
            for p in syl[1]:

                if p[2] in ['c','m','l','x','f','k',"r\'",'']:
                    continue

                sf = round(p[0] * fs / float(hopsize_phoneticSimilarity)) # starting frame
                ef = round(p[1] * fs / float(hopsize_phoneticSimilarity)) # ending frame

                mfcc_p = mfcc[sf:ef,:]  # phoneme syllable

                if not len(mfcc_syl):
                    mfcc_syl = mfcc_p
                else:
                    mfcc_syl = np.vstack((mfcc_syl,mfcc_p))

            dict_syl = {'mfcc':mfcc_syl,'pitch':np.mean(pitchInCents),'N_frame':mfcc_syl.shape[0]}
            dic_final_feature[key].append(dict_syl)
    return dic_final_feature

def processMFCCTrain(featureFilename):
    recordings_train = getRecordingNamesSimi('TRAIN',class_name)
    dic_final_feature_train = dumpFeature(class_name,recordings_train,syllableTierName,phonemeTierName)

    output = open(featureFilename, 'wb')
    pickle.dump(dic_final_feature_train, output)
    output.close()

def loadMFCCTrain(featureFilename):
    # model loading
    pkl_file = open(featureFilename, 'rb')
    dic_syllable_feature_train = pickle.load(pkl_file)
    pkl_file.close()
    return dic_syllable_feature_train

if __name__ == '__main__':
    processMFCCTrain('dic_syllable_feature_train_'+class_name+'.pkl')

    # dic_syllable_feature_train = loadMFCCTrain('dic_syllable_feature_train_'+class_name+'.pkl')
    #
    # N_frames = []
    # for key in dic_syllable_feature_train:
    #     # print key, len(dic_syllable_feature_train[key])
    #     # print dic_syllable_feature_train['ing'][0]['pitch']
    #     N_frame = [syl['N_frame'] for syl in dic_syllable_feature_train[key]]
    #     N_frames += N_frame
    # print max(N_frames),min(N_frames)
