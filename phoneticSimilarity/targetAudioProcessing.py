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

# this file is reserved for other usage

import essentia.standard as ess
from essentia.streaming import *
from general.parameters import *
from general.filePath import *
from sklearn import preprocessing
from acousticModelTraining import getFeature,getMFCCBands1D,getMFCCBands2D,featureReshape
import numpy as np
import pickle
from os import path
from general.pinyinMap import *
from general.phonemeMap import *
import matplotlib.pyplot as plt


def pitchProcessing_audio(filename_wav):
    loader = EqloudLoader(filename=filename_wav)
    fc = FrameCutter(frameSize=framesize_phoneticSimilarity,
                     hopSize=hopsize_phoneticSimilarity)
    loader.audio >> fc.signal

    # windowing:
    w = Windowing(type='blackmanharris62')
    fc.frame >> w.frame

    # spectrum:
    spec = Spectrum()
    w.frame >> spec.frame

    # pitch yin FFT
    pitch = PitchYinFFT(frameSize=framesize_phoneticSimilarity, sampleRate = loader.paramValue('sampleRate'))
    spec.spectrum >> pitch.spectrum

    p = essentia.Pool()

    pitch.pitch >> (p,'pitch')
    pitch.pitchConfidence >> (p,'pitchConfidence')

    essentia.run(loader)

    # discard pitch below 65, higher than 1000 Hz, confidence below 0.85
    index_keep = discardPitch(p['pitch'],p['pitchConfidence'],65,1000,0.85)

    return index_keep

def discardPitch(pitch,pitchConfidence,low_threshold_pitch,high_threshold_pitch,threshold_confidence):
    '''
    keep the pitch if confidence > threshold and pitch > low_threshold and pitch < high_threshold
    '''
    index_keep = []
    for ii in range(len(pitch)):
        if not (pitchConfidence[ii] > threshold_confidence
                and pitch[ii] > low_threshold_pitch
                and pitch[ii] < high_threshold_pitch):
            index_keep.append(ii)

    return index_keep

def mfccFeature_audio(filename_wav,index_keep,feature_type='mfcc'):
    audio               = ess.MonoLoader(downmix = 'left', filename = filename_wav, sampleRate = fs)()
    if feature_type == 'mfcc':
        feature             = getFeature(audio)
    elif feature_type == 'mfccBands1D':
        feature             = getMFCCBands1D(audio)
    elif feature_type == 'mfccBands2D':
        feature             = getMFCCBands2D(audio,nbf=True)

    if feature_type == 'mfccBands1D' or feature_type == 'mfccBands2D':
        feature             = np.log(100000 * feature + 1)
        scaler = pickle.load(open(kerasScaler_path,'rb'))
        feature = scaler.transform(feature)

    # feature             = preprocessing.StandardScaler().fit_transform(feature)
    # index_keep          = pitchProcessing_audio(filename_wav)
    feature_out         = feature[index_keep[0],:]

    for index in index_keep[1:]:
        feature_out = np.vstack((feature_out,feature[index,:]))

    if feature_type == 'mfccBands2D':
        feature_out = featureReshape(feature_out)

    return feature_out

def gmmModelLoad():
    gmmModel = {}
    for final in finals:
        gmmModel_path = path.join(gmmModels_path,final+'.pkl')
        try:
            pkl_file = open(gmmModel_path)
            gmmModel[final] = pickle.load(pkl_file)
            pkl_file.close()
        except:
            print(final+' not found.')
    return gmmModel

def gmmPhoModelLoad():
    gmmModel = {}
    for pho in dic_pho_map.values():
        gmmModel_path = path.join(gmmModels_path,pho+'.pkl')
        try:
            pkl_file = open(gmmModel_path)
            gmmModel[pho] = pickle.load(pkl_file)
            pkl_file.close()
        except:
            print(pho+' not found.')
    return gmmModel

def obsMatrix(feature,gmmModel):
    dim_t       = feature.shape[0]
    print dim_t
    dim_final   = len(finals)
    obsM        = np.zeros((dim_final, dim_t))
    # obsM[:]     = -float('Inf')
    for ii,final in enumerate(finals):
        if final in gmmModel:
            obsM[ii,:] = gmmModel[final].score_samples(feature)
    return obsM

def obsMatrixPho(feature,gmmModel):
    dim_t       = feature.shape[0]
    print dim_t
    dim_pho   = len(dic_pho_map.values())
    obsM        = np.ones((dim_pho, dim_t))*(-float('Inf'))
    # obsM[:]     = -float('Inf')
    for ii,pho in enumerate(dic_pho_map.values()):
        if pho in gmmModel:
            obsM[ii,:] = gmmModel[pho].score_samples(feature)
    return obsM

def processFeature(filename_wav,feature_type='mfcc'):
    index_keep = pitchProcessing_audio(filename_wav)
    feature = mfccFeature_audio(filename_wav,index_keep,feature_type)
    return feature

def obsMPlot(obsM):
    '''
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # print obsM.shape
    y = np.arange(obsM.shape[0]+1)
    x = np.arange(obsM.shape[1]) * hopsize_phoneticSimilarity / float(fs)
    ax.pcolormesh(x,y,obsM)
    #set ticks
    T=np.arange(len(finals))+0.5
    ax.set_yticks(T)
    ax.set_yticklabels(finals)

    plt.show()

def obsMPlotPho(obsM):
    '''
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # print obsM.shape
    y = np.arange(obsM.shape[0]+1)
    x = np.arange(obsM.shape[1]) * hopsize_phoneticSimilarity / float(fs)
    ax.pcolormesh(x,y,obsM)
    #set ticks
    T=np.arange(len(dic_pho_map.values()))+0.5
    ax.set_yticks(T)
    ax.set_yticklabels(dic_pho_map.values())

    plt.show()

