#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import os
import pickle,cPickle,gzip

import numpy as np
from sklearn import mixture
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import essentia.standard as ess

from general.textgridParser import syllableTextgridExtraction
from general.trainTestSeparation import getRecordingNamesSimi
from general.parameters import *
from general.Fdeltas import Fdeltas
from general.Fprev_sub import Fprev_sub
from general.phonemeMap import *
from general.filePath import *


from general.pinyinMap import *
# from src.trainTestSeparation import getRecordings, getRecordingNumber

##-- number of Mel bands used in MFCC function
if am == 'gmm':
    numberBands = 40
elif am == 'cnn':
    numberBands = 80
else:
    raise('am acoustic model param is not exist.')

winAnalysis     = 'hann'
N               = 2 * framesize_phoneticSimilarity                     # padding 1 time framesize
SPECTRUM        = ess.Spectrum(size=N)
MFCC            = ess.MFCC(sampleRate           =fs,
                           highFrequencyBound   =highFrequencyBound,
                           inputSize            =framesize_phoneticSimilarity + 1,
                           numberBands          =numberBands)
WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize_phoneticSimilarity)

def getFeature(audio):

    '''
    MFCC of give audio interval [p[0],p[1]]
    :param audio:
    :param p:
    :return:
    '''

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize_phoneticSimilarity, hopSize=hopsize_phoneticSimilarity):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC(mXFrame)
        mfccFrame       = mfccFrame[1:]
        mfcc.append(mfccFrame)

    mfcc            = np.array(mfcc).transpose()
    dmfcc           = Fdeltas(mfcc,w=5)
    ddmfcc          = Fdeltas(dmfcc,w=5)
    feature         = np.transpose(np.vstack((mfcc,dmfcc,ddmfcc)))


    return feature

def getMFCCBands1D(audio):

    '''
    MFCC bands feature [p[0],p[1]], this function only for DNN acoustic model training
    it needs the array format float32
    :param audio:
    :param p:
    :return:
    '''

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize_phoneticSimilarity, hopSize=hopsize_phoneticSimilarity):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC(mXFrame)
        mfcc.append(bands)

    # the mel bands features
    feature = np.array(mfcc,dtype='float32')

    return feature

def getMFCCBands2D(audio, nbf=False, nlen=10):

    '''
    mel bands feature [p[0],p[1]]
    output feature for each time stamp is a 2D matrix
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need to neighbor frames
    :return:
    '''

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize_phoneticSimilarity, hopSize=hopsize_phoneticSimilarity):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC(mXFrame)
        mfcc.append(bands)

    if nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_out = np.array(mfcc, copy=True)
        for ii in range(1,nlen+1):
            mfcc_right_shift    = Fprev_sub(mfcc, w=ii)
            mfcc_left_shift     = Fprev_sub(mfcc, w=-ii)
            mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
        feature = mfcc_out.transpose()
    else:
        feature = mfcc
    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature

def getMBE(audio):
    '''
    mel band energy feature, for some other usage
    :param audio:
    :return:
    '''

    mfccBands = []
    for frame in ess.FrameGenerator(audio, frameSize=framesize_phoneticSimilarity, hopSize=hopsize_phoneticSimilarity):

        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC(mXFrame)
        mfccBands.append(bands)
    feature         = np.array(mfccBands)
    return feature

def featureReshape(feature):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = 21

    feature_reshaped = np.zeros((n_sample,n_row,n_col),dtype='float32')

    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row,n_col),dtype='float32')
        for jj in range(n_col):
            feature_frame[:,jj] = feature[ii][n_row*jj:n_row*(jj+1)]
        feature_reshaped[ii,:,:] = feature_frame
    return feature_reshaped


def dumpFeaturePho(class_name,recordings,syllableTierName,phonemeTierName,feature_type='mfcc'):
    '''
    dump the MFCC for each phoneme
    :param recordings:
    :return:
    '''
    if class_name == 'danAll':
        textgrid_path = textgrid_path_dan
        wav_path    = wav_path_dan
    elif class_name == 'laosheng':
        textgrid_path = textgrid_path_laosheng
        wav_path = wav_path_laosheng

    ##-- dictionary feature
    dic_pho_feature = {}

    for _,pho in enumerate(set(dic_pho_map.values())):
        dic_pho_feature[pho] = np.array([])

    for recording in recordings:
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)

        # audio
        wav_full_filename   = os.path.join(wav_path,recording+'.wav')
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_full_filename, sampleRate = fs)()

        if feature_type == 'mfcc':
            # MFCC feature
            mfcc = getFeature(audio)
        elif feature_type == 'mfccbands':
            # MFCC energy bands feature
            mfcc = getMFCCBands1D(audio)
        else:
            raise('feature type not exists, either mfcc or mfccbands.')

        for ii,pho in enumerate(nestedPhonemeLists):
            print 'calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists))
            for p in pho[1]:
                # map from annotated xsampa to readable notation
                key = dic_pho_map[p[2]]

                sf = round(p[0] * fs / float(hopsize_phoneticSimilarity)) # starting frame
                ef = round(p[1] * fs / float(hopsize_phoneticSimilarity)) # ending frame

                mfcc_p = mfcc[sf:ef,:]  # phoneme syllable

                if not len(dic_pho_feature[key]):
                    dic_pho_feature[key] = mfcc_p
                else:
                    dic_pho_feature[key] = np.vstack((dic_pho_feature[key],mfcc_p))

    return dic_pho_feature

def trainValidationSplit(dic_pho_feature_train,validation_size=0.2):
    '''
    split the feature in dic_pho_feature_train into train and validation set
    :param dic_pho_feature_train: input dictionary, key: phoneme, value: feature vectors
    :return:
    '''
    feature_all = []
    label_all = []
    for key in dic_pho_feature_train:
        feature = dic_pho_feature_train[key]
        label = [dic_pho_label[key]] * len(feature)

        if len(feature):
            if not len(feature_all):
                feature_all = feature
            else:
                feature_all = np.vstack((feature_all, feature))
            label_all += label
    label_all = np.array(label_all,dtype='int64')

    feature_all = preprocessing.StandardScaler().fit_transform(feature_all)
    feature_train, feature_validation, label_train, label_validation = \
        train_test_split(feature_all, label_all, test_size=validation_size, stratify=label_all)

    return feature_train, feature_validation, label_train, label_validation

def bicGMMModelSelection(X):
    '''
    bic model selection
    :param X: features - observation * dimension
    :return:
    '''
    lowest_bic = np.infty
    bic = []
    n_components_range  = [10,15,20,25,30,35,40,45,50,55,60,65,70]
    best_n_components   = n_components_range[0]
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        print 'Fitting GMM with n_components =',str(n_components)
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='diag')
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_n_components = n_components
            best_gmm          = gmm

    return best_n_components,gmm

def modelSelection(featureFilename):
    '''
    print the best n_component for the phoneme in feature file
    :param featureFilename:
    :return:
    '''
    # model loading
    pkl_file = open(featureFilename, 'rb')
    dic_pho_feature_train = pickle.load(pkl_file)
    pkl_file.close()

    print len(dic_pho_feature_train.keys())
    for ii,key in enumerate(dic_pho_feature_train):
        X = dic_pho_feature_train[key]
        best_n_components = bicGMMModelSelection(X)

        print 'The best n_components for',key,'is',str(best_n_components)


def processAcousticModelTrainPho(class_name,syllableTierName,phonemeTierName,featureFilename,gmmModel_path):
    '''
    Monophonic acoustic model training
    :param mode: sourceSeparation, qmLonUpfLaosheng
    :param syllableTierName: 'pinyin', 'dian'
    :param phonemeTierName: 'details'
    :param featureFilename: 'dic_pho_feature_train.pkl'
    :param gmmModel_path: in parameters.py
    :return:
    '''

    # model training
    recordings_train = getRecordingNamesSimi('TRAIN',class_name)

    dic_pho_feature_train = dumpFeaturePho(class_name,recordings_train,syllableTierName,phonemeTierName)

    output = open(featureFilename, 'wb')
    pickle.dump(dic_pho_feature_train, output)
    output.close()

    # model loading
    pkl_file = open(featureFilename, 'rb')
    dic_pho_feature_train = pickle.load(pkl_file)
    pkl_file.close()

    n_comp = 40
    g = mixture.GaussianMixture(n_components=n_comp,covariance_type='diag')

    print len(dic_pho_feature_train.keys())
    for ii,key in enumerate(dic_pho_feature_train):
        # print key, dic_pho_feature_train[key].shape

        print 'fitting gmm ', key, ' ', str(ii), ' of ', str(len(dic_pho_feature_train.keys()))

        ##-- try just fit the first dim of MFCC
        # x = np.expand_dims(dic_pho_feature_train[key][:,0],axis=1)

        x = dic_pho_feature_train[key]
        print x.shape

        if x.shape[0] > n_comp*5:
            g.fit(x)

            output = open(os.path.join(gmmModel_path,key+'.pkl'),'wb')
            pickle.dump(g, output)
            output.close()
        else:
            # yn not fitted, because too few samples
            print(key+' is not fitted.')

if __name__ == '__main__':

    rp = os.path.dirname(__file__)

    for cn in ['danAll', 'laosheng']:
        if cn == 'danAll':
            gmmModels_path = os.path.join(rp, 'gmmModels/dan')
        elif cn == 'laosheng':
            gmmModels_path = os.path.join(rp, 'gmmModels/laosheng')

        processAcousticModelTrainPho(class_name=cn,
                                      syllableTierName='dian',
                                      phonemeTierName='details',
                                      featureFilename='./gmmModels/dic_pho_feature_train_'+cn+'.pkl',
                                      gmmModel_path=gmmModels_path)

    """
    # dump feature for DNN training, with getFeature output MFCC bands
    # not used
    recordings_train = getRecordingNamesSimi('TRAIN', 'laosheng')

    dic_pho_feature_train_laosheng = dumpFeaturePho('laosheng', recordings_train, syllableTierName, phonemeTierName, feature_type='mfccbands')

    feature_train, feature_validation, label_train, label_validation = trainValidationSplit(dic_pho_feature_train_laosheng, validation_size=0.2)

    cPickle.dump((feature_train,label_train),gzip.open('train_set_laosheng_phraseMatching.pickle.gz', 'wb'),cPickle.HIGHEST_PROTOCOL)
    cPickle.dump((feature_validation, label_validation), gzip.open('validation_set_laosheng_phraseMatching.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)

    print feature_train.shape,len(feature_validation),len(label_train),len(label_validation)


    # dump feature danAll
    recordings_train = getRecordingNamesSimi('TRAIN', 'danAll')

    dic_pho_feature_train_danAll = dumpFeaturePho('danAll', recordings_train, syllableTierName, phonemeTierName, feature_type='mfccbands')

    feature_train, feature_validation, label_train, label_validation = trainValidationSplit(
        dic_pho_feature_train_danAll, validation_size=0.2)

    with gzip.open('train_set_danAll_phraseMatching.pkl.gz', 'wb') as f:
        cPickle.dump((feature_train, label_train), f)

    with gzip.open('validation_set_danAll_phraseMatching.pkl.gz', 'wb') as f:
        cPickle.dump((feature_validation, label_validation), f)
    """