#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle,cPickle,gzip

import numpy as np
from sklearn import mixture
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import essentia.standard as ess

from general.textgridParser import syllableTextgridExtraction
# from src.trainTestSeparation import getRecordings, getRecordingNumber
from general.trainTestSeparation import getRecordingNamesSimi
from general.parameters import *
from general.Fdeltas import Fdeltas
from general.pinyinMap import *
from general.phonemeMap import *
from general.filePath import *

winAnalysis     = 'hann'
N               = 2 * framesize_phoneticSimilarity                     # padding 1 time framesize
SPECTRUM        = ess.Spectrum(size=N)
MFCC            = ess.MFCC(sampleRate=fs, highFrequencyBound=highFrequencyBound, inputSize=framesize_phoneticSimilarity + 1)
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

def getMFCCBands(audio):

    '''
    MFCC bands feature [p[0],p[1]], this function only for pdnn acoustic model training
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

def getMBE(audio):
    '''
    mel band energy feature
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

def dumpFeature(class_name,recordings,syllableTierName,phonemeTierName):
    '''
    dump the MFCC for each final
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
    dic_final_feature = {}

    for final in finals:
        dic_final_feature[final] = np.array([])

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
            # map from annotated xsampa to readable notation
            key = dic_pinyin_2_initial_final_map[syl[0][2]]['final']
            print key
            for p in syl[1]:

                if p[2] in ['c','m','l','x','f','k',"r\'",'']:
                    continue

                sf = round(p[0] * fs / float(hopsize_phoneticSimilarity)) # starting frame
                ef = round(p[1] * fs / float(hopsize_phoneticSimilarity)) # ending frame

                mfcc_p = mfcc[sf:ef,:]  # phoneme syllable

                if not len(dic_final_feature[key]):
                    dic_final_feature[key] = mfcc_p
                else:
                    dic_final_feature[key] = np.vstack((dic_final_feature[key],mfcc_p))

    return dic_final_feature

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
            mfcc = getMFCCBands(audio)
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

def processAcousticModelTrain(class_name,syllableTierName,phonemeTierName,featureFilename,gmmModel_path):
    '''

    :param mode: sourceSeparation, qmLonUpfLaosheng
    :param syllableTierName: 'pinyin', 'dian'
    :param phonemeTierName: 'details'
    :param featureFilename: 'dic_pho_feature_train.pkl'
    :param gmmModel_path: in parameters.py
    :return:
    '''

    # model training
    recordings_train = getRecordingNamesSimi('TRAIN',class_name)

    dic_final_feature_train = dumpFeature(class_name,recordings_train,syllableTierName,phonemeTierName)

    output = open(featureFilename, 'wb')
    pickle.dump(dic_final_feature_train, output)
    output.close()

    # model loading
    pkl_file = open(featureFilename, 'rb')
    dic_final_feature_train = pickle.load(pkl_file)
    pkl_file.close()

    g = mixture.GaussianMixture(n_components=5,covariance_type='diag')

    print len(dic_final_feature_train.keys())
    for ii,key in enumerate(dic_final_feature_train):
        # print key, dic_pho_feature_train[key].shape

        print 'fitting gmm ', key, ' ', str(ii), ' of ', str(len(dic_final_feature_train.keys()))

        ##-- try just fit the first dim of MFCC
        # x = np.expand_dims(dic_pho_feature_train[key][:,0],axis=1)

        x = dic_final_feature_train[key]
        print x.shape
        if x.shape[0] == 0:
            print('not fit this final')
            continue
        g.fit(x)

        output = open(os.path.join(gmmModel_path,key+'.pkl'),'wb')
        pickle.dump(g, output)
        output.close()

def processAcousticModelTrainPho(class_name,syllableTierName,phonemeTierName,featureFilename,gmmModel_path):
    '''

    :param mode: sourceSeparation, qmLonUpfLaosheng
    :param syllableTierName: 'pinyin', 'dian'
    :param phonemeTierName: 'details'
    :param featureFilename: 'dic_pho_feature_train.pkl'
    :param gmmModel_path: in parameters.py
    :return:
    '''
    # recordings      = getRecordings(textgrid_path)
    # number_train    = getRecordingNumber('TRAIN',mode)
    # recordings_train = [recordings[i] for i in number_train]

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

    # processAcousticModelTrain(class_name=class_name,
    #                           syllableTierName=syllableTierName,
    #                           phonemeTierName=phonemeTierName,
    #                           featureFilename='dic_final_feature_train_'+class_name+'.pkl',
    #                           gmmModel_path=gmmModels_path)

    # processAcousticModelTrainPho(class_name=class_name,
    #                           syllableTierName=syllableTierName,
    #                           phonemeTierName=phonemeTierName,
    #                           featureFilename='dic_pho_feature_train_'+class_name+'.pkl',
    #                           gmmModel_path=gmmModels_path)


    # dump feature for DNN training, with getFeature output MFCC bands
    recordings_train = getRecordingNamesSimi('TRAIN', 'laosheng')

    dic_pho_feature_train_laosheng = dumpFeaturePho('laosheng', recordings_train, syllableTierName, phonemeTierName, feature_type='mfccbands')

    feature_train, feature_validation, label_train, label_validation = trainValidationSplit(dic_pho_feature_train_laosheng, validation_size=0.2)

    cPickle.dump((feature_train,label_train),gzip.open('train_set_laosheng_phraseMatching.pickle.gz', 'wb'),cPickle.HIGHEST_PROTOCOL)
    cPickle.dump((feature_validation, label_validation), gzip.open('validation_set_laosheng_phraseMatching.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)

    print len(feature_train),len(feature_validation),len(label_train),len(label_validation)
    print (feature_train,label_train)

    # dump feature danAll
    recordings_train = getRecordingNamesSimi('TRAIN', 'danAll')

    dic_pho_feature_train_danAll = dumpFeaturePho('danAll', recordings_train, syllableTierName, phonemeTierName, feature_type='mfccbands')

    feature_train, feature_validation, label_train, label_validation = trainValidationSplit(
        dic_pho_feature_train_danAll, validation_size=0.2)

    with gzip.open('train_set_danAll_phraseMatching.pkl.gz', 'wb') as f:
        cPickle.dump((feature_train, label_train), f)

    with gzip.open('validation_set_danAll_phraseMatching.pkl.gz', 'wb') as f:
        cPickle.dump((feature_validation, label_validation), f)