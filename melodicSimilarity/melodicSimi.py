import essentia.standard as ess
from essentia.streaming import *
from general.parameters import *
from general.utilsFunctions import hz2cents
import matplotlib.pyplot as plt
import numpy as np
import json
import vamp
from os import path


def pitchProcessing_audio(filename_wav,
                          thres_high_freq=900,
                          thres_pitch_confidence=0.7,
                          movingAve=False):
    '''
    pitch track by yin essentia
    '''
    loader = EqloudLoader(filename=filename_wav)
    fc = FrameCutter(frameSize=framesize_melodicSimilarity,
                     hopSize=hopsize_melodicSimilarity)
    loader.audio >> fc.signal

    # windowing:
    w = Windowing(type='blackmanharris62')
    fc.frame >> w.frame

    # spectrum:
    spec = Spectrum()
    w.frame >> spec.frame

    # pitch yin FFT
    pitch = PitchYinFFT(frameSize=framesize_melodicSimilarity, sampleRate = loader.paramValue('sampleRate'))
    spec.spectrum >> pitch.spectrum

    p = essentia.Pool()

    pitch.pitch >> (p,'pitch')
    pitch.pitchConfidence >> (p,'pitchConfidence')

    essentia.run(loader)

    # discard pitch below 65, higher than 1000 Hz, confidence below 0.85
    pitch,pitchConfidence = discardPitch(p['pitch'],p['pitchConfidence'],65,thres_high_freq,thres_pitch_confidence)

    # convert pitch from hz to cents
    pitchInCents = hz2cents(pitch)

    if movingAve:
        moving = ess.MovingAverage(size=9)
        pitchInCents = moving(pitchInCents)
        pitchInCents = pitchInCents[4:-4]

    # mean normalization
    # pitchInCents = pitchInCents-np.mean(pitchInCents)

    return pitchInCents,pitchConfidence

def discardPitch(pitch,pitchConfidence,low_threshold_pitch,high_threshold_pitch,threshold_confidence):
    '''
    keep the pitch if confidence > threshold and pitch > low_threshold and pitch < high_threshold
    '''
    pitchNew = []
    pitchConfidenceNew = []
    for ii in range(len(pitch)):
        if pitchConfidence[ii] > threshold_confidence \
                and pitch[ii] >low_threshold_pitch \
                and pitch[ii] < high_threshold_pitch:
            pitchNew.append(pitch[ii])
            pitchConfidenceNew.append(pitchConfidence[ii])
    pitchNew = np.array(pitchNew)
    pitchConfidenceNew = np.array(pitchConfidenceNew)
    return pitchNew,pitchConfidenceNew

def pitchProcessingPyin(filename_wav,sr,movingAve=False):
    loader = ess.MonoLoader(filename=filename_wav, downmix = 'left', sampleRate = sr)
    audio = loader()
    data = vamp.collect(audio, sr, "pyin:pyin", output='smoothedpitchtrack')
    pitch = data['vector'][1]
    pitchInCents = hz2cents(pitch)

    if movingAve:
        moving = ess.MovingAverage(size=5)
        pitchInCents = moving(pitchInCents)

    return pitchInCents
    # plotPitch(pitch,[])

def euclideanDist(x,y):
    if len(x) == len(y):
        return np.linalg.norm(x-y)
    else:
        print('Two input array not equal length, euclidean Distance can not be performed.')
        raise ValueError

def plotPitch(pitch,pitchConfidence):
    plt.subplot(2,1,1)
    plt.plot(pitch)
    plt.subplot(2,1,2)
    plt.plot(pitchConfidence)
    plt.show()

def plotTwoPitch(pitch_target,pitch_candidate):
    plt.subplot(211)
    plt.plot(pitch_target)
    plt.subplot(212)
    plt.plot(pitch_candidate)
    plt.show()

def plot3Pitch(pitch_query,
               pitch_groundtruth,
               pitch_match,
               query_phrase_name,
               gt_rank,
               save_fig=False,
               path_fig=None,
               name_fig=None):
    # f, axarr = plt.subplots(3, sharex=True)
    #
    # axarr[0].plot(pitch_query)
    # axarr[0].set_title('query pitch track')
    #
    # axarr[1].plot(pitch_groundtruth)
    # axarr[1].set_title('ground truth pitch track')
    #
    # axarr[2].plot(pitch_match)
    # axarr[2].set_title('best match pitch track')

    plt.figure(figsize=(8,4))
    ax = plt.subplot(111)
    plt.plot(np.array(pitch_query)+700,label='query pitch track')
    plt.plot(np.array(pitch_groundtruth),linestyle = '--',label='ground truth pitch track')
    plt.plot(np.array(pitch_match)-700,linestyle = ':',label='best match pitch track')

    # for plot the figure for ISMIR paper
    # ax.plot(np.array(pitch_query) + 700, linewidth = 3, label='Query phrase F0 contour')
    # ax.plot(np.array(pitch_groundtruth), linestyle='--', linewidth = 3, label='Score 1 F0 contour')
    # ax.plot(np.array(pitch_match) - 700, linestyle=':', linewidth = 3, label='Score 2 F0 contour')

    ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.2),
                    fancybox=True, shadow=True, ncol=2)
    ax.get_yaxis().set_ticks([])
    ax.set_xlim([0,sample_number_total])
    plt.xlabel('Sample')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # plt.title(query_phrase_name+'\n ground truth pitch track ranking: '+str(gt_rank))
    if save_fig:
        plt.savefig(path.join(path_fig,name_fig+'_query_gt_bm.png'),bbox_inches='tight')
    else:
        plt.show()

def plot2PitchAlign(path_resultsDTWAlign,
                    query_pitchtrack,
                    ground_truth_score_pitchtrack,
                    ground_truth_dtw_score,
                    ground_truth_rank,
                    query_phrase_name,
                    ground_truth_phrase_name,
                    save_fig=False,
                    path_fig=None,
                    name_fig=None):

    with open(path.join(path_resultsDTWAlign,'path_align_'+query_phrase_name+'.json')) as openfile:
        dict_dtw_path = json.load(openfile)

    path_align = dict_dtw_path[ground_truth_phrase_name]

    # print path_align[0],path_align[1]
    # print len(query_pitchtrack),len(best_match_score_pitchtrack)
    # print query_phrase_name
    # print best_match_phrase_name

    # try:
    plt.figure()
    plt.plot(np.array(query_pitchtrack),label='query pitch track')
    plt.plot(np.array(ground_truth_score_pitchtrack)-700,label='ground truth pitch track')
    for ii in xrange(0,len(path_align[0]),8):
        x1 = path_align[0][ii]
        x2 = path_align[1][ii]

        y1 = query_pitchtrack[x1]
        y2 = ground_truth_score_pitchtrack[x2]-700
        plt.plot([x1,x2],[y1,y2],linestyle=':',color='r')
    plt.legend(loc='best')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlabel('sample')
    plt.title(name_fig+
              '\n dtw distance: '+ground_truth_dtw_score+
              '\n ground truth pitch track ranking: '+str(ground_truth_rank))
    if save_fig:
        plt.savefig(path.join(path_fig,name_fig+'_dtw_align.png'))
    else:
        plt.show()
    # except:
    #     pass


if __name__ == '__main__':
    with open('results/path/path_align_lseh-Wei_guo_jia-Hong_yang_dong01-lon_0.json','r') as openfile:
        dict_dtw_path = json.load(openfile)
    for key in dict_dtw_path:
        if np.sum(dict_dtw_path[key][0]):
            print dict_dtw_path[key][0],dict_dtw_path[key][0]

# pitchInCents, pitchConfidence = preprocessing_audio('/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/wav/qmLonUpf/laosheng/lseh-Tan_Yang_jia-Hong_yang_dong-qm.wav')
# plotPitch(pitchInCents,pitchConfidence)