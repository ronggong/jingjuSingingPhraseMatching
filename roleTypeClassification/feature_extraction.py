import sys, os
from subprocess import call
import essentia.standard as ess
from essentia.streaming import *
import numpy
import pandas as pd
from scipy.io import wavfile,savemat

from general.textgridParser import syllableTextgridExtraction
from general.trainTestSeparation import getRecordingNamesSimi
from general.filePath import *
from general.utilsFunctions import hz2cents

#for now we extract all features, we will restrict them later to the relevant ones here
FILE_EXT_WAV = (".wav")
FILE_EXT_TEXTGRID = (".textgrid")

class FeatureExtractor(essentia.streaming.CompositeBase):

    def __init__(self, frameSize=2048,
                       hopSize=1024,
                       sampleRate = 44100.):
        super(FeatureExtractor, self).__init__()

        halfSampleRate = sampleRate/2
        minFrequency = sampleRate/frameSize

        fc = FrameCutter(frameSize=frameSize,
                         hopSize=hopSize)
        # define the features
        # zerocrossingrate
        zcr = ZeroCrossingRate()
        fc.frame >> zcr.signal
        # strong decay
        #sd = StrongDecay(sampleRate = sampleRate)
        #fc.frame >> sd.signal
        # windowing:
        w = Windowing(type='blackmanharris62')
        fc.frame >> w.frame
        # spectrum:
        spec = Spectrum()
        w.frame >> spec.frame
        # spectral energy:
        energy = Energy()
        spec.spectrum >> energy.array
        # spectral rms:
        rms = RMS()
        spec.spectrum >> rms.array
        # spectral centroid:
        square = UnaryOperator(type='square')
        centroid = Centroid(range=halfSampleRate)
        spec.spectrum >> square.array >> centroid.array
        # spectral central moments:
        cm = CentralMoments(range=halfSampleRate)
        ds = DistributionShape()
        spec.spectrum >> cm.array
        cm.centralMoments >> ds.centralMoments
        # mfcc:
        mfcc = MFCC(numberBands = 40, numberCoefficients = 13, sampleRate = sampleRate)
        spec.spectrum >> mfcc.spectrum
        mfcc.bands >> None
        # lpc:
        lpc = LPC(order = 10, sampleRate = sampleRate)
        spec.spectrum >> lpc.frame
        lpc.reflection >> None
        # spectral decrease:
        square = UnaryOperator(type='square')
        decrease = Decrease(range=halfSampleRate)
        spec.spectrum >> square.array >> decrease.array
        # spectral energy ratio:
        ebr_low = EnergyBand(startCutoffFrequency=20,
                             stopCutoffFrequency=150,
                             sampleRate = sampleRate)
        ebr_mid_low = EnergyBand(startCutoffFrequency=150,
                                 stopCutoffFrequency=800,
                                 sampleRate = sampleRate)
        ebr_mid_hi = EnergyBand(startCutoffFrequency=800,
                                stopCutoffFrequency=4000,
                                sampleRate = sampleRate)
        ebr_hi = EnergyBand(startCutoffFrequency=4000,
                            stopCutoffFrequency=20000,
                            sampleRate = sampleRate)

        spec.spectrum >> ebr_low.spectrum
        spec.spectrum >> ebr_mid_low.spectrum
        spec.spectrum >> ebr_mid_hi.spectrum
        spec.spectrum >> ebr_hi.spectrum
        # spectral hfc:
        hfc = HFC(sampleRate = sampleRate)
        spec.spectrum >> hfc.spectrum
        # spectral flux:
        flux = Flux()
        spec.spectrum >> flux.spectrum
        # spectral roll off:
        ro = RollOff(sampleRate = sampleRate)
        spec.spectrum >> ro.spectrum
        # spectral strong peak:
        sp = StrongPeak()
        spec.spectrum >> sp.spectrum
        # bark bands:
        barkBands = BarkBands(numberBands=27, sampleRate = sampleRate)
        spec.spectrum >> barkBands.spectrum
        # spectral crest:
        crest = Crest()
        barkBands.bands >> crest.array
        # spectral flatness db:
        flatness = FlatnessDB()
        barkBands.bands >> flatness.array
        # spectral barkbands central moments:
        cmbb = CentralMoments(range=26) # BarkBands::numberBands - 1
        dsbb = DistributionShape()
        barkBands.bands >> cmbb.array
        cmbb.centralMoments >> dsbb.centralMoments
        # spectral complexity:
        scx = SpectralComplexity(magnitudeThreshold=0.005, sampleRate = sampleRate)
        spec.spectrum >> scx.spectrum
        # pitch detection:
        pitch = PitchYinFFT(frameSize=frameSize, sampleRate = sampleRate)
        spec.spectrum >> pitch.spectrum
        # pitch salience:
        ps = PitchSalience(sampleRate = sampleRate)
        spec.spectrum >> ps.spectrum
        # spectral contrast:
        sc = SpectralContrast(frameSize=frameSize,
                              sampleRate=sampleRate,
                              numberBands=6,
                              lowFrequencyBound=20,
                              highFrequencyBound=11000,
                              neighbourRatio=0.4,
                              staticDistribution=0.15)
        spec.spectrum >> sc.spectrum
        # spectral peaks
        peaks = SpectralPeaks(orderBy='frequency',
                              minFrequency=minFrequency,
                              sampleRate = sampleRate)
        spec.spectrum >> peaks.spectrum
        # dissonance:
        diss = Dissonance()
        peaks.frequencies >> diss.frequencies
        peaks.magnitudes >> diss.magnitudes

        # harmonic peaks:
        harmPeaks = HarmonicPeaks()
        peaks.frequencies >> harmPeaks.frequencies
        peaks.magnitudes >>  harmPeaks.magnitudes
        pitch.pitch >> harmPeaks.pitch


        # tristimulus
        tristimulus = Tristimulus()
        harmPeaks.harmonicFrequencies >> tristimulus.frequencies
        harmPeaks.harmonicMagnitudes  >> tristimulus.magnitudes
        # odd2even
        odd2even = OddToEvenHarmonicEnergyRatio()
        harmPeaks.harmonicFrequencies >> odd2even.frequencies
        harmPeaks.harmonicMagnitudes  >> odd2even.magnitudes
        # inharmonicity
        inharmonicity = Inharmonicity()
        harmPeaks.harmonicFrequencies >> inharmonicity.frequencies
        harmPeaks.harmonicMagnitudes  >> inharmonicity.magnitudes

        # define inputs:
        self.inputs['signal'] = fc.signal

        # define outputs:
        # zerocrossingrate
        self.outputs['zcr'] = zcr.zeroCrossingRate
        # strong decay
        # self.outputs['strong_decay'] = sd.strongDecay
        # spectral energy:
        self.outputs['spectral_energy'] = energy.energy
        # spectral rms:
        self.outputs['spectral_rms'] = rms.rms
        # MFCC rate:
        self.outputs['mfcc'] = mfcc.mfcc
        # LPC coef:
        self.outputs['lpc'] = lpc.lpc
        # spectral centroid
        self.outputs['spectral_centroid'] = centroid.centroid
        # spectral kurtosis
        self.outputs['spectral_kurtosis'] = ds.kurtosis
        # spectral spread
        self.outputs['spectral_spread'] = ds.spread
        # spectral skewness
        self.outputs['spectral_skewness'] = ds.skewness
        # spectral dissonance
        self.outputs['spectral_dissonance'] = diss.dissonance
        # spectral contrast
        self.outputs['sccoeffs'] = sc.spectralContrast
        self.outputs['scvalleys'] = sc.spectralValley
        # spectral decrease:
        self.outputs['spectral_decrease'] = decrease.decrease
        # spectral energy ratio:
        self.outputs['spectral_energyband_low'] = ebr_low.energyBand
        self.outputs['spectral_energyband_middle_low'] = ebr_mid_low.energyBand
        self.outputs['spectral_energyband_middle_high'] = ebr_mid_hi.energyBand
        self.outputs['spectral_energyband_high'] = ebr_hi.energyBand
        # spectral hfc:
        self.outputs['hfc'] = hfc.hfc
        # spectral flux:
        self.outputs['spectral_flux'] = flux.flux
        # spectral roll off:
        self.outputs['spectral_rolloff'] = ro.rollOff
        # spectral strong peak:
        self.outputs['spectral_strongpeak'] = sp.strongPeak
        # bark bands:
        self.outputs['barkbands'] = barkBands.bands
        # spectral crest:
        self.outputs['spectral_crest'] = crest.crest
        # spectral flatness db:
        self.outputs['spectral_flatness_db'] = flatness.flatnessDB
        # spectral barkbands central moments:
        self.outputs['barkbands_kurtosis'] = dsbb.kurtosis
        self.outputs['barkbands_spread'] = dsbb.spread
        self.outputs['barkbands_skewness'] = dsbb.skewness
        #spectral complexity
        self.outputs['spectral_complexity'] = scx.spectralComplexity
        # pitch confidence
        self.outputs['pitch_instantaneous_confidence'] = pitch.pitchConfidence
        self.outputs['pitch_instantaneous_pitch'] = pitch.pitch
        # pitch salience
        self.outputs['pitch_salience'] = ps.pitchSalience

        # inharmonicity
        self.outputs['inharmonicity'] = inharmonicity.inharmonicity
        # odd2even
        self.outputs['oddtoevenharmonicenergyratio'] = odd2even.oddToEvenHarmonicEnergyRatio
        # tristimuli
        self.outputs['tristimulus'] = tristimulus.tristimulus

def getLineList(wavDataDir,lineList,filename):
    '''
    detect silence, return the start, end time of each region
    :param lineList: [start_time_line, end_time_line, syllables_line]
    :return: [[start_time, end_time, 'non_silence'],[start_time, end_time, ''], ...], silence part is an empty string
    '''
    filename_wav = os.path.join(wavDataDir,filename+FILE_EXT_WAV)

    dict_mat = {'filename_wav':filename_wav,'lineList':lineList}
    savemat(file_name = join(root_path,'roleTypeClassification','vad','temp.mat'),mdict=dict_mat)

    os.chdir(join(root_path,'roleTypeClassification','vad'))
    call(['/Applications/MATLAB_R2013a.app/bin/matlab','-nodisplay','-r','runVAD;quit;'])
    os.chdir(join(root_path,'roleTypeClassification'))

    lineList_return = []
    with open('vad/lineList_matlab.lab','r') as f:
        for l in f.readlines():
            array_line = l.split()
            if len(array_line):
                lineList_return.append([float(array_line[0]),
                                        float(array_line[1]),
                                        '' if int(array_line[2]) == 0 else 'non_silence'])

    return lineList_return

def removeSilence(audio,sampleRate,lineList):
    audio_remove_silence = essentia.array([])
    for pho in lineList:
        if len(pho[2]):
            start_time_frame  = int(round(float(pho[0]*sampleRate)))
            end_time_frame    = int(round(float(pho[1]*sampleRate)))
            audio_remove_silence = numpy.hstack((audio_remove_silence,audio[start_time_frame:end_time_frame]))
    return audio_remove_silence

def extract_for_one(wavDataDir, lineList, filename, FILE_EXT_WAV):
    filename_wav                 = os.path.join(wavDataDir,filename+FILE_EXT_WAV)
    filename_wav_silence_removed = os.path.join(wavDataDir+'_silence_removed','temp'+FILE_EXT_WAV)

    ##-- remove the silence from audio
    sr = 44100
    audio = ess.MonoLoader(filename=filename_wav,downmix='left',sampleRate=sr)()
    audio_remove_silence = removeSilence(audio,sr,lineList)
    wavfile.write(filename_wav_silence_removed,sr,audio_remove_silence)

    ##-- process the silence removed audio
    loader = essentia.streaming.EqloudLoader(filename=filename_wav_silence_removed)
    fEx = FeatureExtractor(frameSize=2048, hopSize=1024, sampleRate=loader.paramValue('sampleRate'))
    p = essentia.Pool()

    loader.audio >> fEx.signal

    for desc, output in fEx.outputs.items():
        output >> (p, desc)

    essentia.run(loader)

    # convert pitch from hz to cents
    for i in range(len(p['pitch_instantaneous_pitch'])):
        p['pitch_instantaneous_pitch'][i] = hz2cents(p['pitch_instantaneous_pitch'][i])

    stats = ['mean', 'var', 'dmean', 'dvar']
    statsPool = essentia.standard.PoolAggregator(defaultStats=stats)(p)

    return statsPool

def convert_pool_to_dataframe(essentia_pool, class_name, filename):
    pool_dict = dict()
    for desc in essentia_pool.descriptorNames():
        if type(essentia_pool[desc]) is float:
            pool_dict[desc] = essentia_pool[desc]
        elif type(essentia_pool[desc]) is numpy.ndarray:
            # we have to treat multivariate descriptors differently
            for i, value in enumerate(essentia_pool[desc]):
                feature_name = "{desc_name}{desc_number}.{desc_stat}".format(
                    desc_name=desc.split('.')[0],
                    desc_number=i,
                    desc_stat=desc.split('.')[1])
                pool_dict[feature_name] = value
    pool_dict['class'] = class_name
    return pd.DataFrame(pool_dict, index=[os.path.basename(filename)])

if __name__ == '__main__':

    wavDataDir = sys.argv[1]
    textgridDataDir = sys.argv[2]
    class_name = sys.argv[3]
    outfile = sys.argv[4]
    train_test = sys.argv[5]

    feature_set = pd.DataFrame()

    files = [filename for filename in getRecordingNamesSimi(train_test,class_name)]
    for filename in files:
        nestedPhonemeLists, _, _ = syllableTextgridExtraction(textgridDataDir, filename, 'line', 'details')
        print filename
        for i, line_list in enumerate(nestedPhonemeLists):
            # print line_list[0]

            if train_test == 'TRAIN':
                line_list_current = line_list[1]
            elif train_test == 'TEST':
                line_list_current = getLineList(wavDataDir,line_list[0],filename)

                print line_list_current

            pool = extract_for_one(wavDataDir, line_list_current, filename, FILE_EXT_WAV)
            feature_set = feature_set.append(convert_pool_to_dataframe(pool, class_name, filename+str(i)))

    feature_set.to_csv(outfile)
