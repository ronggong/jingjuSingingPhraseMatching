from os import path
import sys
from generalProcess import generalProcess

currentPath = path.dirname(__file__)
lyricsRecognizerPath = path.join(currentPath, 'lyricsRecognizer')
sys.path.append(lyricsRecognizerPath)

from resultAnalysis import resultAnalysisProcess

methods = ['candidateSynthesizeData',
           'candidateSynthesizeGMM',
           'obsMatrix',
           'lyricsRecognizerHMM',
           'lyricsRecognizerHSMM']

# # lyricsRecognizer HMM
# method = methods[3]
#
# generalProcess(method,0,am='dnn')
#
# resultAnalysisProcess(method,0,am='dnn')

# lyricsRecognizer HSMM
method = methods[4]

for proportionality_std in [0.1,0.2,0.3,0.5,1.0,2.0,3.0,5.0]: # for the experiment of HSMM
    generalProcess(method,proportionality_std)
    resultAnalysisProcess(method,proportionality_std)
