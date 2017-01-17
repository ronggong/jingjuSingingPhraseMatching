from os import path
import sys
from generalProcess import generalProcess

currentPath = path.dirname(__file__)
lyricsRecognizerPath = path.join(currentPath, 'lyricsRecognizer')
sys.path.append(lyricsRecognizerPath)

from resultAnalysis import resultAnalysisProcess
from general.filePath import dnn_node
from general.parameters import am

methods = ['candidateSynthesizeData',
           'candidateSynthesizeGMM',
           'obsMatrix',
           'lyricsRecognizerHMM',
           'lyricsRecognizerHSMM']

# # lyricsRecognizer HMM
method = methods[3]

generalProcess(method,0,am=am,dnn_node=dnn_node)

resultAnalysisProcess(method,0,am=am,dnn_node=dnn_node)

# lyricsRecognizer HSMM
# method = methods[4]

# for proportionality_std in [0.1,0.2,0.3,0.5,1.0,2.0,3.0,5.0]: # for the experiment of HSMM
# for proportionality_std in [3.0,5.0]: # for the experiment of HSMM

    # generalProcess(method,proportionality_std,am='gmm')
    # resultAnalysisProcess(method,proportionality_std,am='gmm')