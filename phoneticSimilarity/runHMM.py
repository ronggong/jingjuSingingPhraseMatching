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

"""
run this .py to do experiment for HMM matching or post-processor duration modelling

the evaluation results can be found in eval folder
HMM result file name ends with 1.0_0.0.csv
post result file name ends with either 1.0_0.7.csv for dan role-type and 1.0_1.5.csv for laosheng
"""

from os import path
import sys
import numpy as np
from generalProcess import generalProcess
from general.trainTestSeparation import getRecordingNamesSimi

currentPath = path.dirname(path.abspath(__file__))
lyricsRecognizerPath = path.join(currentPath, 'lyricsRecognizer')
sys.path.append(lyricsRecognizerPath)

from resultAnalysis import resultAnalysisProcess
from general.filePath import cnn_file_name,class_name
from general.parameters import am

################################
##### lyricsRecognizer HMM #####
################################
method = 'lyricsRecognizerHMM'

# query output json, calculated in generalProcess, evaluated in resultAnalysisProcess
path_json_dict_query_phrases = path.join(currentPath, 'results', 'dict_query_phrases_' \
                                       + method + '_' \
                                       + class_name + '_'\
                                       + am+cnn_file_name + '.json')

files = [filename for filename in getRecordingNamesSimi('TEST', class_name)]

generalProcess(method=method,
               path_json_dict_query_phrases=path_json_dict_query_phrases,
               am=am,
               files=files)

# the parameter sweeping for HMM is inside the resultAnalysisProcess function
#
resultAnalysisProcess(method=method,
                      path_json_dict_query_phrases=path_json_dict_query_phrases,
                      am=am,
                      cnn_file_name=cnn_file_name)