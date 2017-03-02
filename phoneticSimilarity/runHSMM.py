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
HSMM decoding of the whole test set is a little bit slow,
so we choose to split the whole dataset into three part in order to run them separately,
We merge the split results into one json file, then evaluate the performance

To do this,

we need to comment mergeResultsDict in __main__,
then run different part 0, 1, 2 in different sessions.

after all calculations are finished, run mergeResultsDict function to get overall analysis

the final evaluation result will be in eval folder
"""

from os import path
import sys
import json
import numpy as np
from generalProcess import generalProcess

currentPath = path.dirname(path.abspath(__file__))
lyricsRecognizerPath = path.join(currentPath, 'lyricsRecognizer')
sys.path.append(lyricsRecognizerPath)

from resultAnalysis import resultAnalysisProcess
from general.filePath import cnn_file_name,class_name
from general.parameters import am
from general.trainTestSeparation import getRecordingNamesSimi

method = 'lyricsRecognizerHSMM'

def mergeResultsDict(proportionality_std=0.1):
    dict_query_phrases_all = {}

    # merge the three part of the dataset into one dictionary
    for ii in range(3):
        # result file name of part ii of test set
        path_json_dict_query_phrases = path.join(currentPath,
                                                 'results',
                                                 'dict_query_phrases_' \
                                                 + method + '_' \
                                                 + class_name + '_' \
                                                 + am + cnn_file_name + '_' \
                                                 + str(proportionality_std) + '_' \
                                                 + 'part' + str(ii) + '.json')
        dict_query_phrases = json.load(open(path_json_dict_query_phrases,'rb'))
        for key in dict_query_phrases:
            dict_query_phrases_all[key] = dict_query_phrases[key]

    # dump the big dictionary into a json
    # result file name of the whole test set
    path_json_dict_query_phrases_all = path.join(currentPath,
                                                 'results',
                                                 'dict_query_phrases_' \
                                                 + method + '_' \
                                                 + class_name + '_' \
                                                 + am + cnn_file_name + '_' \
                                                 + str(proportionality_std) \
                                                 + '.json')
    json.dump(dict_query_phrases_all,open(path_json_dict_query_phrases_all,'wb'))

    # do analysis
    resultAnalysisProcess(method=method,
                          proportionality_std=proportionality_std,
                          path_json_dict_query_phrases=path_json_dict_query_phrases_all,
                          am=am)


def runPart(parts, ii, proportionality_std = 0.1):
    # HSMMs gamma
    # for proportionality_std in np.linspace(0.1, 2.0, 20):
    path_json_dict_query_phrases = path.join(currentPath,
                                             'results',
                                             'dict_query_phrases_' \
                                               + method + '_' \
                                               + class_name + '_' \
                                               + am + cnn_file_name + '_' \
                                               + str(proportionality_std) + '_' \
                                               +'part' + str(ii) + '.json')

    generalProcess(method=method,
                   proportionality_std=proportionality_std,
                   path_json_dict_query_phrases=path_json_dict_query_phrases,
                   am=am,
                   files=parts[ii])

if __name__ == '__main__':

    files = [filename for filename in getRecordingNamesSimi('TEST', class_name)]

    part0 = files[:4]
    part1 = files[4:8]
    part2 = files[8:]
    parts = [part0,part1,part2]

    runPart(parts, 0, proportionality_std=0.1)
    runPart(parts, 1, proportionality_std=0.1)
    runPart(parts, 2, proportionality_std=0.1)

    mergeResultsDict(proportionality_std=0.1)