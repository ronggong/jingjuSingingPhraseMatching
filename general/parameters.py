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

fs = 44100

# for the melodic similarity
framesize_melodicSimilarity = 2048
hopsize_melodicSimilarity = 1024

synthesizeLength = 5 # in second
sample_number_total = int(round(synthesizeLength * (fs / float(hopsize_melodicSimilarity))))

list_N_frames = [50,100,150,200,250,300,350,400,450,500,550,600]

# for the phonetic  similarity
framesize_phoneticSimilarity   = 1024
hopsize_phoneticSimilarity     = 512

framesize_t_phoneticSimiarity = framesize_phoneticSimilarity / float(fs)
hopsize_t_phoneticSimilarity = hopsize_phoneticSimilarity / float(fs)

# MFCC parameters
highFrequencyBound = fs/2 if fs/2<11000 else 11000

# acoustic model
am = 'cnn'