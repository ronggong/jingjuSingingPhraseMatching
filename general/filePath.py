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

from os.path import join,dirname
from parameters import am

###################################
###### set this dataset path ######
###################################

# dataset_path = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/'
dataset_path = '/path/to/your/jingju_a_cappella_singing_dataset/'

root_path    = join(dirname(__file__),'..')

# role-type class
class_name = 'danAll' # dan role-type
#class_name = 'laosheng' # laosheng role-type

# acoustic model if equals to GMMs
# no cnn_filename string defined
if am == 'gmm':
    cnn_file_name = ''
else:
    cnn_file_name = 'mfccBands_2D_all_optim'

if class_name == 'danSource':
    base_path = 'sourceSeparation'
    syllableTierName = 'pinyin'

elif class_name == 'danAll':
    base_path = 'danAll'
    syllableTierName = 'dian'
    gmmModels_path = join(root_path, 'phoneticSimilarity', 'gmmModels/dan')

elif class_name == 'laosheng':
    base_path = 'qmLonUpf/laosheng'
    syllableTierName = 'dian'
    gmmModels_path = join(root_path, 'phoneticSimilarity', 'gmmModels/laosheng')

phonemeTierName = 'details'

wav_path_dan                = join(dataset_path,'wav','danAll')
wav_path_laosheng           = join(dataset_path,'wav','qmLonUpf/laosheng')

textgrid_path_dan           = join(dataset_path,'textgrid','danAll')
textgrid_path_laosheng      = join(dataset_path,'textgrid','qmLonUpf/laosheng')

if class_name == 'danAll':
    textgrid_path = textgrid_path_dan
elif class_name == 'laosheng':
    textgrid_path = textgrid_path_laosheng

# paths for role type classification

# training feature
feature_output_path_dan         = join(root_path,'roleTypeClassification','dan_features.csv')
feature_output_path_laosheng    = join(root_path,'roleTypeClassification','laosheng_features.csv')

# test feature
feature_output_path_dan_test        = join(root_path,'roleTypeClassification','dan_features_test.csv')
feature_output_path_laosheng_test   = join(root_path,'roleTypeClassification','laosheng_features_test.csv')

scaler_path         = join(root_path,'roleTypeClassification','scaler')
classifier_path     = join(root_path,'roleTypeClassification','classifier')

score_path                          = '/Users/gong/Documents/MTG document/Jingju arias/Scores'
score_info_file_old                 = '0. Lyrics.csv' # not used
score_info_file_shenqiang_banshi    = '0. Lyrics_withShenqiangBanshi.csv'

# path for keras cnn models
kerasScaler_path        = join(root_path, 'phoneticSimilarity', 'cnnModels', base_path,
                               'scaler_' + class_name + '_phonemeSeg_mfccBands2D.pkl')
kerasModels_path        = join(root_path, 'phoneticSimilarity', 'cnnModels', base_path,
                               'keras.cnn_' + cnn_file_name + '.h5')