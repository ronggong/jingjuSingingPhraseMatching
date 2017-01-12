from os.path import join,dirname

root_path    = join(dirname(__file__),'..')

class_name = 'danAll'
# class_name = 'laosheng'

if class_name == 'danSource':
    base_path = 'sourceSeparation'
    syllableTierName = 'pinyin'
elif class_name == 'danAll':
    base_path = 'danAll'
    syllableTierName = 'dian'
    gmmModels_path = join(root_path, 'phoneticSimilarity', 'gmmModels/dan')
    dnnModels_path = join(root_path, 'phoneticSimilarity', 'dnnModels/dan')
    dnnModels_cfg_path = join(root_path, 'phoneticSimilarity',
                              'dnnModels/dan/dnn_danAll_phraseMatching_layers_2_512.cfg')
    dnnModels_param_path = join(root_path, 'phoneticSimilarity',
                              'dnnModels/dan/dnn_danAll_phraseMatching_layers_2_512.param')
elif class_name == 'laosheng':
    base_path = 'qmLonUpf/laosheng'
    syllableTierName = 'dian'
    gmmModels_path = join(root_path, 'phoneticSimilarity', 'gmmModels/laosheng')
    dnnModels_cfg_path = join(root_path, 'phoneticSimilarity',
                              'dnnModels/laosheng/dnn_laosheng_phraseMatching_layers_2_512.cfg')
    dnnModels_param_path = join(root_path, 'phoneticSimilarity',
                              'dnnModels/laosheng/dnn_laosheng_phraseMatching_layers_2_512.param')
phonemeTierName = 'details'

dataset_path = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/'

wav_path_dan                = join(dataset_path,'wav','danAll')
wav_path_laosheng           = join(dataset_path,'wav','qmLonUpf/laosheng')

textgrid_path_dan           = join(dataset_path,'textgrid','danAll')
textgrid_path_laosheng      = join(dataset_path,'textgrid','qmLonUpf/laosheng')

feature_output_path_dan         = join(root_path,'roleTypeClassification','dan_features.csv')
feature_output_path_laosheng    = join(root_path,'roleTypeClassification','laosheng_features.csv')

feature_output_path_dan_test        = join(root_path,'roleTypeClassification','dan_features_test.csv')
feature_output_path_laosheng_test   = join(root_path,'roleTypeClassification','laosheng_features_test.csv')

scaler_path         = join(root_path,'roleTypeClassification','scaler')
classifier_path     = join(root_path,'roleTypeClassification','classifier')

score_path                          = '/Users/gong/Documents/MTG document/Jingju arias/Scores'
score_info_file_old                 = '0. Lyrics.csv'
score_info_file_shenqiang_banshi    = '0. Lyrics_withShenqiangBanshi.csv'

dnnModels_base_path = join(root_path, 'phoneticSimilarity', 'dnnModels')