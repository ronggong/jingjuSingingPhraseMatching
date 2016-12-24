from os.path import join,dirname

root_path    = join(dirname(__file__),'..')

# class_name = 'dan'
class_name = 'laosheng'

if class_name == 'dan':
    base_path = 'sourceSeparation'
    syllableTierName = 'pinyin'
elif class_name == 'laosheng':
    base_path = 'qmLonUpf/laosheng'
    syllableTierName = 'dian'
phonemeTierName = 'details'

dataset_path = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/'

wav_path_dan        = join(dataset_path,'wav','sourceSeparation')
wav_path_laosheng        = join(dataset_path,'wav','qmLonUpf/laosheng')

textgrid_path_dan   = join(dataset_path,'textgrid','sourceSeparation')
textgrid_path_laosheng   = join(dataset_path,'textgrid','qmLonUpf/laosheng')

feature_output_path_dan = join(root_path,'dan_features.csv')
feature_output_path_laosheng = join(root_path,'laosheng_features.csv')

feature_output_path_dan_test = join(root_path,'dan_features_test.csv')
feature_output_path_laosheng_test = join(root_path,'laosheng_features_test.csv')

scaler_path     = join(root_path,'roleTypeClassification','scaler')
classifier_path     = join(root_path,'roleTypeClassification','classifier')

score_path = '/Users/gong/Documents/MTG document/Jingju arias/Scores'
score_info_file_old = '0. Lyrics.csv'
score_info_file_shenqiang_banshi = '0. Lyrics_withShenqiangBanshi.csv'

gmmModels_path = join(root_path,'phoneticSimilarity','gmmModels')