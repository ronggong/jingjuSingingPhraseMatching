from general.filePath import *
from subprocess import call

# train

# call(['python','feature_extraction.py',wav_path_dan,textgrid_path_dan,'dan',feature_output_path_dan,'TRAIN'])
# call(['python','feature_extraction.py',wav_path_laosheng,textgrid_path_laosheng,'laosheng',feature_output_path_laosheng,'TRAIN'])

# # cv
# call(['python','xgb_classification.py',feature_output_path_dan,feature_output_path_laosheng,'dan','laosheng','cv'])

# # train model put in classifier folder
# call(['python','xgb_classification.py',feature_output_path_dan,feature_output_path_laosheng,'dan','laosheng','train'])

# test

# call(['python','feature_extraction.py',wav_path_dan,textgrid_path_dan,'dan',feature_output_path_dan_test,'TEST'])
# call(['python','feature_extraction.py',wav_path_laosheng,textgrid_path_laosheng,'laosheng',feature_output_path_laosheng_test,'TEST'])

# call(['python','xgb_classification.py',feature_output_path_dan_test,feature_output_path_laosheng_test,'dan','laosheng','test'])
