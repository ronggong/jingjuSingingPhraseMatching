import pandas as pd

from feature_extraction import getLineList,extract_for_one,convert_pool_to_dataframe
from xgb_classification import feature_preprocessing,feature_scaling_test,imputer_run,buildEstimators

FILE_EXT_WAV = (".wav")
filename_feature = 'temp_feature.csv'

def roletypeProcess(wavDataDir,line_list,filename,query_phrase_name):
    line_list_current = getLineList(wavDataDir,line_list[0],filename)
    pool = extract_for_one(wavDataDir, line_list_current, filename, FILE_EXT_WAV)
    feature_set = pd.DataFrame()
    feature_set = feature_set.append(convert_pool_to_dataframe(pool, None, query_phrase_name))

    feature_set.to_csv(filename_feature)

    X,_ = feature_preprocessing(filename_feature)
    X = feature_scaling_test(X)
    X = imputer_run(X)
    clf = buildEstimators('test')

    return clf.predict_proba(X)







