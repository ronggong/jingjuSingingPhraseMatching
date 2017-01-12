from os import path
import csv

def getRankNPhraseName(path_results,query_phrase_name,N):
    '''
    get a dictionary contains all infos of the query phrase
    :param path_results:
    :param query_phrase_name:
    :return:
    '''
    with open(path.join(path_results, query_phrase_name+'.csv'), 'rb') as csvfile:
        result_info = csv.reader(csvfile)
        list_phrase_name_rank_N = []
        for ii, row_ii in enumerate(result_info):
            list_phrase_name_rank_N.append(row_ii[0])
            if ii > N-1:
                break
        list_phrase_name_rank_N = list_phrase_name_rank_N[1:]
    return list_phrase_name_rank_N

def filterDictScoreRankN(dict_scores, list_phrase_name_rank_N):
    dict_scores_rank_N = {}
    for phrase_name in list_phrase_name_rank_N:
        dict_scores_rank_N[phrase_name] = dict_scores[phrase_name]
    return dict_scores_rank_N

def runDictScoreRankNFilter(dict_scores,path_results,query_phrase_name,N):
    list_phrase_name_rank_N = getRankNPhraseName(path_results,query_phrase_name,N)
    dict_scores_rank_N = filterDictScoreRankN(dict_scores,list_phrase_name_rank_N)
    return dict_scores_rank_N