import json,csv
from os import path
from general.trainTestSeparation import getRecordings
from melodicSimi import plot3Pitch, plot2PitchAlign

with open('scores.json','r') as f:
    dict_score = json.load(f)
with open('list_rank_900_0.7.json','r') as f:
    list_rank_best = json.load(f)
with open('list_rank_1000_0.85.json','r') as f:
    list_rank_worse = json.load(f)

def findTop3Diff():

    top3Diff = {}
    for query_phrase_name in list_rank_best:
        rank_best = list_rank_best[query_phrase_name][0]
        rank_worse = list_rank_worse[query_phrase_name][0]
        print rank_best,rank_worse
        if rank_best <=10 and rank_worse > 10:
            top3Diff[query_phrase_name] = [list_rank_best[query_phrase_name],list_rank_worse[query_phrase_name]]
    return top3Diff

def findLargerTop3():
    phrase_name_larger_3 = []
    for query_phrase_name in list_rank_best:
        rank_best = list_rank_best[query_phrase_name][0]
        if rank_best > 3:
            phrase_name_larger_3.append(query_phrase_name)
    return phrase_name_larger_3

def larger3Plot(dict_query_pitchtracks,
               path_results,
               path_results_dtw_align,
               path_fig):
    '''
    plot rank larger than 3 query, ground truth and best match pitch tracks, dtw alignment
    :param dict_query_pitchtracks:
    :param path_results:
    :param path_results_dtw_align:
    :param path_fig:
    :return:
    '''

    ##-- collect error phrases from results into a list of dict
    phrase_name_larger_3 = findLargerTop3()
    error_phrases = []
    for pn in phrase_name_larger_3:
        dict_error_phrase = getDictErrorPhrase(path_results,pn)
        error_phrases.append(dict_error_phrase)

    print 'error phrase number:',len(error_phrases)

    for ep in error_phrases:
        figPlot(path_results_dtw_align
                ,ep,
                dict_query_pitchtracks,
                save_fig=True,
                path_fig=path_fig,
                name_fig=ep['query_phrase_name']+'_rank>3')
    return error_phrases

def getDictErrorPhrase(path_results,query_phrase_name):
    '''
    get a dictionary contains all infos of the query phrase
    :param path_results:
    :param query_phrase_name:
    :return:
    '''
    with open(path.join(path_results, query_phrase_name+'.csv'), 'rb') as csvfile:
        result_info = csv.reader(csvfile)
        list_result_info = []
        for ii, row_ii in enumerate(result_info):
            list_result_info.append(row_ii)
        groundtruth_rank = int(list_result_info[0][1])
        best_match_phrase_name = list_result_info[1][0]
        groundtruth_phrase_name = list_result_info[groundtruth_rank+1][0]
        groundtruth_phrase_dtw_score = list_result_info[groundtruth_rank+1][2]

        dict_error_phrase = {'query_phrase_name':query_phrase_name,
                             'groundtruth_phrase_name':groundtruth_phrase_name,
                             'groundtruth_rank':groundtruth_rank,
                             'groundtruth_phrase_dtw_score':groundtruth_phrase_dtw_score,
                             'best_match_phrase_name':best_match_phrase_name
                             }
    return dict_error_phrase

def figPlot(path_results_dtw_align,
            dict_query_phrase,
            dict_query_pitchtracks,
            save_fig=False,
            path_fig=None,
            name_fig=None):
    '''
    plot query, ground truth, best match in fig 1
    plot query, best match, dtw alignment in fig 2
    :param ep: dict_phrase_query_info
    :param dict_query_pitchtracks:
    :return:
    '''
    groundtruth_score_pitchtrack = dict_score[dict_query_phrase['groundtruth_phrase_name']]['pitchtrack_cents']
    best_match_score_pitchtrack = dict_score[dict_query_phrase['best_match_phrase_name']]['pitchtrack_cents']
    query_pitchtrack = dict_query_pitchtracks[dict_query_phrase['query_phrase_name']]

    print('query_phrase_name: '+dict_query_phrase['query_phrase_name'])
    print('ground truth rank: '+str(dict_query_phrase['groundtruth_rank']))

    plot3Pitch(query_pitchtrack,
               groundtruth_score_pitchtrack,
               best_match_score_pitchtrack,
               dict_query_phrase['query_phrase_name'],
               dict_query_phrase['groundtruth_rank'],
               save_fig,
               path_fig,
               name_fig)

    #-- it doesn't work until now
    plot2PitchAlign(path_results_dtw_align,
                    query_pitchtrack,
                    groundtruth_score_pitchtrack,
                    dict_query_phrase['groundtruth_phrase_dtw_score'],
                    dict_query_phrase['groundtruth_rank'],
                    dict_query_phrase['query_phrase_name'],
                    dict_query_phrase['groundtruth_phrase_name'],
                    save_fig,
                    path_fig,
                    name_fig)

top3Diff = findTop3Diff()
print top3Diff

query_pitchtracks_best_filename = 'query_pitchtracks_900_0.7.json'
query_pitchtracks_worse_filename = 'query_pitchtracks_1000_0.85.json'

path_results_best = 'results/900_0.7'
path_results_worse = 'results/1000_0.85'

path_results_dtw_align_best = 'resultsPath/900_0.7'
path_results_dtw_align_worse = 'resultsPath/1000_0.85'

# path_fig = 'errorAnalysis/figLarger3'

with open(query_pitchtracks_best_filename,'r') as f:
    dict_query_pitchtracks_best = json.load(f)

with open(query_pitchtracks_worse_filename,'r') as f:
    dict_query_pitchtracks_worse = json.load(f)
'''
for phrase_name in top3Diff:
    dict_phrase_query_best_info = getDictErrorPhrase(path_results_best,phrase_name)
    dict_phrase_query_worse_info = getDictErrorPhrase(path_results_worse,phrase_name)

    figPlot(path_results_dtw_align_best,
            dict_phrase_query_best_info,
            dict_query_pitchtracks_best,
            save_fig=True,
            path_fig=path_fig,
            name_fig=phrase_name+'_best')

    figPlot(path_results_dtw_align_worse,
            dict_phrase_query_worse_info,
            dict_query_pitchtracks_worse,
            save_fig=True,
            path_fig=path_fig,
            name_fig=phrase_name+'_worse')

with open(path.join('errorAnalysis/pitch_track_params.csv'),'wb') as csvfile:
    w = csv.writer(csvfile)
    for phrase_name in top3Diff:
        dict_phrase_query_best_info = getDictErrorPhrase(path_results_best,phrase_name)
        dict_phrase_query_worse_info = getDictErrorPhrase(path_results_worse,phrase_name)
        old_rank = dict_phrase_query_worse_info['groundtruth_rank']
        new_rank = dict_phrase_query_best_info['groundtruth_rank']

        w.writerow([phrase_name,old_rank,new_rank])
'''

path_fig = 'errorAnalysis/figLarger3'

error_phrases = larger3Plot(dict_query_pitchtracks_best,
                   path_results_best,
                   path_results_dtw_align_best,
                   path_fig)

# with open('errorAnalysis/larger3.csv','wb') as csvfile:
#     w = csv.writer(csvfile)
#     for ep in error_phrases:
#         w.writerow([ep['query_phrase_name'],ep['groundtruth_rank']])
