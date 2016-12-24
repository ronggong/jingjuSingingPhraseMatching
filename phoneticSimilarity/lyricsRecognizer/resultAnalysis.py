import json
import unicodecsv as csv
from os import path
import numpy as np
from general.eval import *
from general.trainTestSeparation import getRecordings

currentPath = path.dirname(__file__)
results_path = path.join(currentPath,'..','results/lyricsRecognizer')

filenames = getRecordings(results_path)

for filename in filenames:
    with open(path.join(results_path,filename+'.json'),'r') as openfile:
        dict_result = json.load(openfile)

    lyrics          = dict_result['lyrics']
    posteri_probs   = np.array(dict_result['posteri_probas'])
    phrases         = dict_result['phrases']
    query_lyrics    = dict_result['query_lyrics']

    # print query_lyrics
    # print posteri_probs
    # print lyrics

    # sort
    ppinds = posteri_probs.argsort()
    posteri_probs_sorted = posteri_probs[ppinds[::-1]]
    lyrics_sorted = [lyrics[i] for i in ppinds[::-1]]
    phrases_sorted = [phrases[i] for i in ppinds[::-1]]

    list_sdist = []
    for l in lyrics_sorted:
        sdist = stringDist(query_lyrics,l)
        list_sdist.append(sdist)

    order = list_sdist.index(max(list_sdist))

    with open(path.join(results_path,filename+'.csv'),'wb') as csvfile:
            w = csv.writer(csvfile)
            w.writerow([query_lyrics,str(order),lyrics_sorted[order]])
            for ii in range(len(lyrics_sorted)):
                w.writerow([phrases_sorted[ii],lyrics_sorted[ii],posteri_probs_sorted[ii],list_sdist[ii]])

    print posteri_probs_sorted
    print lyrics_sorted
