# -*- coding: utf-8 -*-

from jingjuScores import getMelodicLine
from general.filePath import *
from general.parameters import *
from general.utilsFunctions import hz2cents,pitchtrackInterp
import numpy as np
from os import path
import csv,json

# score information *.csv
score_info_filepath = path.join(score_path,score_info_file_shenqiang_banshi)

##-- for score old
# def getDictScoreInfo(score_info_filepath):
#     dict_score_info = {}
#     with open(score_info_filepath, 'rb') as csvfile:
#         score_info = csv.reader(csvfile)
#
#         aria_name_old = ''
#         line_number = 0
#         for row in score_info:
#             if row[0] != 'File name':
#                 aria_name = row[0]
#                 if len(aria_name):
#                     # row[0] != empty
#                     line_number = 0
#                     aria_name_old = aria_name
#                     part = 1
#                 else:
#                     line_number += 1
#
#                 # print aria_name_old
#                 if aria_name_old[:2] == 'ls':
#                     roleType = 'laosheng'
#                 elif aria_name_old[:2] == 'da':
#                     roleType = 'dan'
#
#                 if row[1][:4] == 'Part':
#                     # if lyric is 'Part', redefine roleType and part
#                     roleType = row[1].split()[2]
#                     part = int(row[1].split()[1])
#
#                 if row[1][:4] != 'Part':
#                     try:
#                         dict_score_info[aria_name_old+'_'+str(line_number)] = {'lyrics':row[1],
#                                                                                'startEndOffset':[float(row[2]),float(row[3])],
#                                                                                'part':part,
#                                                                                'role_type':roleType}
#                     except ValueError:
#                         print(aria_name_old+'_'+str(line_number)+' '+'valueError: '+row[2]+' '+row[3])
#
#         return dict_score_info

def getDictScoreInfo(score_info_filepath):
    dict_score_info = {}
    with open(score_info_filepath, 'rb') as csvfile:
        score_info = csv.reader(csvfile,delimiter=";")

        aria_name_old = ''
        line_number = 0
        for row in score_info:
            if row[0] != 'File name':
                aria_name = row[0]
                if len(aria_name):
                    # row[0] != empty
                    line_number = 0
                    aria_name_old = aria_name
                    part = 1
                else:
                    line_number += 1

                if row[5][:4] == 'Part':
                    # if lyric is 'Part', redefine roleType and part
                    # roleType = row[1].split()[2]
                    part = int(row[5].split()[1])

                if row[5][:4] != 'Part':
                    try:
                        dict_score_info[aria_name_old + '_' + str(line_number)] = {'lyrics': row[5],
                                                                                   'startEndOffset': [float(row[6]),
                                                                                                      float(
                                                                                                          row[7])],
                                                                                   'part': part,
                                                                                   'roletype': row[1]}
                                                                                   # 'shengqiang': row[2],
                                                                                   # 'banshi': row[3],
                                                                                   # 'couplet': row[4]}
                    except ValueError:
                        print(aria_name_old + '_' + str(line_number) + ' ' + 'valueError: ' + row[6] + ' ' + row[7])

        return dict_score_info

def getScores(key, dict_score_info):
    score_filename = key.split('_')[0]
    start = dict_score_info['startEndOffset'][0]
    end = dict_score_info['startEndOffset'][1]
    part = dict_score_info['part']
    score_file_path = path.join(score_path,score_filename)
    line = getMelodicLine(score_file_path, start, end, partIndex=part, show=False)

    notes = []
    for note in line.flat.notes.stream():
        notes.append({'freq':note.pitch.freq440,'lyric':note.lyric,'quarterLength':float(note.quarterLength)})

    dict_score_info['notes'] = notes
    return dict_score_info

def melodySynthesize(notes_pitch_hz,notes_quarterlength):
    '''
    :param notes_quarterlength: a list of the note quarterLength
    :return: list, pitch track values
    '''
    notes_pitch_cents = hz2cents(np.array(notes_pitch_hz))

    length_total = sum(notes_quarterlength)

    melody = []
    for ii in range(len(notes_quarterlength)):
        sample_note = int(round((notes_quarterlength[ii]/length_total)*sample_number_total))
        melody += [notes_pitch_cents[ii]]*sample_note

    melody = np.array(melody)

    # interpolation
    if len(melody) != sample_number_total:
        melody = pitchtrackInterp(melody)

    melody = melody - np.mean(melody)

    return melody.tolist()

##-- dump json scores
if __name__ == '__main__':

    dict_score_infos = getDictScoreInfo(score_info_filepath)

    for key in dict_score_infos:
        dict_score_info = dict_score_infos[key]
        dict_score_info = getScores(key, dict_score_info)

        # synthesize melody
        notes_quarterLength = []
        notes_pitch_hz = []
        for dict_note in dict_score_infos[key]['notes']:
            notes_pitch_hz.append(dict_note['freq'])
            notes_quarterLength.append(dict_note['quarterLength'])

        pitchtrack_cents = melodySynthesize(notes_pitch_hz,notes_quarterLength)
        dict_score_info['pitchtrack_cents'] = pitchtrack_cents

        dict_score_infos[key] = dict_score_info


    # print dict_score_infos[key]
    with open('scores.json','w') as outfile:
        json.dump(dict_score_infos,outfile)

    # with open('scores.json','r') as f:
    #     dict_score_infos = json.load(f)
