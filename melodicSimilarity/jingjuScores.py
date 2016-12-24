# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:57:07 2016

@author: Rafael.Ctt
"""

from music21 import *

def findVoiceParts(score):
    '''music21.stream.Score --> [music21.stream.Part]

    Given a music21.stream.Score with one or more parts, it returns a list of
    the parts that contain lyrics
    '''

    voiceParts = []

    for p in score.parts:
        if len(p.flat.notes) == 0: continue
        i = 0
        n = p.flat.notes[i]
        while n.quarterLength == 0:
            i += 1
            n = p.flat.notes.stream()[i]
        if n.hasLyrics():
                if p.hasElementOfClass('Instrument'):
                    p.remove(p.getInstrument())
                voiceParts.append(p)
#        n = p.flat.notes.stream()[0]
#        if n.quarterLength != 0:
#            if n.hasLyrics():
#                if p.hasElementOfClass('Instrument'):
#                    p.remove(p.getInstrument())
#                voiceParts.append(p)
#        elif n.next().hasLyrics():
#            if p.hasElementOfClass('Instrument'):
#                p.remove(p.getInstrument())
#            voiceParts.append(p)

    return voiceParts

#def cleanScore(filename, showCleanScore=True, slurs=True):
#    '''str --> music21.stream.Score
#
#    filename = string with the path to the file
#    If showCleanScore=True, the resulting score will be showed according to
#        music21 configuration
#    If slurs=True, it draws a slur for each syllable melisma
#    '''
#
#    s = converter.parse(filename)
#
#    # Transpose from C major (from Medeli) to E major
#    s.transpose('M3', classFilterList=['Note'], inPlace=True)
#
#    # Delete barline objects from measures
#    for i in s.parts:
#        for j in i:
#            if j.isStream:
#                j.removeByClass(bar.Barline)
#
#    s.makeNotation(inPlace=True)
#
#    # Add slurs for syllable melismae
#    if slurs:
#        voicePart = findVoicePart(s)
#        allnotes = voicePart.flat.notes.stream()
#        for n in allnotes: # Removing grace notes for slurring
#            if n.quarterLength == 0:
#                allnotes.remove(n)
#        for n in allnotes:
#            if n.hasLyrics() and not n.next().hasLyrics():
#                slurstart = n
#                i = allnotes.index(n) + 1
#                slurend = allnotes[i]
#                while not slurend.next().hasLyrics() and i < len(allnotes):
#                    slurend = allnotes[i]
#                    i += 1
#                slur = spanner.Slur([slurstart, slurend])
#                voicePart.insert(0, slur)
#
#
#    if showCleanScore: s.show()
#
#    return s

def extractPhrases(filename, lines): #, clean=False, slurs=False):
    '''str, list --> list

    filename: string with the path to the score
    lines: a list of tuples of integers indicating the first and last measure
    of each music phrase.
    '''

#    if clean:
#        s = cleanScore(filename, showCleanScore=False, slurs=slurs)
#    else:
#        s = converter.parse(filename)

    s = converter.parse(filename)

    voicePart = findVoicePart(s)

    fragments = [] # It stores the streams per line

    for line in lines:
        fragment = voicePart.measures(line[0], line[1])
        fragments.append(fragment)

    # Clef, key signature and time signatures in the fragments after using the
    #   the music21.stream.Stream.measures() method are stored in offset 0.0 of
    #   the music21.stream.Part. When different parts are put together to form
    #   a score, this results in an error, so clef, key signature and time
    #   signature should be moved to offset 0.0 in the first meausre of each
    #   part.
#    cl = fragments[0].getClefs()[0]
#    ks = fragments[0].getKeySignatures()[0]
#    ts = fragments[0].getTimeSignatures()[0]
#
#    toRemove = [cl, ks, ts]
#    toInsert = [0, cl, 0, ks, 0, ts]
#
#    for fragment in fragments:
#        fragment.remove(toRemove)
#        fragment[1].insert(toInsert)

    return fragments

def alignLines(scores, showAlignedScore=True): #, clean=False, slurs=False):
    '''{str:[()]}, list --> music21.stream.Score

    scores: dictionary whose keys are strings with the path to the score and
    whose values are lists of tuples of integers indicating the first and last
    measure of each music phrase.
    '''

    parts = [] # It stores the streams per line

    longestLength = 0 # Finding the longest measure length

    for score in scores:
        s = converter.parse(score)
        print(score.split('/')[-1] + ' parsed.')
        voicePart = findVoicePart(s)
        fragments = [] # It stores the streams per line
        for line in scores[score]:
            fragment = voicePart.measures(line[0], line[1])
            fragment.remove(fragment.getTimeSignatures()[0])
            if len(fragment.getElementsByClass('Measure')) > longestLength:
                longestLength = len(fragment.getElementsByClass('Measure'))
            fragments.append(fragment)
        parts.extend(fragments)

    # Completing parts with empty measures so that all of them have the same
    #   length as the longest.

    for part in parts:
        partLength = len(part.getElementsByClass('Measure'))
        if partLength < longestLength:
            part.repeatAppend(stream.Measure(), longestLength-partLength)
    print('\nExtra empty measures appended.\n\nAligning parts...')

    alignedScore = stream.Score()
    for part in parts:
        alignedScore.insert(0, part)

    alignedScore.makeNotation(inPlace=True)

    print('\nDone!')

    if showAlignedScore:
        print ('\nOpening aligned score with MuseScore')
        alignedScore.show()

    return alignedScore

def changeDurations(score, value, showScore=True, save=False):
    '''str, int --> music21.stream.Score

    Given a string with the address of a xml score and an integer, it returns a
    score without time signature and with all the durations multiplied by the
    given integer.
    '''

    s = converter.parse(score)

    for p in s.parts:
        try:
            p.measure(0).removeByClass('TimeSignature')
        except:
            p.measure(1).removeByClass('TimeSignature')

    for n in s.flat.notesAndRests:
        n.quarterLength = n.quarterLength * 2

    if showScore:
        s.show()

    if save:
        scoreName = score[:-4] + '-x' + str(value) + '.xml'
        s.write(fp=scoreName)

# Chinese punctuation marcs: ，。？！
#diacritics = [u'\uff0c', u'\u3002', u'\uff1f', u'\uff01']
diacritics = ['，', '。', '？', '！', '；', '：']

def lyricsFromPart(part, printLyrics=False):
    '''music21.stream.Part --> str
    It takes a music21.stream.Part as input and returns its lyrics. If
    printLyrics is True, it also prints them in the console.
    '''

    notes = part.flat.notes

    rawlyrics = ''
    lyrics = ''
    lines = 0

    for n in notes:
        if n.hasLyrics(): rawlyrics += n.lyric

    for i in range(len(rawlyrics)):
        if rawlyrics[i] not in diacritics:
            lyrics += rawlyrics[i]
        elif rawlyrics[i] != diacritics[0]: # Chinese comma ，
            lyrics += (rawlyrics[i] + '\n')
            lines += 1
        else:
            if i < len(rawlyrics)-5:
                condition1 = ((rawlyrics[i+4] not in diacritics) and
                              (rawlyrics[i+5] not in diacritics) and
                              (rawlyrics[i+6] not in diacritics))
                condition2 = ((rawlyrics[i-4] not in diacritics) and
                              (rawlyrics[i-5] not in diacritics) and
                              (rawlyrics[i-6] not in diacritics))
                if condition1 and condition2:
                    lyrics += (rawlyrics[i] + '\n')
                    lines += 1
                else:
                    lyrics += rawlyrics[i]
            else:
                lyrics += (rawlyrics[i] + '\n')
                lines += 1

    if lyrics[-1] != '\n':
        lyrics += '\n'

    print('One part with', str(lines), 'lines')

    if printLyrics: print(lyrics)

    return lyrics

def lyricsFromScore(filename, printLyrics=False):
    '''music21.stream.Score --> str
    It takes a music21.stream.Score as input and returns the lyrics of all the
    parts that contain lyrics. If printLyrics is True, it also prints them in
    the console.
    '''

    print('Parsing ' + filename.split('/')[-1])
    s = converter.parse(filename)

    voiceParts = findVoiceParts(s)

    if len(voiceParts) == 1:
        lyrics = lyricsFromPart(voiceParts[0], printLyrics)
    else:
        lyrics = ''
        for p in voiceParts:
            heading = 'Part ' + str(voiceParts.index(p) + 1) + '\n'
            partLyrics = lyricsFromPart(p, printLyrics=False)
            lyrics += (heading + partLyrics)
            if printLyrics: print(heading + partLyrics)

    return lyrics

def lyrics2csv(scores, csv, printLyrics=True):
    '''[str], str --> csv file
    Given a list of paths for xml scores, it creates a csv file with all the
    lyrics per score in the given 'csv' path. If printLyrics is true, it prints
    the lyrics of each file in the cosole.
    '''
    for s in scores:
        lyrics = lyricsFromScore(s, printLyrics)
        with open(csv, 'a', encoding='utf-8') as f:
            f.write(s.split('/')[-1] + ',' + lyrics.split('\n')[0])
            for l in lyrics.split('\n')[1:-1]:
                f.write('\n,' + l)
            f.write('\n')

def partSegmentation(part, printLyrics):
    '''music21.stream.Part --> str
    It takes a music21 part as an input and it returns a string in the csv file
    format to be used in the lyricsSegmentation function
    '''
    lyrics = lyricsFromPart(part, printLyrics)
    notes = part.flat.notes.stream()
    offsets = [notes[0].offset]
    index = 0
    limit = False

    for n in notes:
        if not n.hasLyrics(): continue
        nl = n.lyric
        ll = lyrics[index:index+len(nl)]
        if nl != ll:
            print('Error at index ', str(index), ' (', nl, ')')
            break
        if (lyrics[index+len(nl):index+len(nl)+1] == '\n') and (lyrics[index-1]
            != '\n'):
            limit = True
            index += (len(nl) + 1)
        else:
            if limit:
                offsets.append(n.previous().offset)
                offsets.append(n.offset)
                limit = False
                index += len(nl)
            else:
                index += len(nl)
    offsets.append(notes[-1].offset)

    partLyrics = ','
    index = 0
    for line in lyrics.split('\n')[:-1]:
        partLyrics += (line + ',' + str(offsets[index]) + ',' +
                      str(offsets[index+1]) + '\n,')
        index += 2

    return partLyrics

def lyricsSegmentation(scores, csv, printLyrics=False):
    '''[str], str --> csv file
    Given a list of paths for xml scores, it creates a csv file in the given
    'csv' path with all the lyrics per score divede by lines plus the offset of
    the first and last note of each line. If printLyrics is true, it prints the
    lyrics of each file in the cosole.
    '''

    for score in scores:
        print('Parsing ' + score.split('/')[-1])
        s = converter.parse(score)

#        lyrics = lyricsFromScore(score, printLyrics)

        voiceParts = findVoiceParts(s)

#        for p in voiceParts:
#            offsets = [p.flat.notes[0].offset]
#
#            notes = p.flat.notes.stream()
#            index = 0
#            limit = False
#
#            for n in notes:
#                if not n.hasLyrics(): continue
#                nl = n.lyric
#                ll = lyrics[index:index+len(nl)]
#                if nl != ll:
#                    print('Error at index ', str(index), ' (', nl, ')')
#                    break
#                if lyrics[index+len(nl):index+len(nl)+1] == '\n':
#                    limit = True
#                    index += (len(nl) + 1)
#
#                else:
#                    if limit:
#                        offsets.append(n.previous().offset)
#                        offsets.append(n.offset)
#                        limit = False
#                        index += len(nl)
#                    else:
#                        index += len(nl)
#            offsets.append(notes[-1].offset)

        if len(voiceParts) == 1:
            partLyrics = partSegmentation(voiceParts[0], printLyrics)
        else:
            partLyrics = ','
            for p in voiceParts:
                onePartLyrics = partSegmentation(p, printLyrics)
                partLyrics += ('Part ' + str(voiceParts.index(p)+1) + '\n' +
                               onePartLyrics)

        finalFile = score.split('/')[-1] + partLyrics

#        index = 0
#        for line in lyrics.split('\n')[:-1]:
#            finalFile += (line + ',' + str(offsets[index]) + ',' +
#                          str(offsets[index+1]) + '\n,')
#            index += 2

        with open(csv, 'a', encoding='utf-8') as f:
            f.write(finalFile[:-1])

def getMelodicLine(filename, start, end, partIndex=1, show=False):
    '''str, float, float --> music21.stream.Stream
    Given the file path to a jingju score, and the offset value of the first
    and last note of a given line, it returns a music21 stream with that line.
    If the score has more than one singing parts, the part index should be
    introduced (strating from 1). If show is True, the line is shown.
    '''

    print('Parsing ' + filename.split('/')[-1])
    s = converter.parse(filename)

    voiceParts = findVoiceParts(s)

    p = voiceParts[partIndex-1]

    line = p.getElementsByOffset(start, end, mustBeginInSpan=False,
                                 includeElementsThatEndAtStart=False)

    if show:
        line.show()

    return line