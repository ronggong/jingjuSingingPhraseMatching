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

import numpy as np
import time
import matplotlib.pyplot as plt

from LRHMM import _LRHMM
from general.phonemeMap import *
from general.parameters import *

import os,sys
# import json

currentPath = os.path.dirname(__file__)
parentPath = os.path.join(currentPath, '../../CythonModule')
sys.path.append(parentPath)

from helpFuncs import viterbiLoopHelperCython


def transcriptionMapping(transcription):
    transcription_maped = []
    for t in transcription:
        transcription_maped.append(dic_pho_map[t])
    return transcription_maped


class ParallelLRHMM(_LRHMM):

    def __init__(self,lyrics,mat_trans_comb,state_pho_comb,index_start,index_end):
        _LRHMM.__init__(self)

        self.lyrics = lyrics
        self.A = mat_trans_comb
        self.transcription = state_pho_comb
        self.idx_final_head = index_start
        self.idx_final_tail = index_end

        self.n              = len(self.transcription)
        self._initialStateDist()

    def _initialStateDist(self):
        '''
        explicitly set the initial state distribution
        '''
        # list_forced_beginning = [u'nvc', u'vc', u'w']
        self.pi     = np.zeros((self.n), dtype=self.precision)

        # each final head has a change to start
        for ii in self.idx_final_head:
            self.pi[ii] = 1.0
        self.pi /= sum(self.pi)

    # def _makeNet(self):
    #     pass

    def _viterbiLog(self, observations, am='gmm', kerasModel=None):
        '''
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.

        delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
        that generates the highest probability.

        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1),
        i.e: the previous state.
        '''
        # similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.
        if am == 'gmm':
            self._mapBGMM(observations)
        elif am == 'cnn':
            # self._mapBDNN(observations)
            self._mapBKeras(observations,kerasModel=kerasModel)

        pi_log  = np.log(self.pi)
        A_log   = np.log(self.A)
        # print A_log[0][0],A_log[0][1],A_log[1][0]

        delta   = np.ones((len(observations),self.n),dtype=self.precision)
        delta   *= -float('Inf')
        psi     = np.zeros((len(observations),self.n),dtype=self.precision)

        # init
        for x,state in enumerate(self.transcription):
            # print state
            delta[0][x] = pi_log[x]+self.B_map[state][0]
            psi[0][x] = 0
        # print delta[0][:]

        # with open('results/lyricsRecognizer/delta_start.json','w') as outfile:
        #     json.dump(delta[0].tolist(),outfile)

        # induction
        for t in xrange(1,len(observations)):
            print t
            delta_t_minus_one_in    = delta[t-1,:]
            A_log_in                = A_log
            delta_t_in              = delta[t,:]
            psi_t_in                = psi[t,:]

            # looping written in cython for accelerating
            delta[t,:],psi[t,:]   = viterbiLoopHelperCython(delta_t_minus_one_in,A_log_in,delta_t_in,psi_t_in)

            for j,state in enumerate(self.transcription):
                delta[t][j] += self.B_map[state][t]
            # raise


        posteri_probs = np.zeros((len(self.idx_final_tail),),dtype=self.precision)
        paths = []
        counter_posteri = 0
        for i in xrange(self.n):
            if i in self.idx_final_tail:
                # endingProb = 0.0

                posteri_probs[counter_posteri] = delta[len(observations)-1][i]
                path    = np.zeros((len(observations)),dtype=self.precision)

                # tracking all parallel paths
                # path backtracing
                #        path = np.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
                path[len(observations)-1] = i
                for j in xrange(1, len(observations)):
                    path[len(observations)-j-1] = psi[len(observations)-j][ path[len(observations)-j] ]
                paths.append(path)
                counter_posteri += 1
            else:
                pass
                # endingProb = -float('Inf')

            # if (p_max < delta[len(observations)-1][i]+endingProb):
            #     p_max = delta[len(observations)-1][i]+endingProb
            #     path[len(observations)-1] = i

        return paths,posteri_probs

    def _pathStateDur(self,path):
        '''
        path states in phoneme and duration
        :param path:
        :return:
        '''
        dur_frame = 1
        state_dur_path = []
        for ii in xrange(1,len(path)):
            if path[ii] != path[ii-1]:
                state_dur_path.append([self.transcription[int(path[ii-1])], dur_frame * hopsize_phoneticSimilarity / float(fs)])
                dur_frame = 1
            else:
                dur_frame += 1
        state_dur_path.append([self.transcription[int(path[-1])], dur_frame * hopsize_phoneticSimilarity / float(fs)])
        return state_dur_path

    def _plotNetwork(self,path):
        self.net.plotNetwork(path)

    def _pathPlot(self,transcription_gt,path_gt,path):
        '''
        plot ground truth path and decoded path
        :return:
        '''

        ##-- unique transcription and path
        transcription_unique = []
        transcription_number_unique = []
        B_map_unique = np.array([])
        for ii,t in enumerate(self.transcription):
            if t not in transcription_unique:
                transcription_unique.append(t)
                transcription_number_unique.append(ii)
                if not len(B_map_unique):
                    B_map_unique = self.B_map[t]
                else:
                    B_map_unique = np.vstack((B_map_unique,self.B_map[t]))

        trans2transUniqueMapping = {}
        for ii in range(len(self.transcription)):
            trans2transUniqueMapping[ii] = transcription_unique.index(self.transcription[ii])

        path_unique = []
        for ii in range(len(path)):
            path_unique.append(trans2transUniqueMapping[path[ii]])

        ##-- figure plot
        plt.figure()
        n_states = B_map_unique.shape[0]
        n_frame  = B_map_unique.shape[1]
        y = np.arange(n_states+1)
        x = np.arange(n_frame) * hopsize_phoneticSimilarity / float(fs)

        plt.pcolormesh(x,y,B_map_unique)
        plt.plot(x,path_unique,'b',linewidth=3)
        plt.xlabel('time (s)')
        plt.ylabel('states')
        plt.yticks(y, transcription_unique, rotation='horizontal')
        plt.show()

    def _getBestMatchLyrics(self,path):
        idx_best_match = self.idx_final_head.index(path[0])
        return self.lyrics[idx_best_match]

    def _getAllInfo(self):
        return