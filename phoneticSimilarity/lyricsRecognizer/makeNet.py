# import os,sys
# import json,pickle

# currentPath = os.path.dirname(__file__)
# parentPath = os.path.join(currentPath, '..')
# sys.path.append(parentPath)

from scoreManip import retrieveSylInfo
from general.phonemeMap import *
from general.pinyinMap import *
import numpy as np
from scipy.linalg import block_diag

TRANS_PROB_SELF = 0.9
TRANS_PROB_NEXT = 1-TRANS_PROB_SELF

def singleTransMatBuild(dict_score_info):
    syl_finals,_,_ = retrieveSylInfo(dict_score_info)
    # print syl_finals,syl_durations

    state_pho = []
    for sf in syl_finals:
        pho_final = dic_final_2_sampa[sf]
        for pho in pho_final:
            pho_map = dic_pho_map[pho]
            if pho_map == u'H':
                pho_map = u'y'
            elif pho_map == u'9':
                pho_map = u'S'
            elif pho_map == u'yn':
                pho_map = u'in'
            state_pho.append(pho_map)

    num_state = len(state_pho)
    mat_trans = np.zeros((num_state,num_state))
    for ii in range(num_state-1):
        mat_trans[ii][ii] = TRANS_PROB_SELF
        mat_trans[ii][ii+1] = TRANS_PROB_NEXT
    mat_trans[-1][-1] = 1.0
    return mat_trans,state_pho

def multiTransMats(dict_score_infos):
    phrases = []
    lyrics = []
    mats_trans = []
    states_pho = []
    for key in dict_score_infos:
        mat_trans,state_pho = singleTransMatBuild(dict_score_infos[key])
        phrases.append(key)
        lyrics.append(dict_score_infos[key]['lyrics'])
        mats_trans.append(mat_trans)
        states_pho.append(state_pho)
    return phrases,lyrics,mats_trans,states_pho

def combineTransMats(mats_trans,states_pho):
    mat_trans_comb = block_diag(*mats_trans)
    state_pho_comb = sum(states_pho, [])
    index_start = [0]
    index_end = [mats_trans[0].shape[0]-1]
    for ii in range(1,len(mats_trans)):
        index_start.append(index_end[-1]+1)
        index_end.append(index_end[-1]+mats_trans[ii].shape[0])
    return mat_trans_comb,state_pho_comb,index_start,index_end

def makeNet(dict_scores):

    phrases,lyrics,mats_trans,states_pho = multiTransMats(dict_scores)
    mat_trans_comb,state_pho_comb,index_start,index_end = combineTransMats(mats_trans,states_pho)

    return phrases,lyrics,mat_trans_comb,state_pho_comb,index_start,index_end

# if __name__=='__main__':
    # phrases,mat_trans_comb,state_pho_comb,index_start,index_end = makeNet()
    #
    # dict_net = {'phrases':phrases,
    #                'mat_trans_comb':mat_trans_comb,
    #                'state_pho_comb':state_pho_comb,
    #                'index_start':index_start,
    #                'index_end':index_end}
    #
    # output = open('dict_net.pkl', 'wb')
    # pickle.dump(dict_net, output)
    # output.close()

    # print index_start
    # print index_end
    # print len(state_pho_comb)