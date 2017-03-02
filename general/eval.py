# -*- coding: utf-8 -*-
import numpy as np

def stringDist(str0,str1):
    '''
    utf-8 format string
    :param str0:
    :param str1:
    :return:
    '''

    intersection = [val for val in str0 if val in str1]

    dis = len(intersection)/float(len(str0))

    return dis

def MRR(list_rank):
    '''
    mean reciprocal rank
    :param list_rank: [rank_0,rank_1,...], each one starting from 1
    :return:
    '''

    return np.sum(1.0/np.array(list_rank))/len(list_rank)

def topXhit(X,list_rank):
    '''
    :param X:
    :param list_rank:
    :return:
    '''
    counter = 0
    for rank in list_rank:
        if rank <= X:
            counter+=1
    return counter/float(len(list_rank))