import sys,os
import matplotlib.pyplot as plt

# the project folder: fileDir
fileDir = os.path.dirname(os.path.realpath('__file__'))
dtwPath = os.path.join(fileDir, '../../Library_PythonNew/similarityMeasures/dtw/')
# transcribe the path string to full path
dtwPath = os.path.abspath(os.path.realpath(dtwPath))

sys.path.append(dtwPath)

import dtw
import numpy as np

def dtw1d_generic(x, y):

    configuration = {}
    configuration['distType'] = 1         # square euclidean
    configuration['hasGlobalConst'] = 0
    configuration['globalType'] = 0
    configuration['bandwidth'] = 0.2
    configuration['initCostMtx'] = 1
    configuration['reuseCostMtx'] = 0
    configuration['delStep'] = 1        # one is horizontal, another is vertical, but I don't which is which
    configuration['moveStep'] = 2
    configuration['diagStep'] = 1
    configuration['initFirstCol'] = 1
    configuration['isSubsequence'] = 0

    udist,_,path,_  = dtw.dtw1d_GLS(x, y, configuration)

    return udist,path

def dtw1d_std(x,y):

    configuration = {}
    configuration['Output'] = 3
    configuration['Ldistance'] = {}
    configuration['Ldistance']['type'] = 0

    udist,_,path = dtw.dtw1d(x,y,configuration)
    return udist,path

def dtwNd(x,y):
    configuration = {}
    configuration['Output'] = 2
    configuration['Ldistance'] = {}
    configuration['Ldistance']['type'] = 0
    configuration['Ldistance']['weight'] = np.ones((x.shape[1],))/float(x.shape[1])

    udist,plen = dtw.dtwNd(x,y,configuration)
    return udist,plen

def plotDTW(path,cost_arr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(cost_arr)
    plt.show()