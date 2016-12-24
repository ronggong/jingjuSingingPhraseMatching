# This is a cython wrapper around helpFuncs written in C.
#  
#   Author: Rong Gong
#   email: rong.gong@upf.edu
#   Affiliation: Universitat Pompeu Fabra
#   
#   License: to be decided !!!   
#

cdef extern from "helpFuncs.h": 

	int viterbiLoopHelper(double *delta_t, double *psi_t, double *delta_t_minus_one, double *A_log, int n)