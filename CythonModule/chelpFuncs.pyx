# This is a cython wrapper around helpFuncs written in C.
#  
#   Author: Rong Gong
#   email: rong.gong@upf.edu
#   Affiliation: Universitat Pompeu Fabra
#   
#   License: to be decided !!!   
#

import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from chelpFuncs cimport *

np.import_array()

def viterbiLoopHelperCython(delta_t_minus_one_in,A_log_in,delta_t_in,psi_t_in):
	"""
	loop function used in _viterbiLog
	delta_t_minus_one_in 	= delta[t-1][:]
	A_log_in 				= A_log
	delta_t_in 				= delta[t,:]
	psi_t_in 				= psi[t,:]
	"""
	A_log_in = np.transpose(A_log_in)

	cdef np.ndarray[np.float_t, ndim=1] delta_t_minus_one
	cdef np.ndarray[np.float_t, ndim=2] A_log
	cdef np.ndarray[np.float_t, ndim=1] delta_t
	cdef np.ndarray[np.float_t, ndim=1] psi_t

	delta_t_minus_one 	= np.ascontiguousarray(delta_t_minus_one_in, 	dtype=np.float)
	A_log 				= np.ascontiguousarray(A_log_in, 				dtype=np.float)
	delta_t				= np.ascontiguousarray(delta_t_in, 				dtype=np.float)
	psi_t				= np.ascontiguousarray(psi_t_in, 				dtype=np.float)

	viterbiLoopHelper(&delta_t[0], &psi_t[0], <double *>delta_t_minus_one.data, <double *>A_log.data, delta_t_minus_one.shape[0]);

	return delta_t,psi_t
