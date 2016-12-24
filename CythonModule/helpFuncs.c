/*  
Author: Rong Gong
email: rong.gong@upf.edu
Affiliation: Universitat Pompeu Fabra

License: to be decided !!!   
*/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "helpFuncs.h"
#include <float.h>
#include <time.h>

int viterbiLoopHelper(double *delta_t, double *psi_t, double *delta_t_minus_one, double *A_log, int n)
{	

	int i = 0;
	int j = 0;
	// printf("0");
	// printf("%f", A_log[1+0*n]);
	// printf("%f", A_log[0]);
	// printf("2");
	// time_t start = time(NULL);
	for (j=0;j<n;j++)
	{
		for (i=0;i<n;i++)
		{
			// printf("%f", delta_t_minus_one[i]);
			// printf("%f", A_log_j[i]);
	    	if (delta_t[j] < delta_t_minus_one[i] + A_log[i+j*n])
		    {
		        delta_t[j] = delta_t_minus_one[i] + A_log[i+j*n];
		        psi_t[j] = i;
		    }
		}
		// printf("%f", delta_t[j]);

	}
	// printf("1");
    // printf("%lf", (double)(time(NULL) - start));
    // getchar();
	return 1;
}