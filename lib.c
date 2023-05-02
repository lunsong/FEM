#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define points(i,j) _points[dim*i+j]
#define    dis(i,j)    _dis[dim*i+j]

void mutual(_points, N_points, dim, indptr,indices, quality, _dis, is_uni)
    int *indptr, *indices, N_points, dim, is_uni;
    double *_points, *quality, *_dis;
{
#pragma omp parallel for
    for(int i=0; i<N_points; i++){
	for(int k=indptr[i]; k<indptr[i+1]; k++){
	    int j=indices[k];
	    if(i<j){
		double d[3], l;
		d[0] = points(i,0) - points(j,0);
		d[1] = points(i,1) - points(j,1);
		if(dim==3)
		    d[2] = points(i,2) - points(j,2);
		else
		    d[2] = 0;
		if(is_uni)
		    l = quality[0];
		else
		    l = (quality[i]+quality[j])/2;
		l = 1 - l/sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
		dis(i,0) -= d[0] * l;
		dis(j,0) += d[0] * l;
		dis(i,1) -= d[1] * l;
		dis(j,1) += d[1] * l;
		if(dim==3){
		    dis(i,2) -= d[2] * l;
		    dis(j,2) += d[2] * l;
		}
	    }
	}
    }
    return;
}

double avr_dist_satis(indptr, indices, quality, N, _points, dim, is_uni)
    int *indptr, *indices, N, dim, is_uni;
    double *quality, *_points;
{
    double s = 0;

#pragma omp parallel for reduction(+: s)
    for(int i=0; i<N; i++){
	if(is_uni)
	    s -= quality[0] * (indptr[i+1]-indptr[i]) / 2;
	else
	    s -= quality[i] * (indptr[i+1]-indptr[i]) / 2;
	for(int _j=indptr[i]; _j<indptr[i+1]; _j++){
	    int j = indices[_j];
	    if (i<j){
		double d[3];
		d[0] = points(i,0)-points(j,0);
		d[1] = points(i,1)-points(j,1);
		if(dim==3)
		    d[2] = points(i,2)-points(j,2);
		else
		    d[2] = 0;
		s += sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
	    }
	}
    }
    return s;
}
