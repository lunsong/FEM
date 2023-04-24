#include <stdio.h>
#include <omp.h>
#include <math.h>

int bisect(int *arr, int i, int j, int x){
    if(arr[i]==x) return i;
    while(j-i>1){
	int k = (i+j)/2;
	if(arr[k] > x)
	    j = k;
	else
	    if(arr[k] < x) 
		i = k;
	    else
		return k;
    }
    return -1;
}

#define idx(i,j) bisect(\
    indices_int,indptr_int[i##_int], indptr_int[i##_int+1], j##_int)


void sort(int *a, int *b){
    int *_a = a;
    int *_b = b;
    if(b-a<=0) return;
    if(b-a==1){
	if(*a>*b){
	    int c = *a;
	    *a = *b;
	    *b = c;
	}
	return;
    }
    int pivot = *a;
    while(1){
	while(*b > pivot) b--;
	if(a==b) break;
	*(a++) = *b;
	*b = pivot;
	while(*a < pivot) a++;
	if(a==b) break;
	*(b--) = *a;
	*a = pivot;
    }
    *a = pivot;
    sort(_a, a-1);
    sort(a+1,_b);
}

void cross(a,b,c,r)
    double *a,*b,*c,*r;
{
    c[0] = ((a[1]-r[1])*(b[2]-r[2]) - (a[2]-r[2])*(b[1]-r[1]));
    c[1] = ((a[2]-r[2])*(b[0]-r[0]) - (a[0]-r[0])*(b[2]-r[2]));
    c[2] = ((a[0]-r[0])*(b[1]-r[1]) - (a[1]-r[1])*(b[0]-r[0]));
}

#define inner(a,b) (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])

void FEM(
	int *indices,
	int *indptr, 
	int *indices_int, // min size = sizeof(indices)+N_interior
	int *indptr_int,  // min size = N_interior+1
	int *convex_hull,
	int *to_interior,
	int *to_original,
	int *simplices,
	int N_points,	  // N_interior = N_points-N_excluded-N_convexh
	int N_convex_hull,
	int N_excluded,
	int N_simplices,
	int *N_indices_int,
	double *points,
	double *data){
    int N_interior = 0;
    int *convex_hull_end = convex_hull + N_convex_hull;
    for(int i=N_excluded; i<N_points; i++){
	if(convex_hull < convex_hull_end && i==*convex_hull){
	    convex_hull++;
	    continue;
	}
	to_interior[i] = N_interior;
	to_original[N_interior++] = i;
    }
    int N_indptr_int = 0;
    indptr_int[N_indptr_int++] = 0;
    for(int i_int=0; i_int<N_interior; i_int++){
	indices_int[(*N_indices_int)++] = i_int;
	int i = to_original[i_int];
	for(int _j=indptr[i]; _j<indptr[i+1]; _j++){
	    int j_int = to_interior[indices[_j]];
	    if(j_int!=-1)
		indices_int[(*N_indices_int)++] = j_int;
	}
	sort(indices_int+indptr_int[N_indptr_int-1],
	     indices_int+*N_indices_int-1);
	indptr_int[N_indptr_int++] = *N_indices_int;
    }
#pragma omp parallel for
    for(int is=0; is<N_simplices; is++){
	double *p[4], q[2][3];
	p[0] = points + 3*simplices[4*is+0];
	p[1] = points + 3*simplices[4*is+1];
	p[2] = points + 3*simplices[4*is+2];
	p[3] = points + 3*simplices[4*is+3];
	cross(p[1],p[2],q[0],p[0]);
	q[1][0] = p[3][0]-p[0][0];
	q[1][1] = p[3][1]-p[0][1];
	q[1][2] = p[3][2]-p[0][2];
	double V6 = fabs(inner(q[0],q[1]));
	for(int i=0; i<4; i++){
	    int i_int = to_interior[simplices[4*is+i]];
	    if(i_int==-1) continue;
	    int l = (i+1)%4;
	    int k = (i+2)%4;
	    int m = (i+3)%4;
	    cross(p[l],p[k],q[0],p[m]);
	    int _idx = idx(i,i);
	    double val = .5 * inner(q[0],q[0]) / (6*V6);
#pragma omp atomic
	    data[_idx] += val;
	    for(int j=0; j<i; j++){
		int j_int = to_interior[simplices[4*is+j]];
		if(j_int==-1) continue;
		int l = (i+1)%4;
		int k = (i+2)%4;
		if(l==j) l = (i+3)%4;
		else if(k==j) k = (i+3)%4;
		cross(p[i],p[k],q[0],p[l]);
		cross(p[j],p[k],q[1],p[l]);
		_idx = idx(i,j);
		val = - inner(q[0],q[1]) / (6*V6);
#pragma omp atomic
		data[_idx] += val;
	    }
	}
    }
}
