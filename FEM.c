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
	indices_fem,indptr_fem[a[i]], indptr_fem[a[i]+1], a[j])


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
	int *indices_fem, // min size = sizeof(indices)+N_interior
	int *indptr_fem,  // min size = N_interior+1
	int *simplices,
	int *boundary,
	int N_points,      // N_interior = N_points-N_excluded-N_convexh
	int N_simplices,
	double *points,
	double *nabla,
	double *inner,
	double *surface,
	double *V6s)
{
    int * indices_fem_cur = indices_fem;
    int * indptr_fem_cur = indptr_fem;
    *(indptr_fem_cur++) = 0;
    for(int i=0; i<N_points; i++){
	*(indices_fem_cur++) = i;
	for(int _j=indptr[i]; _j<indptr[i+1]; _j++)
	    *(indices_fem_cur++) = indices[_j];
	sort(indices_fem+*(indptr_fem_cur-1), indices_fem_cur-1);
	*(indptr_fem_cur++) = indices_fem_cur - indices_fem;
    }
#pragma omp parallel for
    for(int is=0; is<N_simplices; is++){
	int a[4];
	double *p[4], q[2][3];
	a[0] = simplices[4*is+0];
	a[1] = simplices[4*is+1];
	a[2] = simplices[4*is+2];
	a[3] = simplices[4*is+3];
	p[0] = points + 3*a[0];
	p[1] = points + 3*a[1];
	p[2] = points + 3*a[2];
	p[3] = points + 3*a[3];
	cross(p[1],p[2],q[0],p[0]);
	q[1][0] = p[3][0]-p[0][0];
	q[1][1] = p[3][1]-p[0][1];
	q[1][2] = p[3][2]-p[0][2];
	double V6 = fabs(inner(q[0],q[1]));
	V6s[is] = V6;
	for(int i=0; i<4; i++){
	    int l = (i+1)%4;
	    int m = (i+2)%4;
	    int n = (i+3)%4;
	    cross(p[l],p[m],q[0],p[n]);
	    int _idx = idx(i,i);
	    double val = .5 * inner(q[0],q[0]) / (6*V6);
#pragma omp atomic update
	    nabla[_idx] += val;
	    val = V6 / 120;
#pragma omp atomic update
	    inner[_idx] += val;
	    for(int j=0; j<i; j++){
		l = (i+1)%4;
		m = (i+2)%4;
		if     (l==j) l = (i+3)%4;
		else if(m==j) m = (i+3)%4;
		cross(p[i],p[l],q[0],p[m]);
		cross(p[j],p[l],q[1],p[m]);
		_idx = idx(i,j);
		val = - inner(q[0],q[1]) / (6*V6);
#pragma omp atomic update
		nabla[_idx] += val;
		val = V6 / 120;
#pragma omp atomic update
		inner[_idx] += val;
		if(boundary[a[i]] && boundary[a[j]]){
		    for(int k=0; k<j; k++){
			if(boundary[a[k]]){
			    cross(p[i],p[j],q[0],p[k]);
			    val = sqrt(inner(q[0],q[0])) / 24;
			    _idx = idx(i,i);
#pragma omp atomic update
			    surface[_idx] += val;
			    _idx = idx(j,j);
#pragma omp atomic update
			    surface[_idx] += val;
			    _idx = idx(k,k);
#pragma omp atomic update
			    surface[_idx] += val;
			    _idx = idx(i,j);
#pragma omp atomic update
			    surface[_idx] += val;
			    _idx = idx(i,k);
#pragma omp atomic update
			    surface[_idx] += val;
			    _idx = idx(j,k);
#pragma omp atomic update
			    surface[_idx] += val;
			}
		    }
		}
	    }
	}
    }
}

void diag(
	int *indices_fem,
	int *indptr_fem,
	int *simplices,
	int N_simplices,
	double *V6s,
	double *f,
	double *data)
{
    for(int i=0; i<10; i++)
	printf("%lf ", f[i]);
    fflush(stdout);
#pragma omp parallel for
    for(int is=0; is<N_simplices; is++){
	double V6 = V6s[is];
	int _idx;
	double val;
	int a[4];
	a[0] = simplices[4*is+0];
	a[1] = simplices[4*is+1];
	a[2] = simplices[4*is+2];
	a[3] = simplices[4*is+3];
	for(int i=0; i<4; i++){
	    _idx = idx(i,i);
	    val  = 0;
	    for(int k=0; k<4; k++){
		if(k==i) val += f[a[k]] * V6 / 240;
		else     val += f[a[k]] * V6 / 720;
	    }
#pragma omp atomic update
	    data[_idx] += val;
	    for(int j=0; j<i; j++){
		_idx = idx(i,j);
		val  = 0;
		for(int k=0; k<4; k++){
		    if((k==i)||(k==j)) val += f[a[k]] * V6 / 360;
		    else               val += f[a[k]] * V6 / 720;
		}
#pragma omp atomic update
		data[_idx] += val;
	    }
	}
    }
}


