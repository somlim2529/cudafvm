#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>   
#include <iostream>
#include "my_cuda_time.cu"
using namespace std;
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#define CELLSIZE 1
#define N 2
#define Nu 242
#define NT 500
#define NTPLOT 2
#define TYPE 3
#define DT 0.25
#define DT2 DT/CELLSIZE
#define NUMBLOCK 1
#define MAX(a,b) ((a)>(b) ? a:b)
#define MIN(a,b) ((a)<(b) ? a:b)
#define TX 12
#define TY 12
#define BX N/TX
#define BY N/TY
#define BLOCK_WIDTH 16
#define TOTAL_BLOCK 4
const int BLOCKS = 4;
const int THREADS = 121; 
// C = A + B
__device__ void plusMatrix(float A[TYPE],float B[TYPE],float C[TYPE]){
	for(int i=0;i<TYPE;i++){
		C[i]=A[i]+B[i];
	
	}
		}
// C = A - B
__device__ void minusMatrix(float A[TYPE],float B[TYPE],float C[TYPE]){
	for(int i=0;i<TYPE;i++){
		C[i]=(A[i]-B[i])*0.5 ;
	
	}	
}
//C = A * B
__device__ void multiplyMatrixI(float A[TYPE][TYPE],float B[TYPE][TYPE],float C[TYPE][TYPE]){	

	for (int i = 0; i < 3; i++)
		{ 
			  for (int j = 0; j < 3; j++)
				    {
						C[i][j] = 0;
						for (int k = 0; k < 3; k++)
							{
								C[i][j] += A[i][k] * B[k][j];
								}
						}
		  }
    

	}


__device__ void multiplyMatrixII(float A[TYPE],float B[TYPE][TYPE],float C[TYPE]){
	
	C[0]=B[0][0]*A[0]+B[0][1]*A[1]+B[0][2]*A[2];
	C[1]=B[1][0]*A[0]+B[1][1]*A[1]+B[1][2]*A[2];
	C[2]=B[2][0]*A[0]+B[2][1]*A[1]+B[2][2]*A[2];
	
}
	
__device__ void diagonal(float a1,float a2,float a3,float A[TYPE][TYPE]){
	A[0][0]=a1;
	A[0][1]=0.0;
	A[0][2]=0.0;
	A[1][0]=0.0;
	A[1][1]=a2;
	A[1][2]=0.0;
	A[2][0]=0.0;
	A[2][1]=0.0;
	A[2][2]=a3;
	}
	
__device__ void solver(float hl,float hr,float ul,float ur,float vl,float vr,float sn,float cn,float ANS[TYPE]){
	float grav=9.806;
	float half=0.5;
    //Compute Roe averages	
    float duml=pow(hl,half);
    float dumr=pow(hr,half);
	float cl=pow((grav*hl),half);
    float cr=pow((grav*hr),half);   
    float hhat=duml*dumr;
	float uhat=(duml*ul+dumr*ur)/(duml+dumr);
    float vhat=(duml*vl+dumr*vr)/(duml+dumr);
    float chat=pow((half*grav*(hl+hr)),half);
    float uperp=uhat*cn+vhat*sn;
	
    float dh=hr-hl;
    float du=ur-ul;
    float dv=vr-vl;
    float dupar=-du*sn+dv*cn;
    float duperp=du*cn+dv*sn;
  	float sumMatrix[TYPE],mulMatrixD[TYPE],dW[TYPE],FL[TYPE],FR[TYPE];
  	dW[0]=0.5*(dh-hhat*duperp/chat);
	dW[1]=hhat*dupar;
	dW[2]=0.5*(dh+hhat*duperp/chat);
	float uperpl=ul*cn+vl*sn;
    float uperpr=ur*cn+vr*sn;
    float al1=uperpl-cl;
    float al3=uperpl+cl;
    float ar1=uperpr-cr;
    float ar3=uperpr+cr;
	float R[TYPE][TYPE],mulMatrix[TYPE][TYPE],A[TYPE][TYPE];
	R[0][0]=1.0;
	R[0][1]=0.0;
	R[0][2]=1.0;
	R[1][0]=uhat-chat*cn;
	R[1][1]=-sn;
	R[1][2]=uhat+chat*cn;
	R[2][0]=vhat-chat*sn;
	R[2][1]=cn;
	R[2][2]=vhat+chat*sn;	
	float da1=MAX(0,2*(ar1-al1));
    float da3=MAX(0,2*(ar3-al3));
	float a1=abs(uperp-chat);
    float a2=abs(uperp);
    float a3=abs(uperp+chat);
    //Critical flow fix
	if (a1<da1){
        a1=0.5*(a1*a1/da1+da1);
	}
	else if (a3<da3){
        a3=0.5*(a3*a3/da3+da3);
	}
    //Compute interface flux
	diagonal(a1,a2,a3,A);
   	FL[0]=uperpl*hl;
	FL[1]=ul*uperpl*hl+0.5*grav*hl*hl*cn;
	FL[2]=vl*uperpl*hl+0.5*grav*hl*hl*sn;
 	FR[0]=uperpr*hr;
	FR[1]=ur*uperpr*hr+0.5*grav*hr*hr*cn;
	FR[2]=vr*uperpr*hr+0.5*grav*hr*hr*sn;	
    plusMatrix(FL,FR,sumMatrix);
	multiplyMatrixI(R,A,mulMatrix);
	multiplyMatrixII(dW,mulMatrix,mulMatrixD);
	minusMatrix(sumMatrix,mulMatrixD,ANS);
	}

__global__ void fluxes(float *Uh,float *Uhu,float *Uhv,float *Fh,float *Fhu,float *Fhv,float *Gh,float *Ghu,float *Ghv){
	float sn,cn,hu,hd,uu,ud,vu,vd,hl,hr,ul,ur,vl,vr,ANS[TYPE];
	int i=blockIdx.y;
	int j=blockDim.x*blockIdx.x+threadIdx.x;
	int index=i*Nu+j;
	if(i<Nu-1){
		//Compute fluxes in x direction	
		hu=Uh[i*Nu+j];
		hd=Uh[(i+1)*Nu+j];		
		uu=Uhu[i*Nu+j]/hu;
		ud=Uhu[(i+1)*Nu+j]/hd;
		vu=Uhv[i*Nu+j]/hu;
		vd=Uhv[(i+1)*Nu+j]/hd;
		sn=0.0;
		cn=1.0;
		solver(hu, hd, uu, ud, vu, vd, sn, cn,ANS);
		Fh[i*Nu+j]=ANS[0];
		Fhu[i*Nu+j]=ANS[1];
		Fhv[i*Nu+j]=ANS[2];
		if(j<Nu-1){
			//Compute fluxes in y direction	   
			hl=Uh[i*Nu+j];
			hr=Uh[i*Nu+(j+1)];
			ul=Uhu[i*Nu+j]/hl;
			ur=Uhu[i*Nu+(j+1)]/hr;
			vl=Uhv[i*Nu+j]/hl;
			vr=Uhv[i*Nu+(j+1)]/hr;
			sn=1.0;
			cn=0.0;
			solver(hl, hr, ul, ur, vl, vr, sn, cn,ANS);
			Gh[i*Nu+j]=ANS[0];
			Ghu[i*Nu+j]=ANS[1];
			Ghv[i*Nu+j]=ANS[2];
	
	}
		}
	
	}

__global__ void corrector(float *Uh,float *Uhu,float *Uhv,float *Fh,float *Fhu,float *Fhv,float *Gh,float *Ghu,float *Ghv){
	int i=blockIdx.y;
	int j=blockDim.x*blockIdx.x+threadIdx.x;
	int index=i*Nu+j;
	int tid_u=i*Nu+j;
	int tid_fd=i*Nu+j;
	int tid_fu=(i-1)*Nu+j;
	int tid_gr=i*Nu+j;
	int tid_gl=i*Nu+(j-1);

	if(i>0&&i<Nu-1&&j>0&&j<Nu-1){		
		Uh[tid_u]=Uh[tid_u]-DT2*(Fh[tid_fd]-Fh[tid_fu]+Gh[tid_gr]-Gh[tid_gl]);
		Uhu[tid_u]=Uhu[tid_u]-DT2*(Fhu[tid_fd]-Fhu[tid_fu]+Ghu[tid_gr]-Ghu[tid_gl]);
		Uhv[tid_u]=Uhv[tid_u]-DT2*(Fhv[tid_fd]-Fhv[tid_fu]+Ghv[tid_gr]-Ghv[tid_gl]);
		__syncthreads();

		}
	// copy Boundary
	Uh[j]=Uh[Nu+j];
	Uhu[j]=Uhu[Nu+j];
	Uhv[j]=Uhv[Nu+j];

	Uh[(Nu-1)*Nu+j]=Uh[(Nu-2)*Nu+j];
	Uhu[(Nu-1)*Nu+j]=Uhu[(Nu-2)*Nu+j];
	Uhv[(Nu-1)*Nu+j]=Uhv[(Nu-2)*Nu+j];

	Uh[i*Nu]=Uh[i*Nu+1];
	Uhu[i*Nu]=Uhu[i*Nu+1];
	Uhv[i*Nu]=-Uhv[i*Nu+1];	
	
	Uh[i*Nu+(Nu-1)]=Uh[i*Nu+(Nu-2)];
	Uhu[i*Nu+(Nu-1)]=Uhu[i*Nu+(Nu-2)];
	Uhv[i*Nu+(Nu-1)]=-Uhv[i*Nu+(Nu-2)];	
	
}

int main(){
	//time
	MyCudaTime cudaTime;
					
    //initial condition	h=high
	float h[Nu*Nu],u[Nu*Nu],v[Nu*Nu];
	for(int i=0;i<Nu;i++){
		for(int j=0;j<Nu;j++){
			int offset=i*Nu+j;
			if(i<(Nu/2)){
				h[offset]=1.0;}	
			else{
				h[offset]=0.6;
			}
			u[offset]=0.0;
			v[offset]=0.0;
			}

    }
	//CPU Memory Allocation
	float* Fh=(float*)calloc((Nu-1)*Nu,sizeof(float));
	float* Fhu=(float*)calloc((Nu-1)*Nu,sizeof(float));
	float* Fhv=(float*)calloc((Nu-1)*Nu,sizeof(float));
	float* Gh=(float*)calloc((Nu-1)*Nu,sizeof(float));
	float* Ghu=(float*)calloc((Nu-1)*Nu,sizeof(float));
	float* Ghv=(float*)calloc((Nu-1)*Nu,sizeof(float));
	float* Uh=(float*)calloc(Nu*Nu,sizeof(float));
	float* Uhu=(float*)calloc(Nu*Nu,sizeof(float));
	float* Uhv=(float*)calloc(Nu*Nu,sizeof(float));
	//GPU Memory Allocation
	float *Fh_dev,*Fhu_dev,*Fhv_dev,*Gh_dev,*Ghu_dev,*Ghv_dev,*Uh_dev,*Uhu_dev,*Uhv_dev;
	
	cudaMalloc((void**)&Fh_dev,(Nu-1)*Nu*sizeof(float));
	cudaMalloc((void**)&Fhu_dev,(Nu-1)*Nu*sizeof(float));
	cudaMalloc((void**)&Fhv_dev,(Nu-1)*Nu*sizeof(float));
	cudaMalloc((void**)&Gh_dev,(Nu-1)*Nu*sizeof(float));
	cudaMalloc((void**)&Ghu_dev,(Nu-1)*Nu*sizeof(float));
	cudaMalloc((void**)&Ghv_dev,(Nu-1)*Nu*sizeof(float));
	cudaMalloc((void**)&Uh_dev,Nu*Nu*sizeof(float));
	cudaMalloc((void**)&Uhu_dev,Nu*Nu*sizeof(float));
	cudaMalloc((void**)&Uhv_dev,Nu*Nu*sizeof(float));
	//intial U =h,u,v
	for(int j=0;j<Nu;j++){
		for(int k=0;k<Nu;k++){
			Uh[(j*Nu)+k]=h[j*Nu+k];
			Uhu[(j*Nu)+k]=h[j*Nu+k]*u[j*Nu+k];
            Uhv[(j*Nu)+k]=h[j*Nu+k]*v[j*Nu+k];
			}
	}
	//copy data to GPU
	cudaMemcpy(Fh_dev,Fh,(Nu-1)*Nu*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(Fhu_dev,Fhu,(Nu-1)*Nu*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(Fhv_dev,Fhv,(Nu-1)*Nu*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(Gh_dev,Gh,Nu*Nu*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(Ghu_dev,Ghu,Nu*Nu*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(Ghv_dev,Ghv,Nu*Nu*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(Uh_dev,Uh,Nu*Nu*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(Uhu_dev,Uhu,Nu*Nu*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(Uhv_dev,Uhv,Nu*Nu*sizeof(float),cudaMemcpyHostToDevice);
	dim3 dimGrid(Nu/THREADS,Nu);
	dim3 dimBlock(THREADS,1);
	cudaTime.beforeKernel();
	for(int tstep=0;tstep<NT;tstep++){
		fluxes<<<dimGrid,dimBlock>>>(Uh_dev,Uhu_dev,Uhv_dev,Fh_dev,Fhu_dev,Fhv_dev,Gh_dev,Ghu_dev,Ghv_dev);
		corrector<<<dimGrid,dimBlock>>>(Uh_dev,Uhu_dev,Uhv_dev,Fh_dev,Fhu_dev,Fhv_dev,Gh_dev,Ghu_dev,Ghv_dev);
		}
	
	cudaMemcpy(Uh,Uh_dev,Nu*Nu*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(Fh,Fh_dev,(Nu-1)*Nu*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(Gh,Gh_dev,(Nu-1)*Nu*sizeof(float),cudaMemcpyDeviceToHost);
	
	cudaTime.afterKernel();
	cudaTime.stop();
    cudaTime.report();
	//free memory
	free(Fh);free(Gh);free(Uh);free(Fhu);free(Ghu);free(Uhu);free(Fhv);free(Ghv);free(Uhv);
	cudaFree(Fh_dev);cudaFree(Gh_dev);cudaFree(Uh_dev);cudaFree(Fhu_dev);cudaFree(Ghu_dev);cudaFree(Uhu_dev);cudaFree(Fhv_dev);cudaFree(Ghv_dev);cudaFree(Uhv_dev);	
	return 0;
}