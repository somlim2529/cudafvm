#include <stdio.h>

struct MyCudaTime {

	cudaEvent_t _start;
	cudaEvent_t _beforeKernel, _afterKernel, _stop;

	MyCudaTime() {
		cudaEventCreate(&_start);
		cudaEventCreate(&_beforeKernel);
		cudaEventCreate(&_afterKernel);
		cudaEventCreate(&_stop);

		cudaEventRecord(_start, 0);
	}

	void beforeKernel() {
		cudaEventRecord(_beforeKernel, 0);
	}
	
	void afterKernel() {
		cudaEventRecord(_afterKernel, 0);
	}

	void stop() {  // return elapsed time in milliseconds
		cudaEventRecord(_stop, 0);
		cudaEventSynchronize(_stop);
	}

	void report() {
		float elapsedTime;

		cudaEventElapsedTime(&elapsedTime, _start, _stop);
		printf("Total time %3.2f ms\n", elapsedTime);   // why 3.1?

		cudaEventElapsedTime(&elapsedTime, _start, _beforeKernel);
		printf("\t Before calling kernel %3.2f ms\n", elapsedTime); 

		cudaEventElapsedTime(&elapsedTime, _beforeKernel, _afterKernel);
		printf("\t In kernel %3.2f ms\n", elapsedTime); 

		cudaEventElapsedTime(&elapsedTime, _afterKernel, _stop);
		printf("\t After calling kernel %3.2f ms\n", elapsedTime); 
	}
};