#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define NR 5
#define  INF 99999

using namespace std;

int main()
{
	int *d_matrix;
	
	int h_matrixGraph[][] = {
		{ 0, 2, INF, 10, INF },
		{ 2, 0, 3, INF, INF },
		{ INF, 3, 0, 1, 8 },
		{ 10, INF, 1, 0, INF },
		{ INF, INF, 8, INF, 0 }
	};

	cudaMalloc(&d_matrix, NR*NR * sizeof(int)); //alocam memorie liniara

	for (int h_k = 1; h_k <= NR; h_k++)
	{
		int* d_k;

		cudaMemcpy(d_matrix, h_matrixGraph, NR * NR * sizeof(int), cudaMemcpyHostToDevice); //transferam memoria din host in device
		cudaMalloc(&d_k, sizeof(int)); //alocare memorie
		cudaMemcpy(d_k, &h_k, sizeof(int), cudaMemcpyHostToDevice); //transferam din host in device
		
		int numBlocks = 1;

		dim3 threadsPerBlock(NR, NR); //invocare kernel creem dimensiunea, mai exact doua dimensiuni pentru matrice
		RoyFloyd<<<numBlocks, threadsPerBlock>>>(d_matrix);
		cudaMemcpy(h_matrixGraph, d_matrix, NR * NR * sizeof(int), cudaMemcpyDeviceToHost);
	}

	for (int i = 1; i <= NR; i++) {
		for (int j = 1; j <= NR; j++)
		{
			if (h_matrixGraph[i][j] == INF)
				printf("-, ");
			else
				printf("%d, ", h_matrixGraph[i][j]);
		}
		printf("\n");
	}

	cudaFree(h_matrixGraph); //eliberam memoria
	cudaFree(d_matrix);
	system("pause");
	return 0;

}

__global__ void RoyFloyd(int matrixGraph[NR][NR])
{

	int i = threadIdx.x;
	int j = threadIdx.y;
	for (int k = 1; k <= NR; k++)
	{
		if (matrixGraph[i][k] + matrixGraph[k][j] < matrixGraph[i][j]) {
			matrixGraph[i][j] = matrixGraph[i][k] + matrixGraph[k][j];
		}
	}
}