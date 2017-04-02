

#include <glm\glm.hpp>

__constant__ float devConst_modelMat[16];
__host__ __device__ float4 mul(float matrix4x4[16], float4 tVector)
{
	float4 result;
	result.x = matrix4x4[0] * tVector.x + matrix4x4[4] * tVector.y + matrix4x4[8] * tVector.z + matrix4x4[12] * tVector.w;
	result.y = matrix4x4[1] * tVector.x + matrix4x4[5] * tVector.y + matrix4x4[9] * tVector.z + matrix4x4[13] * tVector.w;
	result.z = matrix4x4[2] * tVector.x + matrix4x4[6] * tVector.y + matrix4x4[10] * tVector.z + matrix4x4[14] * tVector.w;
	result.w = matrix4x4[3] * tVector.x + matrix4x4[7] * tVector.y + matrix4x4[11] * tVector.z + matrix4x4[15] * tVector.w;
	return result;
}

__global__ void transform(float* vbo, int vertexSizeFloat, int vNum, int threadNum)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	while (tid < vNum)
	{
		float4 pos = *(float4*)(&vbo[tid* vertexSizeFloat]);
		pos.w = 1.0f;
		pos = mul(devConst_modelMat, pos);
		vbo[tid*vertexSizeFloat] = pos.x;
		vbo[tid*vertexSizeFloat + 1] = pos.y;
		vbo[tid*vertexSizeFloat + 2] = pos.z;
		tid += threadNum;
	}
}
void cudaModeTransform(float m[16])
{
	
	FDscene[0].mapVBO();
	

	// set mat
	checkCudaErrors(cudaMemcpyToSymbol(devConst_modelMat, m, sizeof(float) * 16, 0, cudaMemcpyHostToDevice));
	dim3 thread(256, 1);
	dim3 grid(65536, 1);
	grid.x = grid.x<FDscene[0].vNum? grid.x: FDscene[0].vNum;

	transform << <grid, thread >> >(FDscene[0].vboPtr, FDscene[0].vertexSizeFloat, FDscene[0].vNum, grid.x * thread.x);
	
	cudaDeviceSynchronize();
	

	FDscene[0].unmapVBO();


}