
#include "moveModel.cuh"
#include "matrixMul.cuh"

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
void cudaModeTransform(GLuint modelVBO, int vertexNumBytes, glm::mat4 m)
{

	// ×¢²áºÍ°ó¶¨cuda resource
	cudaGraphicsResource  *cuda_vbo_resource;

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, modelVBO, cudaGraphicsMapFlagsNone));



	float *d_vbo_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t vbo_num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_vbo_ptr, &vbo_num_bytes,
		cuda_vbo_resource));
	int vNum = vbo_num_bytes / vertexNumBytes;

	

	// set mat
	checkCudaErrors(cudaMemcpyToSymbol(devConst_modelMat, (float*)&(m[0][0]), sizeof(float) * 16, 0, cudaMemcpyHostToDevice));
	dim3 thread(256, 1);
	dim3 grid(65536, 1);
	grid.x = min(grid.x, vNum);

	transform << <grid, thread >> >(d_vbo_ptr, vertexNumBytes / sizeof(float), vNum, grid.x * thread.x);
	
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, NULL));
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));


}