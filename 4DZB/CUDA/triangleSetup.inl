#include <helper_math.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
/**************************************
 setup triangle and computer AABB for each triangle
***************************************/
__device__ __forceinline__ void loadAndStoreVertex(int vid, int tid, int vid_in_tri, float4& triAABBmin, float4& triAABBmax)
{
	// input;
	float* vertex = (float*)c_fdStaticParams.vertexBuffer;
	int verticesNum = c_fdStaticParams.numVertices;
	int vertexSizeFloat = c_fdStaticParams.vertexSizeFloat;

	//
	const cudaTextureObject_t verticesRectangleTex = c_fdStaticParams.verticesRectangleTex;
	// output;
	float4* triPositionBuffer = (float4*)c_fdStaticParams.triPositionBuffer;
	

	float4 pos;
	float4 rectangle;
	pos = *(float4*)&vertex[vid * vertexSizeFloat];
	triPositionBuffer[tid * 3 + vid_in_tri] = pos;

	rectangle = tex1Dfetch<float4>(verticesRectangleTex, vid << 1);
	triAABBmin.x = fminf(triAABBmin.x, rectangle.x);
	triAABBmin.y = fminf(triAABBmin.y, rectangle.y);
	triAABBmin.z = fminf(triAABBmin.z, rectangle.z);

	rectangle = tex1Dfetch<float4>(verticesRectangleTex, vid << 1 | 1);
	triAABBmax.x = fmaxf(triAABBmax.x, rectangle.x);
	triAABBmax.y = fmaxf(triAABBmax.y, rectangle.y);
	triAABBmax.z = fmaxf(triAABBmax.z, rectangle.z);
}

__global__ void triangleVertexSetup_kernel()
{
	// input;
	int * indices = (int *)c_fdStaticParams.indexBuffer;
	int triNum = c_fdStaticParams.numTris;
	float3 lightPlaneB = c_fdLightPlaneParams.begin;
	float3 lightPlaneE = c_fdLightPlaneParams.end;

	// output;
	float4* triAABBBuffer = (float4*)c_fdStaticParams.triAABBBuffer;
	float * aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;
	int * validTempBuffer = (int *)c_fdStaticParams.validTempBuffer;

	// for reduce tri AABB size
	float myXsum = 0.0f, myYsum = 0.0f;

	uint tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < triNum)
	{
		int vid;
		float4 triAABBmin = make_float4(FD_F32_MAX, FD_F32_MAX, FD_F32_MAX, 1.0f), triAABBmax = make_float4(-FD_F32_MAX, -FD_F32_MAX, -FD_F32_MAX, 1.0f);
		int * vPtr = indices + tid * 3;
		
		vid = vPtr[0];
		loadAndStoreVertex(vid, tid, 0, triAABBmin, triAABBmax);

		vid = vPtr[1];
		loadAndStoreVertex(vid, tid, 1, triAABBmin, triAABBmax);

		vid = vPtr[2];
		loadAndStoreVertex(vid, tid, 2, triAABBmin, triAABBmax);

		// store triAABB
		triAABBBuffer[tid << 1] = triAABBmin;
		triAABBBuffer[tid << 1 | 1] = triAABBmax;

		float2 triB, triE;
		triB.x = fmaxf(triAABBmin.x, lightPlaneB.x);
		triB.y = fmaxf(triAABBmin.y, lightPlaneB.y);
		triE.x = fminf(triAABBmax.x, lightPlaneE.x);
		triE.y = fminf(triAABBmax.y, lightPlaneE.y);
		if (triB.x > triE.x || triB.y > triE.y) // 三角形位于lightPlane 外部
		{
			myXsum = 0;
			myYsum = 0;
		}
		else
		{
			myXsum = triE.x - triB.x;
			myYsum = triE.y - triB.y;
			validTempBuffer[tid] = 1;
		}
		
	}
	int opos = blockIdx.x;
	block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 1) | 0x00], myXsum);
	block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 1) | 0x01], myYsum);
	
}

void setupTriangleVertex()
{
	// clear valid buffer
	validBuffer.clear(0);

	dim3 block(REDUCE_BLOCK_SIZE, 1, 1);
	dim3 grid(iDiviUp(m_triangleNum, block.x));
	triangleVertexSetup_kernel << <grid, block >> >();

	while (grid.x > 1)
	{
		int n = grid.x;
		grid.x = (iDiviUp(grid.x, block.x));
		AABBReduce_kernel<reduceXYsumOnly> << <grid, block >> >(n);
	}

	// 在light plane 中的三角形个数。
	thrust::device_ptr<int> dev_valid((int *)validBuffer.devPtr);
	int validTriangleNum = thrust::reduce(dev_valid, dev_valid + m_maxTriangleNum);
	float h_temp[2];
	// load to host;
	checkCudaErrors(cudaMemcpy((void *)&h_temp[0], (void*)AABB3DReduceBuffer.devPtr, sizeof(float) * 2, cudaMemcpyDeviceToHost));
	modelRectangleAABB[0].w = h_temp[0] / validTriangleNum;
	modelRectangleAABB[1].w = h_temp[1] / validTriangleNum;

	my_debug(MY_DEBUG_SECTION_SETUP, 1)("valid tri num is %d triangle aabb avg size is: x: %f y:%f\n", validTriangleNum, modelRectangleAABB[0].w, modelRectangleAABB[1].w);
}