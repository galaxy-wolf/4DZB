#include <helper_math.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
/**************************************
 setup triangle and computer AABB for each triangle
***************************************/
//__device__ __forceinline__ void loadAndStoreVertex(int vid, int tid, int vid_in_tri, float4& triAABBmin, float4& triAABBmax)
//{
//	// input;
//	const float* const vertex = (float*)c_fdStaticParams.vertexBuffer;
//	const int verticesNum = c_fdStaticParams.numVertices;
//	const int vertexSizeFloat = c_fdStaticParams.vertexSizeFloat;
//
//	//
//	const cudaTextureObject_t verticesRectangleTex = c_fdStaticParams.verticesRectangleTex;
//	const cudaTextureObject_t verticesTex = c_fdStaticParams.verticesTex;
//
//	// output;
//	float4* const triPositionBuffer = (float4*)c_fdStaticParams.triPositionBuffer;
//	
//
//	const float4 pos = tex1Dfetch<float4>(verticesTex, vid *(vertexSizeFloat>>2));//*(float4*)&vertex[vid * vertexSizeFloat];
//	float4 rectangle;
//	triPositionBuffer[tid * 3 + vid_in_tri] = pos;
//
//	rectangle = tex1Dfetch<float4>(verticesRectangleTex, vid << 1);
//	triAABBmin.x = fminf(triAABBmin.x, rectangle.x);
//	triAABBmin.y = fminf(triAABBmin.y, rectangle.y);
//	triAABBmin.z = fminf(triAABBmin.z, rectangle.z);
//
//	rectangle = tex1Dfetch<float4>(verticesRectangleTex, vid << 1 | 1);
//	triAABBmax.x = fmaxf(triAABBmax.x, rectangle.x);
//	triAABBmax.y = fmaxf(triAABBmax.y, rectangle.y);
//	triAABBmax.z = fmaxf(triAABBmax.z, rectangle.z);
//}

__device__ __forceinline__ void getTriAABB(float4& v, float4 &triAABBmin, float4& triAABBmax)
{
	const float * const upRightMat = (float*)c_fdStaticParams.light.upRightMat;

	const float2 rectangleFactor = (float2)c_fdStaticParams.light.RectangleFactor;
	const float2 rectangleSubConstant = (float2)c_fdRectangleSubConstant;

	float3 rect;
	FD::makeRectangle(rect, upRightMat, make_float3(v));
	triAABBmin.x = fminf(triAABBmin.x, rect.x);
	triAABBmin.y = fminf(triAABBmin.y, rect.y);
	triAABBmin.z = fminf(triAABBmin.z, rect.z);
	triAABBmax.z = fmaxf(triAABBmax.z, rect.z);

	triAABBmax.x = fmaxf(triAABBmax.x, rect.x + rectangleFactor.x / rect.z - rectangleSubConstant.x);
	triAABBmax.y = fmaxf(triAABBmax.y, rect.y + rectangleFactor.y / rect.z - rectangleSubConstant.y);

}

template<bool enableBackFaceCull>
__global__ void triangleVertexSetup_kernel()
{
	// input;
	const int3 * const indices = (int3 *)c_fdStaticParams.indexBuffer;
	//const int * const indices = (int *)c_fdStaticParams.indexBuffer;
	const int triNum = c_fdStaticParams.numTris;
	const float3 lightPlaneB = c_fdLightPlaneParams.begin;
	const float3 lightPlaneE = c_fdLightPlaneParams.end;

	const float3 upRightPos = c_fdStaticParams.light.upRightCornerPosition;
	const float3 downLeftPos = c_fdStaticParams.light.downLeftCornerPosition;
	const float3 upLeftPos = c_fdStaticParams.light.upLeftCornerPosition;
	const float3 downRightPos = c_fdStaticParams.light.downRightCornerPosition;

	const cudaTextureObject_t verticesTex = c_fdStaticParams.verticesTex;
	const int vertexSizeFloat4 = c_fdStaticParams.vertexSizeFloat / 4;

	// output;
	float4* const triAABBBuffer = (float4*)c_fdStaticParams.triAABBBuffer;
	TriangleData * const triDataBuffer = (TriangleData*)c_fdStaticParams.triDataBuffer;
	float * const aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;
	int * const validTempBuffer = (int *)c_fdStaticParams.validTempBuffer;

	// for reduce tri AABB size
	float myXsum = 0.0f, myYsum = 0.0f;

	const uint tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < triNum)
	{
		int3 vid;
		
		float4 v0, v1, v2;
		float4 triAABBmin = make_float4(FD_F32_MAX, FD_F32_MAX, FD_F32_MAX, 1.0f), triAABBmax = make_float4(-FD_F32_MAX, -FD_F32_MAX, -FD_F32_MAX, 1.0f);

		vid = indices[tid];
		v0 = tex1Dfetch<float4>(verticesTex, vid.x *(vertexSizeFloat4));
		v1 = tex1Dfetch<float4>(verticesTex, vid.y *(vertexSizeFloat4));
		v2 = tex1Dfetch<float4>(verticesTex, vid.z *(vertexSizeFloat4));

		float3 e0 = make_float3(v1) - make_float3(v0);
		float3 e1 = make_float3(v0) - make_float3(v2);
		float3 n = cross(e1, e0);

		if (enableBackFaceCull &&
			dot(n, upRightPos - make_float3(v0)) <= 0 &&
			dot(n, upRightPos - make_float3(v1)) <= 0 &&
			dot(n, upRightPos - make_float3(v2)) <= 0 &&
			dot(n, upLeftPos - make_float3(v0)) <= 0 &&
			dot(n, upLeftPos - make_float3(v1)) <= 0 &&
			dot(n, upLeftPos - make_float3(v2)) <= 0 &&
			dot(n, downRightPos - make_float3(v0)) <= 0 &&
			dot(n, downRightPos - make_float3(v1)) <= 0 &&
			dot(n, downRightPos - make_float3(v2)) <= 0 &&
			dot(n, downLeftPos - make_float3(v0)) <= 0 &&
			dot(n, downLeftPos - make_float3(v1)) <= 0 &&
			dot(n, downLeftPos - make_float3(v2)) <= 0
			)
		{
			// back face culling;
		}
		else{
			getTriAABB(v0, triAABBmin, triAABBmax);
			getTriAABB(v1, triAABBmin, triAABBmax);
			getTriAABB(v2, triAABBmin, triAABBmax);

			if (triAABBmin.x > lightPlaneE.x || triAABBmin.y > lightPlaneE.y || triAABBmax.x < lightPlaneB.x || triAABBmax.y < lightPlaneB.y)// AABB全部在lightPlane外部。
			{
			}
			else{
				// write triData;
				TriangleData triData;
				triData.v0 = make_float3(v0);
				triData.v1 = make_float3(v1);
				triData.v2 = make_float3(v2);
				triData.minz = triAABBmin.z;
				triData.maxz = triAABBmax.z;
				triData.aabb2Dmax = make_float2(triAABBmax);
				triData.aabb2Dmin = make_float2(triAABBmin);
				triDataBuffer[tid] = triData;	

				float2 triB, triE;
				triB.x = fmaxf(triAABBmin.x, lightPlaneB.x);
				triB.y = fmaxf(triAABBmin.y, lightPlaneB.y);
				triE.x = fminf(triAABBmax.x, lightPlaneE.x);
				triE.y = fminf(triAABBmax.y, lightPlaneE.y);

				myXsum = triE.x - triB.x;
				myYsum = triE.y - triB.y;
				validTempBuffer[tid] = 1;
			}
		}

		triAABBBuffer[tid << 1] = triAABBmin;
		triAABBBuffer[tid << 1 | 1] = triAABBmax;

		//int vid;
		//float4 triAABBmin = make_float4(FD_F32_MAX, FD_F32_MAX, FD_F32_MAX, 1.0f), triAABBmax = make_float4(-FD_F32_MAX, -FD_F32_MAX, -FD_F32_MAX, 1.0f);
		//const int * const vPtr = indices + tid * 3;
		//
		//vid = vPtr[0];
		//loadAndStoreVertex(vid, tid, 0, triAABBmin, triAABBmax);

		//vid = vPtr[1];
		//loadAndStoreVertex(vid, tid, 1, triAABBmin, triAABBmax);

		//vid = vPtr[2];
		//loadAndStoreVertex(vid, tid, 2, triAABBmin, triAABBmax);

		//// store triAABB
		//triAABBBuffer[tid << 1] = triAABBmin;
		//triAABBBuffer[tid << 1 | 1] = triAABBmax;

		//float2 triB, triE;
		//triB.x = fmaxf(triAABBmin.x, lightPlaneB.x);
		//triB.y = fmaxf(triAABBmin.y, lightPlaneB.y);
		//triE.x = fminf(triAABBmax.x, lightPlaneE.x);
		//triE.y = fminf(triAABBmax.y, lightPlaneE.y);
		//if (triB.x > triE.x || triB.y > triE.y) // 三角形位于lightPlane 外部
		//{
		//	myXsum = 0;
		//	myYsum = 0;
		//}
		//else
		//{
		//	myXsum = triE.x - triB.x;
		//	myYsum = triE.y - triB.y;
		//	validTempBuffer[tid] = 1;
		//}
		
	}
	int opos = blockIdx.x;
	block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 1) | 0x00], myXsum);
	block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 1) | 0x01], myYsum);
	
}

// Return : 
//			True: no Triangle;
//			False: ok;
bool setupTriangleVertex()
{
	// clear valid buffer
	validBuffer.clear(0);

	dim3 block(REDUCE_BLOCK_SIZE, 1, 1);
	dim3 grid(iDiviUp(m_triangleNum, block.x));
	if (m_enable_backfacecull)
		triangleVertexSetup_kernel<true> << <grid, block >> >();
	else
		triangleVertexSetup_kernel<false> << <grid, block >> >();
	cudaThreadSynchronize();

	while (grid.x > 1)
	{
		int n = grid.x;
		grid.x = (iDiviUp(grid.x, block.x));
		AABBReduce_kernel<reduceXYsumOnly> << <grid, block >> >(n);
		cudaThreadSynchronize();
	}

	// 在light plane 中的三角形个数。
	thrust::device_ptr<int> dev_valid((int *)validBuffer.devPtr);
	m_validTriangleNum = thrust::reduce(dev_valid, dev_valid + m_maxTriangleNum);
	float h_temp[2];
	// load to host;
	checkCudaErrors(cudaMemcpy((void *)&h_temp[0], (void*)AABB3DReduceBuffer.devPtr, sizeof(float) * 2, cudaMemcpyDeviceToHost));
	// no
	if (m_validTriangleNum == 0)
		return 1;
	modelRectangleAABB[0].w = h_temp[0] / m_validTriangleNum;
	modelRectangleAABB[1].w = h_temp[1] / m_validTriangleNum;

	my_debug(MY_DEBUG_SECTION_SETUP, 1)("valid tri num is %d triangle aabb avg size is: x: %f y: %f\n", m_validTriangleNum, modelRectangleAABB[0].w, modelRectangleAABB[1].w);
	return 0;
}
