__global__ void triangleMaxZcal_kernel()
{
	// output;
	float * const aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;

	// input;
	const float* const vertex = (float*)c_fdStaticParams.vertexBuffer;
	const int verticesNum = c_fdStaticParams.numVertices;
	const int vertexSizeFloat = c_fdStaticParams.vertexSizeFloat;
	const float * const upRightMat = (float*)c_fdStaticParams.light.upRightMat;



	// AABB
	float myZmax = -FD_F32_MAX;

	// 
	const uint vid = blockIdx.x * blockDim.x + threadIdx.x;

	if (vid < verticesNum)
	{
		const float3 vPos = *(float3*)&(vertex[vid * vertexSizeFloat]);
		FD::getZ(myZmax, upRightMat, vPos);
	}


	// reduce AABB;
	int opos = blockIdx.x;
	block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[opos], myZmax);

}
int triangleMaxZcal(float &triangleMaxZ)
{
	dim3 block(REDUCE_BLOCK_SIZE, 1, 1);
	dim3 grid(iDiviUp(m_verteicesNum, block.x), 1, 1);
	triangleMaxZcal_kernel << <grid, block >> >();
	cudaThreadSynchronize();

	thrust::device_ptr<float> dev_maxZ((float*)AABB3DReduceBuffer.devPtr);
	triangleMaxZ = thrust::reduce(dev_maxZ, dev_maxZ + grid.x, -1.0f, thrust::maximum<float>());

	if (triangleMaxZ < 0.0f) // Ã»ÓÐÏñËØ
		return 1;
	else
		return 0;
}

__global__ void verticesRectangleAABBcal_kernel()
{
	// output;
	//	float4* const verticeRectangle = (float4*)c_fdStaticParams.verticesRectangleBuffer;
	float * const aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;

	// input;
	const float* const vertex = (float*)c_fdStaticParams.vertexBuffer;
	const int verticesNum = c_fdStaticParams.numVertices;
	const int vertexSizeFloat = c_fdStaticParams.vertexSizeFloat;
	const float * const upRightMat = (float*)c_fdStaticParams.light.upRightMat;
	const float2 rectangleFactor = (float2)c_fdStaticParams.light.RectangleFactor;
	const float2 rectangleSubConstant = (float2)c_fdRectangleSubConstant;



	// AABB
	float myXmin = FD_F32_MAX, myXmax = -FD_F32_MAX;
	float myYmin = FD_F32_MAX, myYmax = -FD_F32_MAX;
	float myZmin = FD_F32_MAX, myZmax = -FD_F32_MAX;

	// 
	uint vid = blockIdx.x * blockDim.x + threadIdx.x;

	if (vid < verticesNum)
	{
		const float3 vPos = *(float3*)&(vertex[vid * vertexSizeFloat]);
		float3 rect;
		FD::makeRectangle(rect, upRightMat, vPos);

		myXmin = rect.x;
		myXmax = rect.x + rectangleFactor.x / rect.z - rectangleSubConstant.x;
		myYmin = rect.y;
		myYmax = rect.y + rectangleFactor.y / rect.z - rectangleSubConstant.y;
		myZmin = rect.z;
		myZmax = rect.z;

		//// write out rectangle
		//verticeRectangle[vid << 1 | 0] = make_float4(rect, 1.0f);
		//verticeRectangle[vid << 1 | 1] = make_float4(myXmax, myYmax, myZmax, 1.0f);
	}

	// reduce AABB;
	int opos = blockIdx.x;
	block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x00], myXmin);
	block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x01], myYmin);
	block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x02], myZmin);

	block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x04], myXmax);
	block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x05], myYmax);
	block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x06], myZmax);
}

void verticesRectangleAABBcal()
{
	my_debug(MY_DEBUG_SECTION_AABB, 2)("vertices num is %d\n", m_verteicesNum);
	dim3 block(REDUCE_BLOCK_SIZE, 1, 1);
	dim3 grid(iDiviUp(m_verteicesNum, block.x), 1, 1);
	verticesRectangleAABBcal_kernel << <grid, block >> >();
	cudaThreadSynchronize();

	while (grid.x > 1)
	{
		int n = grid.x;
		grid.x = (iDiviUp(grid.x, block.x));
		AABBReduce_kernel<reduceMaxMinOnly> << <grid, block >> >(n);
		cudaThreadSynchronize();
	}


	// load to host;
	checkCudaErrors(cudaMemcpy((void *)&modelRectangleAABB[0], (void*)AABB3DReduceBuffer.devPtr, sizeof(float4) * 2, cudaMemcpyDeviceToHost));

	my_debug(MY_DEBUG_SECTION_AABB, 1)("model vertices aabb is: x: %f~%f y: %f~%f z: %f~%f\n", modelRectangleAABB[0].x, modelRectangleAABB[1].x,
		modelRectangleAABB[0].y, modelRectangleAABB[1].y, modelRectangleAABB[0].z, modelRectangleAABB[1].z);

}