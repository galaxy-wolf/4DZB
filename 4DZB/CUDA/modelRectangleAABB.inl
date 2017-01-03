
__global__ void verticesRectangleAABBcal_kernel()
{
	// output;
	float4* verticeRectangle = (float4*)c_fdStaticParams.verticesRectangleBuffer;
	float * aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;

	// input;
	float* vertex = (float*)c_fdStaticParams.vertexBuffer;
	int verticesNum = c_fdStaticParams.numVertices;
	int vertexSizeFloat = c_fdStaticParams.vertexSizeFloat;
	float * upRightMat = (float*)c_fdStaticParams.light.upRightMat;
	float2 rectangleFactor = (float2)c_fdStaticParams.light.RectangleFactor;



	// AABB
	float myXmin = FD_F32_MAX, myXmax = -FD_F32_MAX;
	float myYmin = FD_F32_MAX, myYmax = -FD_F32_MAX;
	float myZmin = FD_F32_MAX, myZmax = -FD_F32_MAX;

	// 
	uint vid = blockIdx.x * blockDim.x + threadIdx.x;

	if (vid < verticesNum)
	{
		float3 vPos = *(float3*)&(vertex[vid * vertexSizeFloat]);
		float3 rect;
		FD::makeRectangle(rect, upRightMat, vPos);

		myXmin = rect.x;
		myXmax = rect.x + rectangleFactor.x / rect.z;
		myYmin = rect.y;
		myYmax = rect.y + rectangleFactor.y / rect.z;
		myZmin = rect.z;
		myZmax = rect.z;

		// write out rectangle
		verticeRectangle[vid << 1 | 0] = make_float4(rect, 1.0f);
		verticeRectangle[vid << 1 | 1] = make_float4(myXmax, myYmax, myZmax, 1.0f);
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

	while (grid.x > 1)
	{
		int n = grid.x;
		grid.x = (iDiviUp(grid.x, block.x));
		AABBReduce_kernel<reduceMaxMinOnly> << <grid, block >> >(n);
	}
	// load to host;
	checkCudaErrors(cudaMemcpy((void *)&modelRectangleAABB[0], (void*)AABB3DReduceBuffer.devPtr, sizeof(float4) * 2, cudaMemcpyDeviceToHost));

	my_debug(MY_DEBUG_SECTION_AABB, 1)("model vertices aabb is: x: %f~%f y: %f~%f z: %f~%f\n", modelRectangleAABB[0].x, modelRectangleAABB[1].x,
		modelRectangleAABB[0].y, modelRectangleAABB[1].y, modelRectangleAABB[0].z, modelRectangleAABB[1].z);
}