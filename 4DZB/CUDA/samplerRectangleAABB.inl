
// 对于无效采样点，其rectangle 中Xmin > Xmax;
__global__ void sampleRectangleAABBcal_kernel()
{
	// output
	float * aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;
	int * valid = (int *)c_fdStaticParams.validTempBuffer;
	// input
	int viewportWidth = c_fdStaticParams.viewportWidth;
	int viewportHeight = c_fdStaticParams.viewportHeight;
	int viewWidth_LOG2 = c_fdStaticParams.viewportWidth_LOG2_UP;
	float * upRightMat = (float*)c_fdStaticParams.light.upRightMat;
	float2 rectangleFactor = (float2)c_fdStaticParams.light.RectangleFactor;
	cudaSurfaceObject_t sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;

	// AABB
	float myXmin = FD_F32_MAX, myXmax = -FD_F32_MAX;
	float myYmin = FD_F32_MAX, myYmax = -FD_F32_MAX;
	float myZmin = FD_F32_MAX, myZmax = -FD_F32_MAX;
	float myXsum = 0.0f, myYsum = 0.0f;

	uint sid = blockIdx.x * blockDim.x + threadIdx.x;
	uint sampleY = sid >> viewWidth_LOG2;
	uint sampleX = sid & ((1 << viewWidth_LOG2) - 1);

	if (sampleX < viewportWidth && sampleY < viewportHeight)
	{
		float4 samplePos = tex2D(samplePositionTex, sampleX, sampleY);
		float3 rect;
		if (!isnan(samplePos.x)) // 剔除屏幕上的无效采样点;
		{
			// set valid;
			valid[sampleX * viewportWidth + sampleY] = 1;

			FD::makeRectangle(rect, upRightMat, make_float3(samplePos));

			myXmin = rect.x;
			myXmax = rect.x + rectangleFactor.x / rect.z;
			myYmin = rect.y;
			myYmax = rect.y + rectangleFactor.y / rect.z;
			myZmin = rect.z;
			myZmax = rect.z;

			myXsum = rectangleFactor.x / rect.z;
			myYsum = rectangleFactor.y / rect.z;
		}
		surf2DLayeredwrite(make_float4(myXmin, myYmin, myZmin, 1.0f), sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 0, cudaBoundaryModeTrap);
		surf2DLayeredwrite(make_float4(myXmax, myYmax, myZmax, 1.0f), sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 1, cudaBoundaryModeTrap);
		
	}
	// reduce AABB;
	int opos = blockIdx.x;
	block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x00], myXmin);
	block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x01], myYmin);
	block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x02], myZmin);
	block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x03], myXsum);
	block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x04], myXmax);
	block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x05], myYmax);
	block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x06], myZmax);
	block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x07], myYsum);
}

void sampleRectangleAABBcal()
{
	validBuffer.clear(0);

	int threadNum = m_viewHeight << m_viewWidth_LOG2UP;
	dim3 block(REDUCE_BLOCK_SIZE, 1, 1);
	dim3 grid(iDiviUp(threadNum, block.x), 1, 1);
	sampleRectangleAABBcal_kernel << <grid, block >> >();
	getLastCudaError("after sample cal");

	while (grid.x > 1)
	{
		int n = grid.x;
		grid.x = (iDiviUp(grid.x, block.x));
		AABBReduce_kernel<reduceBoth> << <grid, block >> >(n);	
	}
	cudaDeviceSynchronize();
	getLastCudaError("aabb reduce");
	// load to host;
	checkCudaErrors(cudaMemcpy((void *)&sampleRectangleAABB[0], AABB3DReduceBuffer.devPtr, sizeof(float)*8, cudaMemcpyDeviceToHost));

	my_debug(MY_DEBUG_SECTION_AABB, 1)("sample AABB is\n x: %f~%f\ny: %f~%f\nz: %f~%f\n", sampleRectangleAABB[0].x, sampleRectangleAABB[1].x,
		sampleRectangleAABB[0].y, sampleRectangleAABB[1].y, sampleRectangleAABB[0].z, sampleRectangleAABB[1].z);

	// get valid sample num;
	int sampleNum = m_viewWidth * m_viewHeight;
	thrust::device_ptr<int> dev_valid((int *)validBuffer.devPtr);
	int validNum = thrust::reduce(dev_valid, dev_valid + sampleNum);

	// get avg;
	sampleRectangleAABB[0].w /= validNum;
	sampleRectangleAABB[1].w /= validNum;
	my_debug(MY_DEBUG_SECTION_SETUP, 1)("%d valid sample rectangle, agv is x: %f y %f\n", validNum, sampleRectangleAABB[0].w, sampleRectangleAABB[1].w);
}