__global__ void sampleMaxZcal_kernel()
{
	// output
	float * const aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;

	// input
	const int viewportWidth = c_fdStaticParams.viewportWidth;
	const int viewportHeight = c_fdStaticParams.viewportHeight;
	const int viewWidth_LOG2 = c_fdStaticParams.viewportWidth_LOG2_UP;
	const float * const upRightMat = (float*)c_fdStaticParams.light.upRightMat;


	// AABB
	float myZmax = -FD_F32_MAX;

	uint sid = blockIdx.x * blockDim.x + threadIdx.x;
	uint sampleY = sid >> viewWidth_LOG2;
	uint sampleX = sid & ((1 << viewWidth_LOG2) - 1);

	if (sampleX < viewportWidth && sampleY < viewportHeight)
	{
		float4 samplePos = tex2D(samplePositionTex, sampleX, sampleY);
		if (!isnan(samplePos.x)) // 剔除屏幕上的无效采样点;
		{
			FD::getZ(myZmax, upRightMat, make_float3(samplePos));
		}

	}
	// reduce AABB;
	int opos = blockIdx.x;
	block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[opos], myZmax);


}
int sampleMaxZcal(float & sampleMaxZ)
{
	int threadNum = m_viewHeight << m_viewWidth_LOG2UP;
	dim3 block(REDUCE_BLOCK_SIZE, 1, 1);
	dim3 grid(iDiviUp(threadNum, block.x), 1, 1);
	sampleMaxZcal_kernel << <grid, block >> >();
	cudaThreadSynchronize();
	thrust::device_ptr<float> dev_maxZ((float*)AABB3DReduceBuffer.devPtr);
	sampleMaxZ = thrust::reduce(dev_maxZ, dev_maxZ + grid.x, -1.0f, thrust::maximum<float>());

	if (sampleMaxZ < 0.0f) // 没有像素
		return 1;
	else
		return 0;

}

// 对于无效采样点，其rectangle 中Xmin > Xmax;
__global__ void sampleRectangleAABBcal_kernel()
{
	// output
	float * const aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;
	int * const valid = (int *)c_fdStaticParams.validTempBuffer;
	// input
	const int viewportWidth = c_fdStaticParams.viewportWidth;
	const int viewportHeight = c_fdStaticParams.viewportHeight;
	const int viewWidth_LOG2 = c_fdStaticParams.viewportWidth_LOG2_UP;
	const float * const upRightMat = (float*)c_fdStaticParams.light.upRightMat;
	const float2 rectangleFactor = (float2)c_fdStaticParams.light.RectangleFactor;
	const float2 rectangleSubConstant = (float2)c_fdRectangleSubConstant;
	const cudaSurfaceObject_t sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;

	// AABB
	float myXmin = FD_F32_MAX, myXmax = -FD_F32_MAX;
	float myYmin = FD_F32_MAX, myYmax = -FD_F32_MAX;
	float myZmin = FD_F32_MAX, myZmax = -FD_F32_MAX;
	float myXsum = 0.0f, myYsum = 0.0f;

	const uint sid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint sampleY = sid >> viewWidth_LOG2;
	const uint sampleX = sid & ((1 << viewWidth_LOG2) - 1);

	if (sampleX < viewportWidth && sampleY < viewportHeight)
	{
		float4 samplePos = tex2D(samplePositionTex, sampleX, sampleY);
		float3 rect;
		if (!isnan(samplePos.x)) // 剔除屏幕上的无效采样点;
		{
			// set valid;
			valid[sampleY * viewportWidth + sampleX] = 1;

			FD::makeRectangle(rect, upRightMat, make_float3(samplePos));

			myXmin = rect.x;
			myXmax = rect.x + rectangleFactor.x / rect.z - rectangleSubConstant.x;
			myYmin = rect.y;
			myYmax = rect.y + rectangleFactor.y / rect.z - rectangleSubConstant.y;
			myZmin = rect.z;
			myZmax = rect.z;

			myXsum = rectangleFactor.x / rect.z - rectangleSubConstant.x;
			myYsum = rectangleFactor.y / rect.z - rectangleSubConstant.y;
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
void saveAABBtemp(const char* des, int n);
/*
	Return: 
	1: for no sample;
	0: for ok;
*/
int sampleRectangleAABBcal()
{
	validBuffer.clear(0);

	int threadNum = m_viewHeight << m_viewWidth_LOG2UP;
	dim3 block(REDUCE_BLOCK_SIZE, 1, 1);
	dim3 grid(iDiviUp(threadNum, block.x), 1, 1);
	sampleRectangleAABBcal_kernel << <grid, block >> >();
	cudaThreadSynchronize();
	//saveAABBtemp("sampleAABBtemp", grid.x);

	while (grid.x > 1)
	{
		int n = grid.x;
		grid.x = (iDiviUp(grid.x, block.x));
		AABBReduce_kernel<reduceBoth> << <grid, block >> >(n);
		cudaThreadSynchronize();
	}


	// load to host;
	checkCudaErrors(cudaMemcpy((void *)&sampleRectangleAABB[0], AABB3DReduceBuffer.devPtr, sizeof(float)*8, cudaMemcpyDeviceToHost));

	my_debug(MY_DEBUG_SECTION_AABB, 1)("sample AABB is\n x: %f~%f\ny: %f~%f\nz: %f~%f\n", sampleRectangleAABB[0].x, sampleRectangleAABB[1].x,
		sampleRectangleAABB[0].y, sampleRectangleAABB[1].y, sampleRectangleAABB[0].z, sampleRectangleAABB[1].z);



	// get valid sample num;
	int sampleNum = m_viewWidth * m_viewHeight;
	thrust::device_ptr<int> dev_valid((int *)validBuffer.devPtr);
	int validNum = thrust::reduce(dev_valid, dev_valid + sampleNum);

	// get avg;
	if (validNum ==0)
	{
		return 1;
	}


	sampleRectangleAABB[0].w /= validNum;
	sampleRectangleAABB[1].w /= validNum;
	my_debug(MY_DEBUG_SECTION_SETUP, 1)("%d valid sample rectangle, agv is x: %f y %f\n", validNum, sampleRectangleAABB[0].w, sampleRectangleAABB[1].w);


	return 0;
}
void saveAABBtemp(const char* des, int n)
{
	char fileName[32];
	sprintf(fileName, "%s.txt", des);
	FILE * f = fopen(fileName, "w+");
	
	float * h = (float*)malloc(sizeof(float)*n *8);
	if (!f || !h)
		exit(0x1111);
	checkCudaErrors(cudaMemcpy(h, AABB3DReduceBuffer.devPtr, sizeof(float)*n *8, cudaMemcpyDeviceToHost));

	float myXmin = FD_F32_MAX, myXmax = -FD_F32_MAX;
	float myYmin = FD_F32_MAX, myYmax = -FD_F32_MAX;
	float myZmin = FD_F32_MAX, myZmax = -FD_F32_MAX;
	float myXsum = 0.0f, myYsum = 0.0f;


	fprintf(f, "%d\n", n);
	for (int i = 0; i < n; i++)
	{
		fprintf(f, "%f %f %f %f %f %f %f %f\n", h[i << 3 | 0x00], h[i << 3 | 0x01], h[i << 3 | 0x02], h[i << 3 | 0x03], h[i << 3 | 0x04], h[i << 3 | 0x05], h[i << 3 | 0x06], h[i << 3 | 0x07]);
		
		myXmin = fmin(myXmin, h[i <<3 |0x00]);
		myYmin = fmin(myYmin, h[i << 3 | 0x01]);
		myZmin = fmin(myZmin, h[i << 3 | 0x02]);
		myXsum = (myXsum+ h[i << 3 | 0x03]);
		myXmax= fmax(myXmax, h[i <<3 |0x04]);
		myYmax= fmax(myYmax, h[i <<3 |0x05]);
		myZmax= fmax(myZmax, h[i <<3 |0x06]);
		myYsum= (myYsum+ h[i <<3 |0x07]);
	}
	fprintf(f, "cpu result :\n%f %f %f %f %f %f %f %f \n", myXmin, myYmin, myZmin, myXsum, myXmax, myYmax, myZmax, myYsum);
	fclose(f);
}