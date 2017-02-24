

#define REDUCE_MAXZ_BLOCK_SIZE 64

__global__ void writeSampleAndBinPair()
{
	// input;
	const int viewportX = c_fdStaticParams.viewportWidth;
	const int viewportY = c_fdStaticParams.viewportHeight;
	const int viewportX_LOG2UP = c_fdStaticParams.viewportWidth_LOG2_UP;
	const int binWidth = c_fdLightPlaneParams.widthBins;
	const int binWidth_LOG2 = c_fdLightPlaneParams.widthBins_LOG2;
	const int binHeight = c_fdLightPlaneParams.heightBins;
	const int INVALID_BIN = c_fdLightPlaneParams.numBins;
	const float3 lightPlaneB = c_fdLightPlaneParams.begin;
	const float3 lightPlaneE = c_fdLightPlaneParams.end;
	const float2 lightPlaneFactor = c_fdLightPlaneParams.factor;
	const cudaSurfaceObject_t sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;

	// output;
	int * const pair_sample = (int *)c_fdStaticParams.pb_pair_pixel;
	int * const pair_bin = (int*)c_fdStaticParams.pb_pair_bin;
	int * const valid = (int *)c_fdStaticParams.validTempBuffer;


	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < viewportX && y < viewportY)
	{
		const float4 rectB = surf2DLayeredread<float4>(sampleRectangleSurf, x * sizeof(float4), y, 0, cudaBoundaryModeTrap);
		const float4 rectE = surf2DLayeredread<float4>(sampleRectangleSurf, x * sizeof(float4), y, 1, cudaBoundaryModeTrap);

		int result = INVALID_BIN;
		if ((rectB.x > rectE.x || rectB.y > rectE.y) ||// 该像素点rectangle不存在，可能是像素是屏幕的背景。
			 (rectB.x < lightPlaneB.x || rectB.y <lightPlaneB.y || rectE.x > lightPlaneE.x || rectE.y > lightPlaneE.y) // 该矩形没有全部在light plane 中，说明没有三角形能遮挡该像素
																														 //|| (rectB.x > lightPlaneE.x || rectE.x < lightPlaneB.x || rectB.y > lightPlaneE.y || rectE.y < lightPlaneB.y)
			)
		{
			// 此时阴影值为0， 不需要进一步计算；
		}
		else
		{
			const float2 center = (make_float2(rectE) + make_float2(rectB)) / 2;//make_float2(rectB);//
			int2 mybin;
			mybin.x = floorf((center.x - lightPlaneB.x) * lightPlaneFactor.x);
			mybin.y = floorf((center.y - lightPlaneB.y) * lightPlaneFactor.y);
			result = mybin.y << binWidth_LOG2 | mybin.x;

			valid[y * viewportX + x] = 1;

		}

		// write out
		const int writePos = y * viewportX + x;
		pair_sample[writePos] = y << viewportX_LOG2UP | x;
		pair_bin[writePos] = result;
	}
}

__global__ void maxZpreBinReduce_kernel()
{
	const int binWidth_LOG2 = c_fdLightPlaneParams.widthBins_LOG2;
	const int myBin = blockIdx.y << binWidth_LOG2 | blockIdx.x;
	const cudaSurfaceObject_t  sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;
	// 
	const int viewportWidth_LOG2 = c_fdStaticParams.viewportWidth_LOG2_UP;

	// input;
	const int sampleStart = ((int*)c_fdStaticParams.binPixelStart)[myBin];
	const int sampleEnd = ((int*)c_fdStaticParams.binPixelEnd)[myBin];
	const int sampleNum = sampleEnd - sampleStart;
	const int * const sample = ((int *)c_fdStaticParams.pb_pair_pixel) + sampleStart;

	// output;
	float * const MaxZopos = ((float*)c_fdStaticParams.binPixelMaxZperBin) + myBin;
	float4 * const minRanger = ((float4*)c_fdStaticParams.binPixelMinRangeperBin) + myBin;
	float* const maxX_startPos = &(minRanger->x);
	float* const minX_endPos = &(minRanger->z);
	float* const maxY_startPos = &(minRanger->y);
	float* const minY_endPos = &(minRanger->w);

	float myMaxZ = -FD_F32_MAX;
	float4 myMinRange = make_float4(-FD_F32_MAX, -FD_F32_MAX, FD_F32_MAX, FD_F32_MAX);
	for (int i = threadIdx.x; i < sampleNum; i += blockDim.x)
	{
		const int mySample = sample[i];
		const int sampleX = mySample &((1 << viewportWidth_LOG2) - 1);
		const int sampleY = mySample >> viewportWidth_LOG2;
		float4 rect = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX * sizeof(float4), sampleY, 0, cudaBoundaryModeTrap);
		myMaxZ = fmax(myMaxZ, rect.z);
		myMinRange.x = fmax(myMinRange.x, rect.x);
		myMinRange.y = fmax(myMinRange.y, rect.y);
		rect = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX * sizeof(float4), sampleY, 1, cudaBoundaryModeTrap);
		myMinRange.z = fmin(myMinRange.z, rect.x);
		myMinRange.w = fmin(myMinRange.w, rect.y);
	}

	block_reduce<float, OperatorMax<float>, REDUCE_MAXZ_BLOCK_SIZE>(MaxZopos, myMaxZ);
	block_reduce<float, OperatorMax<float>, REDUCE_MAXZ_BLOCK_SIZE>(maxX_startPos, myMinRange.x);
	block_reduce<float, OperatorMax<float>, REDUCE_MAXZ_BLOCK_SIZE>(maxY_startPos, myMinRange.y);
	block_reduce<float, OperatorMin<float>, REDUCE_MAXZ_BLOCK_SIZE>(minX_endPos, myMinRange.z);
	block_reduce<float, OperatorMin<float>, REDUCE_MAXZ_BLOCK_SIZE>(minY_endPos, myMinRange.w);
}
void bindSampleToBin()
{

	validBuffer.clear(0);

	// write pair;
	{
		dim3 block(16, 16);
		dim3 grid(iDiviUp(m_viewWidth, block.x), iDiviUp(m_viewHeight, block.y));
		writeSampleAndBinPair << <grid, block >> >();
	}
	cudaDeviceSynchronize();
	//getLastCudaError("writeSampleAndBinPair");

	// get valid sample num;
	int sampleNum = m_viewWidth * m_viewHeight;
	thrust::device_ptr<int> dev_valid((int *)validBuffer.devPtr);
	m_validSampleNum = thrust::reduce(dev_valid, dev_valid + sampleNum);

	// sort pair;
	thrust::device_ptr<int> dev_sample((int*)binSamplePairSampleBuffer.devPtr);
	thrust::device_ptr<int> dev_bin((int*)binSamplePairBinBuffer.devPtr);
	thrust::sort_by_key(dev_bin, dev_bin + binSamplePairBinBuffer.size_in_element, dev_sample);

	cudaDeviceSynchronize();
	//getLastCudaError("sort");

	// find pair start;
	findSampleStartEnd();
	cudaDeviceSynchronize();
	//	getLastCudaError("find start end");
	// cal maxZ for per bin;
	{
		dim3 block(REDUCE_MAXZ_BLOCK_SIZE);
		dim3 grid(h_fdLightPlaneParams.widthBins, h_fdLightPlaneParams.heightBins);
		maxZpreBinReduce_kernel << <grid, block >> >();
	}
	cudaDeviceSynchronize();
	my_debug(MY_DEBUG_SECTION_RASTER, 1)("bind sample to bin done!\n");

#ifdef _DEBUG
	saveBinImage((int *)binSampleStartBuffer.devPtr, (int *)binSampleEndBuffer.devPtr, (int *)binSamplePairSampleBuffer.devPtr, binSamplePairSampleBuffer.size_in_element, "E:\\Results\\binSample");
#endif// _DEBUG
}

