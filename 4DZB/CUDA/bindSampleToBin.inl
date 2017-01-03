

#define REDUCE_MAXZ_BLOCK_SIZE 64

__global__ void writeSampleAndBinPair()
{
	// input;
	int viewportX = c_fdStaticParams.viewportWidth;
	int viewportY = c_fdStaticParams.viewportHeight;
	int viewportX_LOG2UP = c_fdStaticParams.viewportWidth_LOG2_UP;
	int binWidth = c_fdStaticParams.widthBins;
	int binWidth_LOG2 = c_fdStaticParams.widthBins_LOG2;
	int binHeight = c_fdStaticParams.heightBins;
	int INVALID_BIN = c_fdStaticParams.numBins;
	float3 lightPlaneB = c_fdLightPlaneParams.begin;
	float3 lightPlaneE = c_fdLightPlaneParams.end;
	float2 lightPlaneFactor = c_fdLightPlaneParams.factor;
	cudaSurfaceObject_t sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;

	// output;
	int * pair_sample = (int *)c_fdStaticParams.pb_pair_pixel;
	int * pair_bin = (int*)c_fdStaticParams.pb_pair_bin;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < viewportX && y < viewportY)
	{
		float4 rectB, rectE;
		rectB = surf2DLayeredread<float4>(sampleRectangleSurf, x*sizeof(float4), y, 0, cudaBoundaryModeTrap);
		rectE = surf2DLayeredread<float4>(sampleRectangleSurf, x*sizeof(float4), y, 1, cudaBoundaryModeTrap);

		int result = INVALID_BIN;
		if ((rectB.x > rectE.x || rectB.y > rectE.y) // 该像素点rectangle不存在，可能是像素是屏幕的背景。
			|| (rectB.x < lightPlaneB.x || rectB.y <lightPlaneB.y || rectE.x > lightPlaneE.x || rectE.y > lightPlaneE.y) // 该矩形没有全部在light plane 中，说明没有三角形能遮挡该像素
			)
		{
			// 此时阴影值为0， 不需要进一步计算；
		}
		else{
			float2 center = (make_float2(rectE) + make_float2(rectB)) / 2;
			int2 mybin;
			mybin.x = floorf((center.x - lightPlaneB.x) * lightPlaneFactor.x);
			mybin.y = floorf((center.y - lightPlaneB.y) * lightPlaneFactor.y);
			result = mybin.y << binWidth_LOG2 | mybin.x;
		}

		// write out
		int writePos = y * viewportX + x;
		pair_sample[writePos] = y << viewportX_LOG2UP | x;
		pair_bin[writePos] = result;
	}
}

__global__ void maxZpreBinReduce_kernel()
{
	int binWidth_LOG2 = c_fdStaticParams.widthBins_LOG2;
	int myBin = blockIdx.y << binWidth_LOG2 | blockIdx.x;
	cudaSurfaceObject_t  sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;
	// 
	int viewportWidth_LOG2 = c_fdStaticParams.viewportWidth_LOG2_UP;

	// input;
	int sampleStart = ((int*)c_fdStaticParams.binPixelStart)[myBin];
	int sampleEnd = ((int*)c_fdStaticParams.binPixelEnd)[myBin];
	int sampleNum = sampleEnd - sampleStart;
	int * sample = ((int *)c_fdStaticParams.pb_pair_pixel) + sampleStart;

	// output;
	float * opos = ((float*)c_fdStaticParams.binPixelMaxZperBin) + myBin;

	float myMaxZ = -FD_F32_MAX;
	for (int i = threadIdx.x; i < sampleNum; i += blockDim.x)
	{
		int mySample = sample[i];
		int sampleX = mySample &((1 << viewportWidth_LOG2) - 1);
		int sampleY = mySample >> viewportWidth_LOG2;
		float4 rect = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 0, cudaBoundaryModeTrap);
		myMaxZ = fmax(myMaxZ, rect.z);
	}

	block_reduce<float, OperatorMax<float>, REDUCE_MAXZ_BLOCK_SIZE>(opos, myMaxZ);
}
void bindSampleToBin()
{
	// write pair;
	{
		dim3 block(16, 16);
		dim3 grid(iDiviUp(m_viewWidth, block.x), iDiviUp(m_viewHeight, block.y));
		writeSampleAndBinPair << <grid, block >> >();
	}
	cudaDeviceSynchronize();
	getLastCudaError("writeSampleAndBinPair");
	// sort pair;
	thrust::device_ptr<int> dev_sample((int*)binSamplePairSampleBuffer.devPtr);
	thrust::device_ptr<int> dev_bin((int*)binSamplePairBinBuffer.devPtr);
	thrust::sort_by_key(dev_bin, dev_bin + binSamplePairBinBuffer.size_in_element, dev_sample);

	cudaDeviceSynchronize();
	getLastCudaError("sort");

	// find pair start;
	findSampleStartEnd();
	cudaDeviceSynchronize();
	getLastCudaError("find start end");
	// cal maxZ for per bin;
	{
		dim3 block(REDUCE_MAXZ_BLOCK_SIZE);
		dim3 grid(m_binNum, m_binNum);
		maxZpreBinReduce_kernel << <grid, block >> >();
	}
	my_debug(MY_DEBUG_SECTION_RASTER, 1)("bind sample to bin done!\n");

	saveBinImage((int *)binSampleStartBuffer.devPtr, (int *)binSampleEndBuffer.devPtr, (int *)binSamplePairSampleBuffer.devPtr, binSamplePairSampleBuffer.size_in_element, "binSample");
}

