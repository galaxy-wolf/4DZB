
__global__ void isBinValid_kernle()
{
	//input;
	const int * const binTriStart = (int *)c_fdStaticParams.binTriStart;
	const int * const binTriEnd = (int *)c_fdStaticParams.binTriEnd;

	const int * const binSampleStart = (int *)c_fdStaticParams.binPixelStart;
	const int * const binSampleEnd = (int*)c_fdStaticParams.binPixelEnd;

	const int binWidth = c_fdLightPlaneParams.widthBins;
	const int binHeight = c_fdLightPlaneParams.heightBins;
	const int binWidthLog2 = c_fdLightPlaneParams.widthBins_LOG2;

	// output;
	int * const isBinValid = (int *)c_fdStaticParams.isBinValidBuffer;

	const int binX = blockIdx.x * blockDim.x + threadIdx.x;
	const int binY = blockIdx.y * blockDim.y + threadIdx.y;

	if ( binX >= binWidth || binY >= binHeight)
		return; 

	const int binID = binY << binWidthLog2 | binX;

	const int triNum = binTriEnd[binID] - binTriStart[binID];
	const int sampleNum = binSampleEnd[binID] - binSampleStart[binID];

	if (triNum <= 0 || sampleNum <= 0)
	{
		//// atomicAdd to global; 
		if (triNum <= 0 && sampleNum > 0)
		{
			atomicAdd(&(g_fdAtomics.invalidSampleCounter), sampleNum);
		}
		if (sampleNum <= 0 && triNum > 0)
		{
			atomicAdd(&(g_fdAtomics.invalidBT_PairCounter), triNum);
		}
	}
	else
		isBinValid[binID] = 1;
}

__global__ void writeValidBin_kernel()
{
	//input;
	const int * const isBinValid = (int *)c_fdStaticParams.isBinValidBuffer;
	const int * const writePos = (int *)c_fdStaticParams.isBinValidPrefixSumBuffer;

	const int binWidth = c_fdLightPlaneParams.widthBins;
	const int binHeight = c_fdLightPlaneParams.heightBins;
	const int binWidthLog2 = c_fdLightPlaneParams.widthBins_LOG2;

	// output;
	int * const validBin = (int *)c_fdStaticParams.validBinBuffer;


	const int binX = blockIdx.x * blockDim.x + threadIdx.x;
	const int binY = blockIdx.y * blockDim.y + threadIdx.y;

	if (binX >= binWidth || binY >= binHeight)
		return;

	const int binID = binY << binWidthLog2 | binX;

	if (isBinValid[binID] == 1)
	{
		//printf("%d: bin %d;\n", writePos[bid] - 1, bid);
		validBin[writePos[binID] - 1] = binID;
	}
}


void prepareBin(int & sampleNumPerBlock, int & sharedMemPerBlockSizeByte, int & warpNumPerBlock, int & gridSizeX)
{
	//////////////////////////////////////////////////
	// step 1 get valid bin
	//////////////////////////////////////////////////////
	isBinValidBuffer.clear(0);

	const int binNum = h_fdLightPlaneParams.heightBins << h_fdLightPlaneParams.widthBins_LOG2;
	dim3 block(16, 16);
	dim3 grid(iDiviUp(h_fdLightPlaneParams.widthBins, block.x), iDiviUp(h_fdLightPlaneParams.heightBins, block.y));
	isBinValid_kernle << <grid, block >> >();

	thrust::device_ptr<int> dev_isValidBin((int *)isBinValidBuffer.devPtr);
	thrust::device_ptr<int> dev_isValidBinPrefixSum((int *)isBinValidPrefixSumBuffer.devPtr);
	thrust::inclusive_scan(dev_isValidBin, dev_isValidBin + binNum, dev_isValidBinPrefixSum);

	writeValidBin_kernel<<<grid, block>>>();

	checkCudaErrors(cudaMemcpy(&m_validBinNum, ((int *)isBinValidPrefixSumBuffer.devPtr) + binNum-1, sizeof(int), cudaMemcpyDeviceToHost));
	
	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("valid bin num is %d\n", m_validBinNum);

	//////////////////////////////////////////////////////
	// step 2 count avg sample per bin and avg tri per bin
	//////////////////////////////////////////////////////
	FDAtomics temp;
	checkCudaErrors(cudaMemcpy(&temp, g_fdAtomics_addr, sizeof(FDAtomics), cudaMemcpyDeviceToHost));
	m_validSampleNum -= temp.invalidSampleCounter;
	m_validBT_PairNum -= temp.invalidBT_PairCounter;

	const float avgSamplePerBin = (float)m_validSampleNum / (float)m_validBinNum;
	const float avgBT_pairPerBin = (float)m_validBT_PairNum / (float)m_validBinNum;

	///////////////////////////////////////////////////
	// step 3 将一个device 限制的最大的block 分成小的block，得到小block 中sample 和线程warp 个数 
	///////////////////////////////////////////////////
	const int lightResWidth = h_fdStaticParams.light.lightResWidth;
	const int lightResHeight = h_fdStaticParams.light.lightResHeight;
	const int manageSizeU32 = 12; // constant shared memory 48 Bytes，
	
	int sampleSizeU32;
	if (lightResWidth <= 32)// use U32
	{
		sampleSizeU32 = (lightResHeight + 1) * 1 + 9; // 9 是存放在share memory 中的sample pos 等数据的大小。
	}
	else{ // use U64;
		sampleSizeU32 = (lightResHeight + 1) * 2 + 9;
	}

	// find best block size;
	for (int i = FD_MAX_BLOCK_PER_SM; i >= FD_MIN_BLOCK_PER_SM; i /= 2)
	{
		warpNumPerBlock = FD_MAX_WARPS_PER_SM / i;
		int threadsPerBlock = warpNumPerBlock * 32;
		sharedMemPerBlockSizeByte = FD_SHARED_MEM_PER_SM / i - manageSizeU32 * 4;
		sampleNumPerBlock = (sharedMemPerBlockSizeByte / 4) / sampleSizeU32;
		gridSizeX = i * my_device.SM_num;

		if (sampleNumPerBlock >= avgSamplePerBin || threadsPerBlock >= 1.2f * avgBT_pairPerBin)
		{
			return;
		}
	}
	// at last use max block;
	return;


	//// sample 决定 block num 的初始值
	//int smallBlockNumPerBigBlock = (float)my_device.maxSharedMemPerBlock / 4.0f / (float)(manageSizeU32 + sampleSizeU32 * avgSamplePerBin);
	//smallBlockNumPerBigBlock = max(1, smallBlockNumPerBigBlock);
	//// 根据 每个block 最多能装的线程数减少blockNum， 同时增大了每个block中sample 和 thread的个数。保证线程数能被block num 整除。
	//while (m_maxShadowCalWarpNumPerBlock % smallBlockNumPerBigBlock) smallBlockNumPerBigBlock--;

	//if (m_maxShadowCalWarpNumPerBlock / smallBlockNumPerBigBlock == 1)  smallBlockNumPerBigBlock /= 2;
	////smallBlockNumPerBigBlock = 20;

	//smallBlockNumPerBigBlock = 16;

	//warpNumPerBlock = m_maxShadowCalWarpNumPerBlock / smallBlockNumPerBigBlock;
	//sampleNumPerBlock = ((my_device.maxSharedMemPerBlock / 4.0f / smallBlockNumPerBigBlock) - manageSizeU32 ) / sampleSizeU32 ;
	//gridSizeX = smallBlockNumPerBigBlock * my_device.bigBlockPerSM * my_device.SM_num;
	//sharedMemPerBlockSizeByte = my_device.maxSharedMemPerBlock / smallBlockNumPerBigBlock - manageSizeU32 * 4;


	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("warp num per block is %d\n", warpNumPerBlock);
	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("sample num per block is %d\n", sampleNumPerBlock);
	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("gridSizeX is %d\n", gridSizeX);
	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("sharedMemPerBlockSizeByte is %d\n", sharedMemPerBlockSizeByte);

}