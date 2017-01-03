
__global__ void isBinValid_kernle()
{
	//input;
	int * const binTriStart = (int *)c_fdStaticParams.binTriStart;
	int * const binTriEnd = (int *)c_fdStaticParams.binTriEnd;

	int * const binSampleStart = (int *)c_fdStaticParams.binPixelStart;
	int * const binSampleEnd = (int*)c_fdStaticParams.binPixelEnd;

	// output;
	int * isBinValid = (int *)c_fdStaticParams.isBinValidBuffer;

	int bid = blockIdx.x * blockDim.x + threadIdx.x;

	if (bid >= c_fdStaticParams.numBins)
		return;

	int triNum = binTriEnd[bid] - binTriStart[bid];
	int sampleNum = binSampleEnd[bid] - binSampleStart[bid];

	if (triNum <= 0 || sampleNum <= 0)
		isBinValid[bid] = 0;
	else
		isBinValid[bid] = 1;
}

__global__ void writeValidBin_kernel()
{
	//input;
	int * isBinValid = (int *)c_fdStaticParams.isBinValidBuffer;
	int * writePos = (int *)c_fdStaticParams.isBinValidPrefixSumBuffer;

	// output;
	int * validBin = (int *)c_fdStaticParams.validBinBuffer;

	int bid = blockIdx.x * blockDim.x + threadIdx.x;

	if (bid >= c_fdStaticParams.numBins)
		return;

	if (isBinValid[bid]==1)
	{
		//printf("%d: bin %d;\n", writePos[bid] - 1, bid);
		validBin[writePos[bid] - 1] = bid;
	}
}

void prepareBin()
{
	int binNum = m_binNum * m_binNum;
	dim3 block(512);
	dim3 grid(iDiviUp(binNum, block.x), 1);
	isBinValid_kernle << <grid, block >> >();

	thrust::device_ptr<int> dev_isValidBin((int *)isBinValidBuffer.devPtr);
	thrust::device_ptr<int> dev_isValidBinPrefixSum((int *)isBinValidPrefixSumBuffer.devPtr);
	thrust::inclusive_scan(dev_isValidBin, dev_isValidBin + binNum, dev_isValidBinPrefixSum);

	writeValidBin_kernel<<<grid, block>>>();

	checkCudaErrors(cudaMemcpy(&m_validBinNum, ((int *)isBinValidPrefixSumBuffer.devPtr) + binNum-1, sizeof(int), cudaMemcpyDeviceToHost));
	
	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("valid bin num is %d\n", m_validBinNum);

}