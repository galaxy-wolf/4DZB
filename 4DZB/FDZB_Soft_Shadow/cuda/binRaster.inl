

#define FIND_START_END_BLOCK_SIZE 512

__global__ void FindStart_kernel(int *cellStart, int* cellEnd, int* pairs_c, int pairsNum)
{
	__shared__ int sharedHash[FIND_START_END_BLOCK_SIZE + 1];    // blockSize + 1 elements
	int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= pairsNum) return;

	int cellIndex = pairs_c[index];
	sharedHash[threadIdx.x + 1] = cellIndex;
	if (index > 0 && threadIdx.x == 0)
	{
		// first thread in block must load neighbor particle hash
		sharedHash[0] = pairs_c[index - 1];
	}

	__syncthreads();

	if (index == 0 || cellIndex != sharedHash[threadIdx.x])
	{
		cellStart[cellIndex] = index;
		if (index > 0)
			cellEnd[sharedHash[threadIdx.x]] = index;
	}

	if (index == pairsNum - 1)
	{
		cellEnd[cellIndex] = index + 1;
	}

}


void findSampleStartEnd()
{
	// clear;
	binSampleStartBuffer.clear(0);
	binSampleEndBuffer.clear(0);

	dim3 block(FIND_START_END_BLOCK_SIZE, 1);
	int pairNum = binSamplePairSampleBuffer.size_in_element;
	dim3 grid(iDiviUp(pairNum, block.x));
	FindStart_kernel << <grid, block >> >((int *)binSampleStartBuffer.devPtr, (int *)binSampleEndBuffer.devPtr, (int *)binSamplePairBinBuffer.devPtr, pairNum);
}


void findTriangleStartEnd(int pairNum)
{
	// clear;
	binTriStartBuffer.clear(0);
	binTriEndBuffer.clear(0);

	dim3 block(FIND_START_END_BLOCK_SIZE, 1);
	dim3 grid(iDiviUp(pairNum, block.x));
	FindStart_kernel << <grid, block >> >((int *)binTriStartBuffer.devPtr, (int *)binTriEndBuffer.devPtr, (int *)binTriPairBinBuffer.devPtr, pairNum);

}

void saveBinImage(int * binStart, int * binEnd, int * id, int n, char* desc)
{
	int * h_binStart = NULL;
	int * h_binEnd = NULL;
	int * h_id = NULL;

	const int m_LP_GridSizeWidth = h_fdLightPlaneParams.widthBins;
	const int m_LP_GridSizeHeight = h_fdLightPlaneParams.heightBins;
	const int m_LP_GridSizeWidthLog2 = h_fdLightPlaneParams.widthBins_LOG2;
	const int m_LP_GridSizeWidthBit = (1 << m_LP_GridSizeWidthLog2) - 1;

	const int binNum = m_LP_GridSizeHeight << m_LP_GridSizeWidthLog2;

	h_binStart = (int*)malloc(sizeof(int) * binNum);
	h_binEnd = (int*)malloc(sizeof(int) *  binNum);
	h_id = (int*)malloc(sizeof(int) * n);

	if (!h_binStart || !h_binEnd || !h_id)
	{
		fprintf(stderr, "malloc error %s %d", __FUNCTION__, __LINE__);
		exit(0x111);
	}

	checkCudaErrors(cudaMemcpy(h_binStart, binStart, sizeof(int)* binNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_binEnd, binEnd, sizeof(int)* binNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_id, id, sizeof(int)*n, cudaMemcpyDeviceToHost));

	long long sum = 0;
	int num = 0;
	int imax = std::numeric_limits<int>::min();
	int imin = std::numeric_limits<int>::max();

	int rowMax = std::numeric_limits<int>::min();
	int rowMin = std::numeric_limits<int>::max();
	int colMax = std::numeric_limits<int>::min();
	int colMin = std::numeric_limits<int>::max();

	BYTE * buf = (BYTE *)malloc(binNum * sizeof(BYTE) * 3);

	for (int i = 0; i < binNum; i++)
	{
		if ((i & m_LP_GridSizeWidthBit) >= m_LP_GridSizeWidth)
			continue;

		int len = h_binEnd[i] - h_binStart[i];
		if (len < 0)
			printf("error bin\n");
		if (len)
		{
			rowMax = std::max(rowMax, i >> m_LP_GridSizeWidthLog2);
			rowMin = std::min(rowMin, i >> m_LP_GridSizeWidthLog2);
			colMax = std::max(colMax, i & m_LP_GridSizeWidthBit);
			colMin = std::min(colMin, i & m_LP_GridSizeWidthBit);
			num++;
		}
		if (len)
			imin = min(imin, len);
		sum += len;
		imax = max(imax, len);
	}

	printf("%s 统计信息:\n", desc);
	printf("avg1: %I64d, avg2: %I64d, min %d max %d nZeroNum is %d\n", sum / m_LP_GridSizeWidth / m_LP_GridSizeHeight, sum == 0 ? sum : sum / num, imin, imax, num);
	printf("row %d~%d col %d~%d\n", rowMin, rowMax, colMin, colMax);

	char fileName[32];
	sprintf(fileName, "%s.txt", desc);
	FILE* f = fopen(fileName, "w+");
	if (!f)
	{
		fprintf(stderr, "open file err: %s\n", fileName);
		exit(0x111);
	}
	//FILE *fTemp = fopen("temp.txt", "a+");
	//if (fTemp == NULL)
	//	printf("open file fail : %s\n", "temp.txt");

	//fprintf(fTemp, "%s\n", desc);
	for (int i = 0; i < m_LP_GridSizeHeight; i++)
	{
		for (int j = 0; j < m_LP_GridSizeWidth; j++)
		{
			int len = h_binEnd[i << m_LP_GridSizeWidthLog2 | j] - h_binStart[i << m_LP_GridSizeWidthLog2 | j];

			if (imax <= 1)
				len = len * 255;
			else {
				if (len <= 0)
					len = 0;
				else
					len = (int)((255 * log10(len) / log10(imax)));
			}

			//	if (len > 0)
			//		fprintf(fTemp, "1");
			//	else
			//		fprintf(fTemp, "0");

			buf[(i* m_LP_GridSizeWidth + j) * 3] = len;
			buf[(i* m_LP_GridSizeWidth + j) * 3 + 1] = len;
			buf[(i* m_LP_GridSizeWidth + j) * 3 + 2] = len;

			fprintf(f, "%d %d (Num %d):\n", i, j, h_binEnd[i << m_LP_GridSizeWidthLog2 | j] - h_binStart[i << m_LP_GridSizeWidthLog2 | j]);
			for (int pos = h_binStart[i << m_LP_GridSizeWidthLog2 | j]; pos < h_binEnd[i << m_LP_GridSizeWidthLog2 | j]; pos++)
			{
				fprintf(f, "%d ", h_id[pos]);
			}
			fprintf(f, "\n");

		}
		//fprintf(fTemp, "\n");
	}
	//fclose(fTemp);
	fclose(f);
	sprintf(fileName, "%s.bmp", desc);
	SaveBMP1(fileName, buf, m_LP_GridSizeWidth, m_LP_GridSizeHeight);

	free(h_binStart);
	free(h_binEnd);
	free(h_id);
	free(buf);
}