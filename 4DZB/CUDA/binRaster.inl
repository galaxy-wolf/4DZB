

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
	dim3 block(FIND_START_END_BLOCK_SIZE, 1);
	int pairNum = binSamplePairSampleBuffer.size_in_element;
	dim3 grid(iDiviUp(pairNum, block.x));
	FindStart_kernel << <grid, block >> >((int *)binSampleStartBuffer.devPtr, (int *)binSampleEndBuffer.devPtr, (int *)binSamplePairBinBuffer.devPtr, pairNum);	
}


void findTriangleStartEnd()
{
	dim3 block(FIND_START_END_BLOCK_SIZE, 1);
	int pairNum = binTriPairTriBuffer.size_in_element;
	dim3 grid(iDiviUp(pairNum, block.x));
	FindStart_kernel << <grid, block >> >((int *)binTriStartBuffer.devPtr, (int *)binTriEndBuffer.devPtr, (int *)binTriPairBinBuffer.devPtr, pairNum);
	
}

void saveBinImage(int * binStart, int * binEnd, int * id, int n, char* desc)
{
	int * h_binStart = NULL;
	int * h_binEnd = NULL;
	int * h_id = NULL;

	h_binStart = (int*)malloc(sizeof(int) * m_binNum * m_binNum);
	h_binEnd = (int*)malloc(sizeof(int) * m_binNum* m_binNum);
	h_id = (int*)malloc(sizeof(int) * n);

	if (!h_binStart || !h_binEnd || !h_id)
	{
		fprintf(stderr, "malloc error %s %d", __FUNCTION__, __LINE__);
		exit(0x111);
	}

	checkCudaErrors(cudaMemcpy(h_binStart, binStart, sizeof(int)*m_binNum* m_binNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_binEnd, binEnd, sizeof(int)*m_binNum* m_binNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_id, id, sizeof(int)*n, cudaMemcpyDeviceToHost));

	long long sum = 0;
	int num = 0;
	int imax = std::numeric_limits<int>::min();
	int imin = std::numeric_limits<int>::max();

	int rowMax = std::numeric_limits<int>::min();
	int rowMin = std::numeric_limits<int>::max();
	int colMax = std::numeric_limits<int>::min();
	int colMin = std::numeric_limits<int>::max();

	BYTE * buf = (BYTE *)malloc(m_binNum* m_binNum * sizeof(BYTE) * 3);

	for (int i = 0; i < m_binNum* m_binNum; i++)
	{	
		int len = h_binEnd[i] - h_binStart[i];
		if (len)
		{
			rowMax = std::max(rowMax, i / m_binNum);
			rowMin = std::min(rowMin, i / m_binNum);
			colMax = std::max(colMax, i%m_binNum);
			colMin = std::min(colMin, i%m_binNum);
			num++;
		}
		if (len)
			imin = min(imin, len);
		sum += len;
		imax = max(imax, len);
	}
	printf("%s 统计信息:\n", desc);
	printf("avg1: %I64d, avg2: %I64d, min %d max %d nZeroNum is %d\n", sum / m_binNum, sum / num, imin, imax, num);
	printf("row %d~%d col %d~%d\n", rowMin, rowMax, colMin, colMax);

	char fileName[32];
	sprintf(fileName, "%s.txt", desc);
	FILE* f = fopen(fileName, "w+");
	for (int i = 0; i < m_binNum; i++)
	{
		for (int j = 0; j < m_binNum; j++)
		{
			int len = h_binEnd[i*m_binNum + j] - h_binStart[i*m_binNum + j];
			
			if (len <= 0)
				len = 0;
			else
				len = (int)(255 * log10(len) / log10(imax));

			buf[(i* m_binNum + j) * 3] = len;
			buf[(i* m_binNum + j) * 3 + 1] = len;
			buf[(i* m_binNum + j) * 3 + 2] = len;

			fprintf(f, "%d %d (Num %d):\n", i, j, h_binEnd[i*m_binNum + j] - h_binStart[i*m_binNum + j]);
			for (int pos = h_binStart[i*m_binNum + j]; pos < h_binEnd[i*m_binNum + j]; pos++)
			{
				fprintf(f, "%d ", h_id[pos]);
			}
			fprintf(f, "\n");
		}
	}
	fclose(f);
	sprintf(fileName, "%s.bmp", desc);
	SaveBMP1(fileName, buf, m_binNum, m_binNum);

	free(h_binStart);
	free(h_binEnd);
	free(h_id);
	free(buf);
}