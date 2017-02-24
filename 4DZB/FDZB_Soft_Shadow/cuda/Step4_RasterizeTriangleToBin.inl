

// cache bin range to  triBinRangeBuffer 
//template<bool enableBackFaceCull>
__global__ void countPairNumForPerTri_kernel()
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// input;
	//bool enable_backfaceCull = c_fdStaticParams.enable_backfaceCull;
	const int tNum = c_fdStaticParams.numTris;
	/*const float3 upRightPos = c_fdStaticParams.light.upRightCornerPosition;
	const float3 downLeftPos = c_fdStaticParams.light.downLeftCornerPosition;
	const float3 upLeftPos = c_fdStaticParams.light.upLeftCornerPosition;
	const float3 downRightPos = c_fdStaticParams.light.downRightCornerPosition;*/
	const int binWidth = c_fdLightPlaneParams.widthBins;
	const int binWidth_LOG2 = c_fdLightPlaneParams.widthBins_LOG2;
	const int binHeight = c_fdLightPlaneParams.heightBins;
	const float3 lightPlaneB = c_fdLightPlaneParams.begin;
	const float3 lightPlaneE = c_fdLightPlaneParams.end;
	const float2 lightPlaneFactor = c_fdLightPlaneParams.factor;
	const cudaTextureObject_t triangleAABBTex = c_fdStaticParams.triangleAABBTex;
	//const cudaTextureObject_t triangleVertexTex = c_fdStaticParams.triangleVertexTex;
	const cudaTextureObject_t binSampleMaxZTex = c_fdStaticParams.binSampleMaxZTex;
	const cudaTextureObject_t binSampleMinRangeTex = c_fdStaticParams.binSampleMinRangeTex;
	
	// output;
	struct BinRange * const myBinRange = ((struct BinRange*)c_fdStaticParams.triBinRangeBuffer) + tid;
	int * const myBinNumPos = ((int *)c_fdStaticParams.triBinNum) + tid;

	if (tid >= tNum)
		return;
	//float4 p[3];
	float4 AABBb, AABBe;
	ushort2 binB = { 6000, 6000 }, binE = {0, 0};

	AABBb = tex1Dfetch<float4>(triangleAABBTex, tid << 1 | 0);
	AABBe = tex1Dfetch<float4>(triangleAABBTex, tid << 1 | 1);

	int result = 0;
	if (AABBb.x > AABBe.x || AABBb.y > AABBe.y || AABBb.z > AABBe.z || AABBb.x > lightPlaneE.x || AABBb.y > lightPlaneE.y || AABBe.x < lightPlaneB.x || AABBe.y < lightPlaneB.y) // AABB全部在lightPlane外部。
	{
		// 三角形被剔除。
	}
	else{
		/*p[0] = tex1Dfetch<float4>(triangleVertexTex, tid * 3);
		p[1] = tex1Dfetch<float4>(triangleVertexTex, tid * 3 + 1);
		p[2] = tex1Dfetch<float4>(triangleVertexTex, tid * 3 + 2);
		float3 e0 = make_float3(p[1]) - make_float3(p[0]);
		float3 e1 = make_float3(p[0]) - make_float3(p[2]);
		float3 n = cross(e1, e0);*/


		//do backface cull
		//if (enableBackFaceCull &&
		//	dot(n, upRightPos - make_float3(p[1])) <= 0 &&
		//	dot(n, upRightPos - make_float3(p[2])) <= 0 &&
		//	dot(n, upRightPos - make_float3(p[0])) <= 0 &&
		//	dot(n, upLeftPos - make_float3(p[0])) <= 0 &&
		//	dot(n, upLeftPos - make_float3(p[1])) <= 0 &&
		//	dot(n, upLeftPos - make_float3(p[2])) <= 0 &&
		//	dot(n, downRightPos - make_float3(p[0])) <= 0 &&
		//	dot(n, downRightPos - make_float3(p[1])) <= 0 &&
		//	dot(n, downRightPos - make_float3(p[2])) <= 0 &&
		//	dot(n, downLeftPos - make_float3(p[0])) <= 0 &&
		//	dot(n, downLeftPos - make_float3(p[1])) <= 0 &&
		//	dot(n, downLeftPos - make_float3(p[2])) <= 0
		//	)										 
		//{
		//	// 三角形相对面光源是反面被剔除掉。
		//}
		//else
		{
			AABBb.x = fmaxf(AABBb.x, lightPlaneB.x);
			AABBb.y = fmaxf(AABBb.y, lightPlaneB.y);
			AABBe.x = fminf(AABBe.x, lightPlaneE.x);
			AABBe.y = fminf(AABBe.y, lightPlaneE.y);

			ushort2 c_binB, c_binE;
			c_binB.x = fmaxf(floorf((AABBb.x - lightPlaneB.x) * lightPlaneFactor.x), 0.0f);
			c_binB.y = fmaxf(floorf((AABBb.y - lightPlaneB.y) * lightPlaneFactor.y), 0.0f);
			c_binE.x = fminf(ceilf((AABBe.x - lightPlaneB.x) * lightPlaneFactor.x), binWidth);
			c_binE.y = fminf(ceilf((AABBe.y - lightPlaneB.y) * lightPlaneFactor.y), binHeight);
			
			
			for (int j = c_binB.y; j < c_binE.y; j++)
			{
				for (int i = c_binB.x; i < c_binE.x; i++)
				{
					int binPos = j << binWidth_LOG2 | i;
					float BinsampleMaxZ = tex1Dfetch<float>(binSampleMaxZTex, binPos);
					float4 BinsampleMinRange = tex1Dfetch<float4>(binSampleMinRangeTex, binPos);

					if (BinsampleMaxZ >= AABBb.z && AABBb.x < BinsampleMinRange.x && AABBb.y < BinsampleMinRange.y && BinsampleMinRange.z < AABBe.x && BinsampleMinRange.w < AABBe.y)
					{
						result++;
						binB.x = min(binB.x, i);
						binB.y = min(binB.y, j);
						binE.x = max(binE.x, i+1);
						binE.y = max(binE.y, j + 1);
					}
				}
			}

		}

	}

	//write out;
	*myBinNumPos = result;
	// write bin range
	*myBinRange = BinRange{ binB, binE, AABBb.z, AABBe.z };
}

void showPairNumPerTri();
int countTriBinPairNum()
{
	// count per tri's bin num
	{
		dim3 block(512);
		dim3 grid(iDiviUp(m_triangleNum, block.x));
		//if (m_enable_backfacecull)
			countPairNumForPerTri_kernel/*<true> */<< <grid, block >> >();
		//else
			//countPairNumForPerTri_kernel<false> << <grid, block >> >();
	}
	cudaDeviceSynchronize();

	// count prefixSum;
	thrust::device_ptr<int> dev_pairNumPrefixsum((int *)triPairNumPrefixSumBuffer.devPtr);
	thrust::device_ptr<int> dev_pairNum((int *)triPairNumBuffer.devPtr);
	thrust::exclusive_scan(dev_pairNum, dev_pairNum + m_triangleNum, dev_pairNumPrefixsum);
	cudaDeviceSynchronize();

	showPairNumPerTri();

	// load to host;
	int pairNumALL;
	int lastTriPairNum;
	checkCudaErrors(cudaMemcpy(&pairNumALL, ((int *)triPairNumPrefixSumBuffer.devPtr) + m_triangleNum - 1, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&lastTriPairNum, ((int *)triPairNumBuffer.devPtr) + m_triangleNum - 1, sizeof(int), cudaMemcpyDeviceToHost));

	pairNumALL += lastTriPairNum;

	// alloc buffers;
	/*if (binTriPairBinBuffer.size_in_element < pairNumALL)
	{
		binTriPairBinBuffer.freeBuffer();
		binTriPairBinBuffer.size_in_element = pairNumALL;
		binTriPairBinBuffer.allocBuffer();

		binTriPairTriBuffer.freeBuffer();
		binTriPairTriBuffer.size_in_element = pairNumALL;
		binTriPairTriBuffer.allocBuffer();
	}*/
	my_debug(MY_DEBUG_SECTION_RASTER, 1)("count tri pair num done, pair num is %.2fMillion\n", (float)pairNumALL/ 1024.0f/1024.0f);
	
	return pairNumALL;
}

__global__ void writeTriBinPair_kernel()
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// input;
	const struct BinRange  myBinRange = ((struct BinRange*)c_fdStaticParams.triBinRangeBuffer)[tid];
	const int tNum = c_fdStaticParams.numTris;
	const int binWidth_LOG2 = c_fdLightPlaneParams.widthBins_LOG2;
	const cudaTextureObject_t binSampleMaxZTex = c_fdStaticParams.binSampleMaxZTex;
	const cudaTextureObject_t binSampleMinRangeTex = c_fdStaticParams.binSampleMinRangeTex;
	const cudaTextureObject_t triangleAABBTex = c_fdStaticParams.triangleAABBTex;

	if (tid >= tNum)
		return;
	const int myFirstPos = ((int *)c_fdStaticParams.triBinNumPrefixSum)[tid];

	const float4 AABBb = tex1Dfetch<float4>(triangleAABBTex, tid << 1 | 0);
	const float4 AABBe = tex1Dfetch<float4>(triangleAABBTex, tid << 1 | 1);

	// output;
	int * const binPos = (int *)c_fdStaticParams.bt_pair_bin + myFirstPos;
	int * const triPos = (int *)c_fdStaticParams.bt_pair_tri + myFirstPos;

	int cnt = 0;
	for (int j = myBinRange.start.y; j < myBinRange.end.y; j++)
	{
		for (int i = myBinRange.start.x; i < myBinRange.end.x; i++)
		{
			const int binNum = j << binWidth_LOG2 | i;
			float BinsampleMaxZ = tex1Dfetch<float>(binSampleMaxZTex, binNum);
			float4 BinsampleMinRange = tex1Dfetch<float4>(binSampleMinRangeTex, binNum);

			if (BinsampleMaxZ >= AABBb.z && AABBb.x < BinsampleMinRange.x && AABBb.y < BinsampleMinRange.y && BinsampleMinRange.z < AABBe.x && BinsampleMinRange.w < AABBe.y)
			//if (BinsampleMaxZ >= myBinRange.minZ)
			{
				binPos[cnt] = j<<binWidth_LOG2|i;
				triPos[cnt] = tid;
				cnt++;
			}

		}
	}

}

void showBinTriPair(int pairNum);
void bindTriToBin(int pairNum)
{
	//printf("pari num is %d\n", pairNum);
	// write pair
	{
		dim3 block(512);
		dim3 grid(iDiviUp(m_triangleNum, block.x));
		writeTriBinPair_kernel << <grid, block >> >();
	}
	cudaDeviceSynchronize();

	showBinTriPair(pairNum);

	// sort pair
	thrust::device_ptr<int> dev_tri((int*)binTriPairTriBuffer.devPtr);
	thrust::device_ptr<int> dev_bin((int *)binTriPairBinBuffer.devPtr);
	thrust::sort_by_key(dev_bin, dev_bin + pairNum, dev_tri);

	//find start;
	findTriangleStartEnd(pairNum);
	//
	cudaDeviceSynchronize();

	my_debug(MY_DEBUG_SECTION_RASTER, 1)("bind tri to bin done!\n");
#ifdef _DEBUG
	saveBinImage((int *)binTriStartBuffer.devPtr, (int *)binTriEndBuffer.devPtr, (int *)binTriPairTriBuffer.devPtr, pairNum, "E:\\Results\\binTri");
#endif
}

void showBinTriPair(int pairNum)
{
	int * h_tri = (int *) malloc(pairNum * sizeof(int));
	int * h_bin = (int *) malloc(pairNum * sizeof(int));

	checkCudaErrors(cudaMemcpy(h_tri, binTriPairTriBuffer.devPtr, pairNum * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_bin, binTriPairBinBuffer.devPtr, pairNum * sizeof(int), cudaMemcpyDeviceToHost));

	FILE *f = fopen("E:\\Results\\binTriPairO.txt", "w+");
	for (int i = 0; i < pairNum; i++)
		fprintf(f, "%d %d\n", h_tri[i], h_bin[i]);
	fclose(f);
	free(h_tri);
	free(h_bin);
}

void showPairNumPerTri()
{
	printf("trinum \n");
	int * h_p = (int *)malloc(m_triangleNum * sizeof(int));
	int * h_pp = (int*)malloc(m_triangleNum * sizeof(int));

	checkCudaErrors(cudaMemcpy(h_p, triPairNumBuffer.devPtr, sizeof(int)* m_triangleNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_pp, triPairNumPrefixSumBuffer.devPtr, sizeof(int)* m_triangleNum, cudaMemcpyDeviceToHost));

	FILE *f = fopen("E:\\Results\\triNum.txt", "w+");
	for (int i = 0; i < m_triangleNum; i++)
		fprintf(f, "%d %d\n", h_p[i], h_pp[i]);
	fclose(f);
	free(h_p);
	free(h_pp);

	printf("trinum end\n");
}