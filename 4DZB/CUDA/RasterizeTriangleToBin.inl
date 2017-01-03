

// cache bin range to  triBinRangeBuffer 
__global__ void countPairNumForPerTri_kernel()
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// input;
	bool enable_backfaceCull = c_fdStaticParams.enable_backfaceCull;
	int tNum = c_fdStaticParams.numTris;
	float3 upRightPos = c_fdStaticParams.light.upRightCornerPosition;
	float3 downLeftPos = c_fdStaticParams.light.downLeftCornerPosition;
	float3 upLeftPos = c_fdStaticParams.light.upLeftCornerPosition;
	float3 downRightPos = c_fdStaticParams.light.downRightCornerPosition;
	int binWidth = c_fdStaticParams.widthBins;
	int binWidth_LOG2 = c_fdStaticParams.widthBins_LOG2;
	int binHeight = c_fdStaticParams.heightBins;
	float3 lightPlaneB = c_fdLightPlaneParams.begin;
	float3 lightPlaneE = c_fdLightPlaneParams.end;
	float2 lightPlaneFactor = c_fdLightPlaneParams.factor;
	cudaTextureObject_t triangleAABBTex = c_fdStaticParams.triangleAABBTex;
	cudaTextureObject_t triangleVertexTex = c_fdStaticParams.triangleVertexTex;
	cudaTextureObject_t binSampleMaxZTex = c_fdStaticParams.binSampleMaxZTex;
	
	// output;
	struct BinRange * myBinRange = ((struct BinRange*)c_fdStaticParams.triBinRangeBuffer) + tid;
	int * myBinNumPos = ((int *)c_fdStaticParams.triBinNum) + tid;

	if (tid >= tNum)
		return;
	float4 p[3];
	float4 AABBb, AABBe;

	AABBb = tex1Dfetch<float4>(triangleAABBTex, tid << 1 | 0);
	AABBe = tex1Dfetch<float4>(triangleAABBTex, tid << 1 | 1);

	int result = 0;
	if (AABBb.x > lightPlaneE.x || AABBb.y > lightPlaneE.y || AABBe.x < lightPlaneB.x || AABBe.y < lightPlaneB.y) // AABB全部在lightPlane外部。
	{
		// 三角形被剔除。
	}
	else{
		p[0] = tex1Dfetch<float4>(triangleVertexTex, tid * 3);
		p[1] = tex1Dfetch<float4>(triangleVertexTex, tid * 3 + 1);
		p[2] = tex1Dfetch<float4>(triangleVertexTex, tid * 3 + 2);
		float3 e0 = make_float3(p[1]) - make_float3(p[0]);
		float3 e1 = make_float3(p[0]) - make_float3(p[2]);
		float3 n = cross(e1, e0);

		//do backface cull
		if (enable_backfaceCull &&
			dot(n, upRightPos - make_float3(p[0])) < 0 &&
			dot(n, upRightPos - make_float3(p[1])) < 0 &&
			dot(n, upRightPos - make_float3(p[2])) < 0 &&
			dot(n, upLeftPos - make_float3(p[0])) < 0 &&
			dot(n, upLeftPos - make_float3(p[1])) < 0 &&
			dot(n, upLeftPos - make_float3(p[2])) < 0 &&
			dot(n, downRightPos - make_float3(p[0])) < 0 &&
			dot(n, downRightPos - make_float3(p[1])) < 0 &&
			dot(n, downRightPos - make_float3(p[2])) < 0 &&
			dot(n, downLeftPos - make_float3(p[0])) < 0 &&
			dot(n, downLeftPos - make_float3(p[1])) < 0 &&
			dot(n, downLeftPos - make_float3(p[2])) < 0
			)
		{
			// 三角形相对面光源是反面被剔除掉。
		}
		else{
			AABBb.x = fmaxf(AABBb.x, lightPlaneB.x);
			AABBb.y = fmaxf(AABBb.y, lightPlaneB.y);
			AABBe.x = fminf(AABBe.x, lightPlaneE.x);
			AABBe.y = fminf(AABBe.y, lightPlaneE.y);

			ushort2 binB, binE;
			binB.x = fmaxf(floorf((AABBb.x - lightPlaneB.x) * lightPlaneFactor.x), 0.0f);
			binB.y = fmaxf(floorf((AABBb.y - lightPlaneB.y) * lightPlaneFactor.y), 0.0f);
			binE.x = fminf(ceilf((AABBe.x - lightPlaneB.x) * lightPlaneFactor.x), binWidth);
			binE.y = fminf(ceilf((AABBe.y - lightPlaneB.y) * lightPlaneFactor.y), binHeight);
			// write bin range
			*myBinRange = BinRange{ binB, binE, AABBb.z, AABBe.z };
			
			for (int j = binB.y; j < binE.y; j++)
			{
				for (int i = binB.x; i < binE.x; i++)
				{
					float BinsampleMaxZ = tex1Dfetch<float>(binSampleMaxZTex, j << binWidth_LOG2 | i);
					
					if (BinsampleMaxZ >= AABBb.z)
						result++;
				}
			}

		}

	}

	//write out;
	*myBinNumPos = result;

}
void countTriBinPairNum()
{
	// count per tri's bin num
	{
		dim3 block(512);
		dim3 grid(iDiviUp(m_triangleNum, block.x));
		countPairNumForPerTri_kernel << <grid, block >> >();
	}
	// count prefixSum;
	thrust::device_ptr<int> dev_pairNumPrefixsum((int *)triPairNumPrefixSumBuffer.devPtr);
	thrust::device_ptr<int> dev_pairNum((int *)triPairNumBuffer.devPtr);
	thrust::exclusive_scan(dev_pairNum, dev_pairNum + m_triangleNum, dev_pairNumPrefixsum);
	// load to host;
	int pairNumALL;
	int lastTriPairNum;
	checkCudaErrors(cudaMemcpy(&pairNumALL, ((int *)triPairNumPrefixSumBuffer.devPtr) + m_triangleNum - 1, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&lastTriPairNum, ((int *)triPairNumBuffer.devPtr) + m_triangleNum - 1, sizeof(int), cudaMemcpyDeviceToHost));

	pairNumALL += lastTriPairNum;

	// alloc buffers;
	if (binTriPairBinBuffer.size_in_element < pairNumALL)
	{
		binTriPairBinBuffer.freeBuffer();
		binTriPairBinBuffer.size_in_element = pairNumALL;
		binTriPairBinBuffer.allocBuffer();

		binTriPairTriBuffer.freeBuffer();
		binTriPairTriBuffer.size_in_element = pairNumALL;
		binTriPairTriBuffer.allocBuffer();
	}
	my_debug(MY_DEBUG_SECTION_RASTER, 1)("counte tri pair num done, pair num is %.2fMillion\n", (float)pairNumALL/ 1024.0f/1024.0f);
}

__global__ void writeTriBinPair_kernel()
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// input;
	struct BinRange  myBinRange = ((struct BinRange*)c_fdStaticParams.triBinRangeBuffer)[tid];
	int tNum = c_fdStaticParams.numTris;
	int binWidth_LOG2 = c_fdStaticParams.widthBins_LOG2;
	cudaTextureObject_t binSampleMaxZTex = c_fdStaticParams.binSampleMaxZTex;

	if (tid >= tNum)
		return;
	int myFirstPos = ((int *)c_fdStaticParams.triBinNumPrefixSum)[tid];

	// output;
	int * binPos = (int *)c_fdDynamicParams.bt_pair_bin + myFirstPos;
	int * triPos = (int *)c_fdDynamicParams.bt_pair_tri + myFirstPos;

	int cnt = 0;
	for (int j = myBinRange.start.y; j < myBinRange.end.y; j++)
	{
		for (int i = myBinRange.start.x; i < myBinRange.end.x; i++)
		{
			float BinsampleMaxZ = tex1Dfetch<float>(binSampleMaxZTex, j << binWidth_LOG2 | i);

			if (BinsampleMaxZ >= myBinRange.minZ)
			{
				binPos[cnt] = j<<binWidth_LOG2|i;
				triPos[cnt] = tid;
				cnt++;
			}

		}
	}

}
void bindTriToBin()
{
	// write pair
	{
		dim3 block(512);
		dim3 grid(iDiviUp(m_triangleNum, block.x));
		writeTriBinPair_kernel << <grid, block >> >();
	}
	// sort pair
	thrust::device_ptr<int> dev_tri((int*)binTriPairTriBuffer.devPtr);
	thrust::device_ptr<int> dev_bin((int *)binTriPairBinBuffer.devPtr);
	thrust::sort_by_key(dev_bin, dev_bin + binTriPairBinBuffer.size_in_element, dev_tri);

	//find start;
	findTriangleStartEnd();
	//

	my_debug(MY_DEBUG_SECTION_RASTER, 1)("bind tri to bin done!\n");
	saveBinImage((int *)binTriStartBuffer.devPtr, (int *)binTriEndBuffer.devPtr, (int *)binTriPairTriBuffer.devPtr, binTriPairTriBuffer.size_in_element, "binTri");
}
