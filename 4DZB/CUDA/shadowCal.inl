template <typename T>
__device__ __forceinline int CounteNumof1(const T& a);

template<>
__device__ __forceinline int CounteNumof1<unsigned int>(const unsigned int& a){ return __popc(a); };

template<>
__device__ __forceinline int CounteNumof1<unsigned long long>(const unsigned long long &a) { return __popcll(a); }


template <typename T>
__device__ __forceinline int FFS(const T&a);

template<>
__device__ __forceinline int FFS<unsigned int>(const unsigned int&a){ return __ffs(a); }

template<>
__device__ __forceinline int FFS<unsigned long long>(const unsigned long long &a) { return __ffsll(a); }


template <typename T>
__device__ __forceinline T butterflyReduceOr(const T a);

template<>
__device__ __forceinline unsigned int butterflyReduceOr<unsigned int>(const unsigned int a)
{
	U32 v = a;

	for (int i = 16; i >= 1; i /= 2)
		v |= __shfl_xor(v, i, 32);

	return v;
}

template<>
__device__ __forceinline unsigned long long butterflyReduceOr<unsigned long long >(const unsigned long long a)
{
	U32 vl, vh;
	vl = a &(((U64)1 << 32) - 1);
	vh = a >> 32;

	for (int i = 16; i >= 1; i /= 2)
	{
		vl |= __shfl_xor(vl, i, 32);
		vh |= __shfl_xor(vh, i, 32);
	}

	return ((U64)vh << 32) | (U64)vl;

}

template<typename T>
__device__ __forceinline void writeLightBitToShareMem(T * const addr, const int pos, const int flagPos, const T & data, const T& FullRowBit)
{
	atomicOr(&(addr[pos]), data);
	if (addr[pos] == FullRowBit)
		atomicOr(&addr[flagPos], (T)1 << pos);
	//printf("%d %d write\n", threadIdx.y, threadIdx.x);
}

__device__ __forceinline void loadTriangle(float3 &v0, float3 &v1, float3 &v2, float3& AABBb, float3 &AABBe, int tid)
{

	//input ;
	const cudaTextureObject_t triangleVertexTex = c_fdStaticParams.triangleVertexTex;
	const cudaTextureObject_t triangleAABBTex = c_fdStaticParams.triangleAABBTex;
	
	float4 temp;
	temp = tex1Dfetch<float4>(triangleVertexTex, tid * 3);
	v0 = make_float3(temp);
	temp = tex1Dfetch<float4>(triangleVertexTex, tid * 3 + 1);
	v1 = make_float3(temp);
	temp = tex1Dfetch<float4>(triangleVertexTex, tid * 3 + 2);
	v2 = make_float3(temp);

	temp = tex1Dfetch<float4>(triangleAABBTex, tid << 1);
	AABBb = make_float3(temp);
	temp = tex1Dfetch<float4>(triangleAABBTex, tid << 1 | 1);
	AABBe = make_float3(temp);
}

template<typename T>
__device__ __forceinline T calLightHeightRange(float A, float B, float C)
{
	const int lightResWidth = c_fdStaticParams.lightResWidth;
	const int lightResHeight = c_fdStaticParams.lightResHeight;

	/*T a;
	float D;
	int i, j;

	if (B > 0)
		D = B* lightResWidth + C;
	else
		D = C;

	if (D == 0)
		a = 0;
	else{
		i = ceilf(-1.0f * D / A);
		i = min(i, lightResHeight);
		i = max(0, i);
		a = ((T)1 << i) - 1;
	}

	if (A >= 0)
		return ~a;
	
	return a;*/

	if (A <= 0 && B <= 0 && C <= 0)
		return 0;
	float D = max(B*lightResWidth + C, C);

	if (A <= 0 && D <= 0) return 0;
	
	if (A == 0) thrust::swap(A, D);
	if (D == 0) return A > 0 ? ((T)1 << lightResHeight) - 1 : 0;

	T a = ((T)1 << (int)fmaxf(fminf(ceilf(-1 * D / A), lightResHeight), 0)) -1;

	return A < 0 ? a : ~a;
}

template<typename T>
__device__ __forceinline T calLightRowRange(float B, float D)
{
	//T a;
	//int j;
	const int lightResWidth = c_fdStaticParams.lightResWidth;

	/*if (D == 0)
	{
		a = 0;
	}
	else{
		j = ceilf(-1.0f * D / B);
		j = min(j, lightResWidth);
		j = max(0, j);
		a = ((T)1 << j) - 1;
	}

	if (B >= 0)
		return ~a;
	return a;*/

	if (B <= 0 && D <= 0)
		return 0;

	if (B == 0) thrust::swap(B, D);
	if (D == 0) return B > 0 ? ((T)1 << lightResWidth) - 1 : 0;

	T a = ((T)1 << (int)fmaxf(0, fminf(lightResWidth, ceilf(-1 * D / B)))) - 1;
	return B < 0 ? a : ~a;
}

template<typename T>
__device__ __forceinline T calLightRowShadowWithOutDepth(int i, float A0, float A1, float A2, float B0, float B1, float B2, float C0, float C1, float C2)
{
	const int lightResWidth = c_fdStaticParams.lightResWidth;
	const T   lightWidthBit = ((T)1 << lightResWidth) - 1;
	T light_bit = lightWidthBit;
	float D;

	D = A0 * i + C0;
	light_bit &= calLightRowRange<T>(B0, D);

	D = A1 * i + C1;
	light_bit &= calLightRowRange<T>(B1, D);

	D = A2 * i + C2;
	light_bit &= calLightRowRange<T>(B2, D);

	/*D = A3 * i + C3;
	light_bit &= calLightRowRange(B3, D);*/

	light_bit &= lightWidthBit;
	return light_bit;
}

template<typename T>
__device__ __forceinline T calLightRowShadowWithDepth(int i, float A0, float A1, float A2, float A3, float B0, float B1, float B2, float B3, float C0, float C1, float C2, float C3)
{
	const int lightResWidth = c_fdStaticParams.lightResWidth;
	const T   lightWidthBit = ((T)1 << lightResWidth) - 1;

	T light_bit = calLightRowShadowWithOutDepth<T>(i, A0, A1, A2, B0, B1, B2, C0, C1, C2);

	float D = A3 * i + C3;
	light_bit &= calLightRowRange<T>(B3, D);

	light_bit &= lightWidthBit;

	return light_bit;
}

template<typename T, int maxLightResHeight, int sampleBitMaskBufferNum>
__global__ void shadowCal_CTA_kernel(int validBinNum)
{
	// input;
	int * const binTriStart = (int *)c_fdStaticParams.binTriStart;
	int * const binTriEnd = (int *)c_fdStaticParams.binTriEnd;
	int * const pair_tri = (int *)c_fdDynamicParams.bt_pair_tri;

	int * const binSampleStart = (int *)c_fdStaticParams.binPixelStart;
	int * const binSampleEnd = (int*)c_fdStaticParams.binPixelEnd;
	int * const pair_sample = (int *)c_fdStaticParams.pb_pair_pixel;

	const int	viewportWidth_LOG2_UP = c_fdStaticParams.viewportWidth_LOG2_UP;
	const int	viewportWidth = c_fdStaticParams.viewportWidth;

	int * const validBin = (int *)c_fdStaticParams.validBinBuffer;

	const int lightResWidth = c_fdStaticParams.lightResWidth;
	const int lightResHeight = c_fdStaticParams.lightResHeight;
	const T   lightWidthBit = ((T)1 << lightResWidth) - 1;
	const T	  lightHeightBit = ((T)1 << lightResHeight) - 1;
	const int lightResNum = c_fdStaticParams.lightResNum;

	const cudaSurfaceObject_t sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;

	//light
	const float3 upRightCornerPosition = c_fdStaticParams.light.upRightCornerPosition;
	const float3 x_delt = c_fdStaticParams.light.x_delt;
	const float3 y_delt = c_fdStaticParams.light.y_delt;
	// output;
	float * const shadowResult = (float*)c_fdStaticParams.shadowValue;

	// shared memory; 
	////////////////////////////////////////////////////////////////////////////////////////
	// bit mask
	//32*32 提供256个sample，use 42KB; 64*64 提供76个sample，use 41.004KB；
	////////////////////////////////////////////////////////////////////////////////////////
	__shared__			T		s_sampleBitMask		[sampleBitMaskBufferNum][maxLightResHeight+1]; // +1 for test bitMask is all full; 
	__shared__ volatile float	s_samplePosition	[3]						  [sampleBitMaskBufferNum]; // for avoid bank collisions;
	__shared__ volatile float	s_sampleRectangle	[4]						  [sampleBitMaskBufferNum];
	__shared__ volatile float	s_sampleZ			[sampleBitMaskBufferNum];
	__shared__ volatile int		s_sampleWritePos	[sampleBitMaskBufferNum];
	__shared__ volatile int		s_sampleBitMaskNum;

	__shared__ volatile int		s_ourBin;
	__shared__ volatile int*	s_ourTri;
	__shared__ volatile int		s_ourTriNum;
	__shared__ volatile int*	s_ourSample;
	__shared__ volatile int		s_ourSampleNum;
	//

	int threadId = threadIdx.y * blockDim.x + threadIdx.x;
	for (;;)
	{
		__syncthreads();
		//ask a bin;
		if (threadId == 0)
		{
			int idx= atomicAdd(&g_fdAtomics.shadowComputerCounter, 1);
			if (idx > validBinNum)
				s_ourBin = -1;
			else
			{
				s_ourBin = validBin[idx];
				int triStart, triEnd;
				triStart = binTriStart[s_ourBin];
				triEnd = binTriEnd[s_ourBin];
				s_ourTriNum = triEnd - triStart;
				s_ourTri = pair_tri + triStart;

				int sampleStart, sampleEnd;
				sampleStart = binSampleStart[s_ourBin];
				sampleEnd = binSampleEnd[s_ourBin];
				s_ourSampleNum = sampleEnd - sampleStart;
				s_ourSample = pair_sample + sampleStart;
				//printf("CTA %d cal bin %d\n", blockIdx.x, s_ourBin);
			}
		}
		__syncthreads();

		// no more bins;
		if (s_ourBin == -1)
			break;


		for (int sampleCounter = 0; sampleCounter < s_ourSampleNum; sampleCounter += sampleBitMaskBufferNum)
		{
			__syncthreads();
			// load samples with block syncthreads;
			if (threadId < sampleBitMaskBufferNum && sampleCounter + threadId < s_ourSampleNum)
			{
				if (threadId == 0)
					s_sampleBitMaskNum = min(sampleBitMaskBufferNum, s_ourSampleNum - sampleCounter);

				int sampleID = s_ourSample[sampleCounter + threadId];
				

				int sampleX = sampleID & ((1 << viewportWidth_LOG2_UP) - 1);
				int sampleY = sampleID >> viewportWidth_LOG2_UP;

				s_sampleWritePos[threadId] = sampleY * viewportWidth + sampleX;

				// 
				float4 temp;
				temp = tex2D(samplePositionTex, sampleX, sampleY);
				s_samplePosition[0][threadId] = temp.x;
				s_samplePosition[1][threadId] = temp.y;
				s_samplePosition[2][threadId] = temp.z;

				temp = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 0, cudaBoundaryModeTrap);
				s_sampleRectangle[0][threadId] = temp.x;
				s_sampleRectangle[1][threadId] = temp.y;

				temp = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 1, cudaBoundaryModeTrap);
				s_sampleRectangle[2][threadId] = temp.x;
				s_sampleRectangle[3][threadId] = temp.y;

				s_sampleZ[threadId] = temp.z;

				// we can load previous mask?
				for (int i = 0; i <= lightResHeight; i++) // lightResHeight + 1 
					s_sampleBitMask[threadId][i] = (T)0;
			}
			__syncthreads();

			// load triangle, in warps, no sysncthreads;
			for (int triangleCounter = 0; triangleCounter + threadIdx.y * blockDim.x < s_ourTriNum; triangleCounter += blockDim.x * blockDim.y)
			{
				// cal
				// load my triangle;
				int mytriangleID = -1;
				if (triangleCounter + threadId < s_ourTriNum)
				{
					mytriangleID = s_ourTri[triangleCounter + threadId];

				}

				
				float3 v0 = make_float3(0, 0, 0);
				float3 v1 = make_float3(0, 0, 0);
				float3 v2 = make_float3(0, 0, 0);

				float3 AABBb = make_float3(FD_F32_MAX, FD_F32_MAX, FD_F32_MAX);
				float3 AABBe = make_float3(-FD_F32_MAX, -FD_F32_MAX, -FD_F32_MAX);

				// triangle 缓存 占用 33个寄存器
				float3 c_v0_a = make_float3(0, 0, 0);
				float3 c_v0_b = make_float3(0, 0, 0);
				float3 c_v0_c = make_float3(0, 0, 0);

				float3 c_v1_a = make_float3(0, 0, 0);
				float3 c_v1_b = make_float3(0, 0, 0);
				float3 c_v1_c = make_float3(0, 0, 0);
			
				float3 c_v2_a = make_float3(0, 0, 0);
				float3 c_v2_b = make_float3(0, 0, 0);
				float3 c_v2_c = make_float3(0, 0, 0);

				// for depth 
				float A3_o = 0;
				float B3_o = 0;
				float C3_o = 0;

				float3 n = make_float3(0, 0, 0);
				/////////////////////////////


				if (mytriangleID != -1)
				{
					loadTriangle(v0, v1, v2, AABBb, AABBe, mytriangleID);

					// make triangle cache;
					float3 v0_v1 = v0 - v1;
					float3 v1_v2 = v1 - v2;
					float3 v2_v0 = v2 - v0;

					float3 L_v1 = upRightCornerPosition - v1;
					c_v1_a = cross(v0_v1, y_delt);
					c_v1_b = cross(v0_v1, x_delt);
					c_v1_c = cross(v0_v1, L_v1);

					float3 L_v2 = upRightCornerPosition - v2;
					c_v2_a = cross(v1_v2, y_delt);
					c_v2_b = cross(v1_v2, x_delt);
					c_v2_c = cross(v1_v2, L_v2);

					float3 L_v0 = upRightCornerPosition - v0;
					c_v0_a = cross(v2_v0, y_delt);
					c_v0_b = cross(v2_v0, x_delt);
					c_v0_c = cross(v2_v0, L_v0);

					n = cross(v2_v0, v1_v2);

					A3_o = dot(n, y_delt);
					B3_o = dot(n, x_delt);
					C3_o = dot(n, L_v0);
					

				}
				if (__any(mytriangleID != -1))
				{
					// 为了减少写入share memory 时 aotmicOr 冲突， 每个warp 开始的位置都不同。
					int sampleIdx = min(threadIdx.y, s_sampleBitMaskNum - 1); // at least one sample
					for (int cnt = 0; cnt < s_sampleBitMaskNum; cnt++, sampleIdx = (sampleIdx + 1) % s_sampleBitMaskNum)
					{
						if (s_sampleBitMask[sampleIdx][lightResHeight] != lightHeightBit &&        // this sample is not in all light 's shadow
							__any(AABBb.x <= s_sampleRectangle[0][sampleIdx] && AABBb.y <= s_sampleRectangle[1][sampleIdx] &&
							s_sampleRectangle[2][sampleIdx] <= AABBe.x && s_sampleRectangle[3][sampleIdx] <= AABBe.y && // triangle AABB covered all rectangle
							AABBb.z <= s_sampleZ[sampleIdx]))													// triangle minZ  < sample Z  有一部分三角形位于灯和sample之间。
						{
							
							/// cal shadow volume params A, B, C;
							float A0, B0, C0, A1, B1, C1, A2, B2, C2, A3, B3, C3;
							float3 S = make_float3(s_samplePosition[0][sampleIdx], s_samplePosition[1][sampleIdx], s_samplePosition[2][sampleIdx]);

							float3 s_v1 = S - v1;
							A0 = dot(c_v1_a, s_v1);
							B0 = dot(c_v1_b, s_v1);
							C0 = dot(c_v1_c, s_v1);

							float3 s_v2 = S - v2;
							A1 = dot(c_v2_a, s_v2);
							B1 = dot(c_v2_b, s_v2);
							C1 = dot(c_v2_c, s_v2);

							float3 s_v0 = S - v0;
							A2 = dot(c_v0_a, s_v0);
							B2 = dot(c_v0_b, s_v0);
							C2 = dot(c_v0_c, s_v0);

							A3 = A3_o;
							B3 = B3_o;
							C3 = C3_o;

							float temp = dot(n, s_v0);
							T initHeightBit = lightHeightBit;
							// 目的： 消除S所在三角形挡住S的情况。
							// 这里会存在误差：
							// optix 中使用S到三角形交点处的距离做判断。
							//因为我们没有办法对每一条光线都算出S到交点距离，只能使用S到三角形所在平面的距离代替，（我们使用直角三角形的直角边代替了斜边）我们得到的距离比optix要小。
							if (fabsf(temp) < 1e-6)
								initHeightBit = 0;
							if ( temp > 0)
							{
								A3 *= -1.0f;
								B3 *= -1.0f;
								C3 *= -1.0f;
							}
							
							
							// find light height range; 
							T light_bit_y = initHeightBit;
							
							light_bit_y &= calLightHeightRange<T>(A0, B0, C0);
							light_bit_y &= lightHeightBit;
							if (__all(light_bit_y == 0)) // no light ;
								continue;

							light_bit_y &= calLightHeightRange<T>(A1, B1, C1);
							light_bit_y &= lightHeightBit;
							if (__all(light_bit_y == 0)) // no light ;
								continue;

							light_bit_y &= calLightHeightRange<T>(A2, B2, C2);
							light_bit_y &= lightHeightBit;
							if (__all(light_bit_y == 0)) // no light ;
								continue;
							
							light_bit_y &= calLightHeightRange<T>(A3, B3, C3);
							light_bit_y &= lightHeightBit;
							if (__all(light_bit_y == 0)) // no light ;
								continue;

							if (__all((light_bit_y | s_sampleBitMask[sampleIdx][lightResHeight]) == s_sampleBitMask[sampleIdx][lightResHeight])) // all lights triangles affected are already in shadow;
								continue;



							
							// reduce all range ;
							int myHeightSize = CounteNumof1<T>(light_bit_y);
							int warpHeightSize = 1<<20; // 当只有一个三角形时， 标识不使用合并求解。

							T warpHeightRange = 0;
							if (__popc(__ballot(light_bit_y != 0)) > 1) // more than one triangle affect this sample
							{
								warpHeightRange = butterflyReduceOr<T>(light_bit_y);
								warpHeightSize = CounteNumof1<T>(warpHeightRange);
							}
							// if range are closer; cal together; use one atomicOr;
							if (__any(warpHeightSize * 4 <=  myHeightSize *8) )
							{
								if (__all(AABBe.z < s_sampleZ[sampleIdx] || light_bit_y == 0))// not need depth test
								{
									int i = max(FFS<T>(warpHeightRange)-1, 0);
									warpHeightRange >>= i;
									for (; warpHeightRange; warpHeightRange >>= 1, i++)
									{
										// this row is not in range or this row is all in shadow; skip
										if (warpHeightRange & 0x01 == 0 || s_sampleBitMask[sampleIdx][i] == lightWidthBit)
											continue;

										T light_bit = (light_bit_y & (T)1<<i)==0?0:  calLightRowShadowWithOutDepth<T>(i, A0, A1, A2, B0, B1, B2, C0, C1, C2);

										// reduce all
										light_bit = butterflyReduceOr<T>(light_bit);
										if (threadIdx.x == 0)
										{
											writeLightBitToShareMem<T>(&(s_sampleBitMask[sampleIdx][0]), i, lightResHeight, light_bit, lightWidthBit);
										}
									}

								}
								else{ // need depth test;
									int i = max(FFS<T>(warpHeightRange)-1, 0);
									warpHeightRange >>= i;
									for (; warpHeightRange; warpHeightRange >>= 1, i++)
									{
										// this row is not in range or this row is all in shadow; skip
										if (warpHeightRange & 0x01 == 0 || s_sampleBitMask[sampleIdx][i] == lightWidthBit)
											continue;

										T light_bit = (light_bit_y & (T)1 << i) == 0 ? 0 : calLightRowShadowWithDepth<T>(i, A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3);

										// reduce all
										light_bit = butterflyReduceOr<T>(light_bit);
										if (threadIdx.x == 0)
										{
											writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx][0], i, lightResHeight, light_bit, lightWidthBit);
										}
									}
								}
							}
							else// else cal respective; 
							{// <!>for simple just else;

								// for depth test;
								if (__all(AABBe.z < s_sampleZ[sampleIdx] || light_bit_y == 0))// not need depth test
								{
									if (light_bit_y != 0)
									{
										int i = max(FFS<T>(light_bit_y)-1, 0);
										light_bit_y >>= i;
										for (; light_bit_y; light_bit_y >>= 1, i++) // cal all light;
										{
											// this row is all in shadow; skip
											if (s_sampleBitMask[sampleIdx][i] == lightWidthBit)
												continue;
											T light_bit = calLightRowShadowWithOutDepth<T>(i, A0, A1, A2, B0, B1, B2, C0, C1, C2);

											/*T light_bit = 0;
											for (int j = 0; j < lightResWidth; j++)
												if (A0*i + B0*j + C0 > 0 && A1*i + B1*j + C1 > 0 && A2*i + B2*j + C2 > 0)
													light_bit |= 1 << j;*/

											writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx][0], i, lightResHeight, light_bit, lightWidthBit);
										}
									}
								}
								else{// do depth test;
									if (light_bit_y != 0)
									{
										int i = max(FFS<T>(light_bit_y)-1, 0);
										light_bit_y >>= i;
										for (; light_bit_y; light_bit_y >>= 1, i++) // cal all light;
										{
											// this row is all in shadow; skip
											if (s_sampleBitMask[sampleIdx][i] == lightWidthBit)
												continue;

											T light_bit = calLightRowShadowWithDepth<T>(i, A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3);

											/*T light_bit = 0;
											for (int j = 0; j < lightResWidth; j++)
												if (A0*i + B0*j + C0 > 0 && A1*i + B1*j + C1 > 0 && A2*i + B2*j + C2 > 0 && A3*i + B3*j + C3 > 0)
													light_bit |= 1 << j;*/

											writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx][0], i, lightResHeight, light_bit, lightWidthBit);

											// more row lights;
										}
									}
									//
								}

								//  
							}
							
			
							/*
							for (int i = 0; i < lightResHeight; i++)
							{
								if (s_sampleBitMask[sampleIdx][lightResHeight] & (1 << i))
									continue;
								for (int j = 0; j < lightResWidth; j++)
								{
									float3 S = make_float3(s_samplePosition[0][sampleIdx], s_samplePosition[1][sampleIdx], s_samplePosition[2][sampleIdx]);
									if (s_sampleBitMask[sampleIdx][i] & 1 << j)
										continue;
									float3 L = c_fdStaticParams.light.lightPos[i*lightResWidth + j] - S;
									float dist = sqrt(dot(L, L));
									float3 rayDirection = L / dist;
									float  rayTmin = 1e-3;

									if (intersect_triangle_branchless(rayDirection, S, rayTmin, dist, v0, v1, v2))
									{
										writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx][0], i, lightResHeight, (T)1 << j, lightWidthBit);
									}
								}
							}
							*/
					


							// end of need to cal;
						}
						
						// more sample need to cal;
					}
					// end of have triangle;
				}

				// more triangle
			}

			__syncthreads();
			// store sampleMask to global;
			if (threadId < sampleBitMaskBufferNum && sampleCounter + threadId < s_ourSampleNum)
			{
				// for simple we just computer shadow once;
				int result = 0;
				for (int i = 0; i < lightResHeight; i++)
				{
					result += CounteNumof1<T>(s_sampleBitMask[threadId][i]);
				}
				// write out
				shadowResult[s_sampleWritePos[threadId]] = (float)result / (float)lightResNum;
			}
			// more samples

		}
		//end of this bin


	}// loop forever

}
void shadowCal()
{
	prepareBin();
	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("shadow cal start!\n");
	// clear global counter
	checkCudaErrors(cudaMemset(g_fdAtomics_addr, 0, sizeof(FDAtomics)));

	// cal
	{
		dim3 block(32, 10); // <!> register limit warps; threads num >= sampleBitMaskBufferNum
		dim3 grid(m_CTA_num); // <share memory per block> <share memory per SM>and <SM num> limit;
		if (m_lightResWidth <= 32 && m_lightResHeight <= 32){
			// 32*32
			shadowCal_CTA_kernel<uint, 32, 256> << <grid, block >> >(m_validBinNum);
		}
		else{
			// 64*64
			if (m_lightResWidth <= 64 && m_lightResHeight <= 64)
				shadowCal_CTA_kernel<U64, 64, 76> << <grid, block >> >(m_validBinNum);
			else{
				// lightRes too large;
			}
		}
			
	}
	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("shadow cal end!\n");

}