
#if _DEBUG
#define __AddCounter(counter) atomicAdd(& counter, (U64)1); 

#else
#define __AddCounter(counter) ;
#endif





/////////////////////////////////////////////////////////////////////////////////////////////
// __popc():
// input: unsigned int;
// output: number of 1s in input interger;
/////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline int CountNumof1(volatile const T& a);

template<>
__device__ __forceinline int CountNumof1<unsigned int>(volatile const unsigned int& a){ return __popc(a); };

template<>
__device__ __forceinline int CountNumof1<unsigned long long>(volatile const unsigned long long &a) { return __popcll(a); }


/////////////////////////////////////////////////////////////////////////////////////////////
// __ffs:
//    输入:  一个unsigned int， 
//	  输出： 这个整数最低位的1的位置，范围[1~32]
//	  特殊情况： 输入0， 输出0；
/////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline int FFS(const T&a);

template<>
__device__ __forceinline int FFS<unsigned int>(const unsigned int&a){ return __ffs(a); }

template<>
__device__ __forceinline int FFS<unsigned long long>(const unsigned long long &a) { return __ffsll(a); }

/////////////////////////////////////////////////////////////////////////////////////////////
// btterflyReduceOr
// 描述： warp 内线程间进行Or操作，循环5次后，每个线程会得到与其他31个线程中寄存器求Or后的结果；
/////////////////////////////////////////////////////////////////////////////////////////////
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
__device__ __forceinline T getLightWidthBit();

template<>
__device__ __forceinline U32 getLightWidthBit<U32>() { return c_fdStaticParams.light.U32lightWidthBit; }

template<>
__device__ __forceinline U64 getLightWidthBit<U64>() { return c_fdStaticParams.light.U64lightWidthBit; }

//template<typename T>
//__device__ __forceinline T getLightHeightBit();
//
//template<>
//__device__ __forceinline U32 getLightHeightBit<U32>() { return c_fdStaticParams.light.U32lightHeightBit; }
//
//template<>
//__device__ __forceinline U64 getLightHeightBit<U64>() { return c_fdStaticParams.light.U64lightHeightBit; }
//

/////////////////////////////////////////////////////////////
// return dot(cross(a, b), c);
/////////////////////////////////////////////////////////////
__device__ __forceinline float dotCross(const float3& a, const float3& b, const float3 &c)
{

	float A1 = a.y*b.z, A2 = a.z * b.y;
	float B1 = a.z *b.x, B2 = a.x*b.z;
	float C1 = a.x*b.y, C2 = a.y*b.x;

	A1 = (A1 - A2) * c.x;
	B1 = (B1 - B2) * c.y;
	C1 = (C1 - C2) * c.z;
	return A1 + B1 + C1;
}

__device__ __forceinline void makeShadowVolume(
	float & A0, float & B0, float& C0, 
	float & A1, float & B1, float& C1, 
	float & A2, float & B2, float& C2, 
	float & dist,
	const float3& v0, const float3& v1, const float3& v2, const float3& n, const float3 &S)
{

	//light
	const float3 upRightCornerPosition = c_fdStaticParams.light.upRightCornerPosition;
	const float3 x_delt = c_fdStaticParams.light.x_delt;
	const float3 y_delt = c_fdStaticParams.light.y_delt;

	
	float3 v0_v1 = v0 - v1;
	float3 s_v1 = S - v1;
	float3 L_v1 = upRightCornerPosition - v1;

	A0 = dotCross(v0_v1, y_delt, s_v1);
	B0 = dotCross(v0_v1, x_delt, s_v1);
	C0 = dotCross(v0_v1, L_v1, s_v1);

	float3 v1_v2 = v1 - v2;
	float3 s_v2 = S - v2;
	float3 L_v2 = upRightCornerPosition - v2;

	A1 = dotCross(v1_v2, y_delt, s_v2);
	B1 = dotCross(v1_v2, x_delt, s_v2);
	C1 = dotCross(v1_v2, L_v2, s_v2);
	
	float3 v2_v0 = v2 - v0;
	float3 s_v0 = S - v0;
	float3 L_v0 = upRightCornerPosition - v0;

	A2 = dotCross(v2_v0, y_delt, s_v0);
	B2 = dotCross(v2_v0, x_delt, s_v0);
	C2 = dotCross(v2_v0, L_v0, s_v0);
	
	dist = dot(n, s_v0);

}

template<typename T>
__device__ __forceinline void writeLightBitToShareMem( volatile T * const addr, const int Row, const T & data)
{
	const int flagPos = c_fdStaticParams.light.lightResHeight;
	const  T  lightWidthFullBit = getLightWidthBit<T>();
	// 由于atomic 中不能使用volatile ；
	T * _addr = const_cast<T*> (addr);

	T val = data | atomicOr(&(_addr[Row]), data);
	if (val == lightWidthFullBit)
		atomicOr(&_addr[flagPos], (T)1 << Row);
	//printf("%d %d write\n", threadIdx.y, threadIdx.x);
}

__device__ __forceinline void loadTriangle(float3 &v0, float3 &v1, float3 &v2, float3& AABBb, float3 &AABBe, int tid)
{

	//input ;
	//const cudaTextureObject_t triangleVertexTex = c_fdStaticParams.triangleVertexTex;
	//const cudaTextureObject_t triangleAABBTex = c_fdStaticParams.triangleAABBTex;

	const cudaTextureObject_t triDataTex = c_fdStaticParams.triangleDataTex;
//	const int triDataSizeFloat4 = sizeof(TriangleData) / sizeof(float4);

	TriangleData data;
	float4 *data_f4 = (float4*)&data;

	data_f4[0] = tex1Dfetch<float4>(triDataTex, tid * 4);
	data_f4[1] = tex1Dfetch<float4>(triDataTex, tid * 4 + 1);
	data_f4[2] = tex1Dfetch<float4>(triDataTex, tid * 4 + 2);
	data_f4[3] = tex1Dfetch<float4>(triDataTex, tid * 4 + 3);

	v0 = data.v0;
	v1 = data.v1;
	v2 = data.v2;
	AABBb.x = data.aabb2Dmin.x;
	AABBb.y = data.aabb2Dmin.y;

	AABBe.x = data.aabb2Dmax.x;
	AABBe.y = data.aabb2Dmax.y;

	AABBb.z = data.minz;
	AABBe.z = data.maxz;

	/*float4 temp;
	temp = tex1Dfetch<float4>(triangleVertexTex, tid * 3);
	v0 = make_float3(temp);
	temp = tex1Dfetch<float4>(triangleVertexTex, tid * 3 + 1);
	v1 = make_float3(temp);
	temp = tex1Dfetch<float4>(triangleVertexTex, tid * 3 + 2);
	v2 = make_float3(temp);

	temp = tex1Dfetch<float4>(triangleAABBTex, tid << 1);
	AABBb = make_float3(temp);
	temp = tex1Dfetch<float4>(triangleAABBTex, tid << 1 | 1);
	AABBe = make_float3(temp);*/

}

__device__ __forceinline bool test3DPointInTriangle(float3 A, float3 B, float3 C, float3 P)
{
	float3 N1 = cross((B - A), (P - B));
	float3 N2 = cross((C - B), (P - C));
	if (dot(N1, N2) < 0.0f) return false;

	float3 N3 = cross((A - C), (P - A));
	if (dot(N2, N3) < 0.0f) return false;
	else return true;
}


template<typename T>
__device__ __forceinline T calLightHeightRange(float A, float B, float C)
{
	//const int lightResWithSubOne = c_fdStaticParams.light.lightResWidthSubOne;

	const int lightResWith = c_fdStaticParams.light.lightResWidth;
	const int lightResHeight = c_fdStaticParams.light.lightResHeight;
	const T	  lightHeightFullBit = (T)(1 << lightResHeight) - 1;//getLightHeightBit<T>();

	/*T a;
	float D;
	int i, j;

	if (B > 0)
		D = B* FD_LIGHT_RES_WIDTH + C;
	else
		D = C;

	if (D == 0)
		a = 0;
	else{
		i = ceilf(-1.0f * D / A);
		i = min(i, FD_LIGHT_RES_HEIGHT);
		i = max(0, i);
		a = ((T)1 << i) - 1;
	}

	if (A >= 0)
		return ~a;
	
	return a;*/

	/*if (C <= 0)
		return 0;
	else
		return 0x1;*/
	
	//if (A <= 0 && B <= 0 && C <= 0)
		//return 0;
	


	//if (A <= 0 && D <= 0) return 0;
	
	//if (A == 0) thrust::swap(A, D);
	//if (D == 0) return A > 0 ? lightHeightFullBit : 0;


	//float D = max(B*(lightResWith - 1) + C, C);
	//if (A <= 0 && D <= 0) return 0;
	////if (A == 0) return D > 0 ? lightHeightFullBit : 0;

	//if (A == 0) thrust::swap(A, D);
	//if (D == 0) return A > 0 ? lightHeightFullBit : 0;

	//T a = ((T)1 << (int)fmaxf(fminf(ceilf(-1 * D / A), lightResHeight), 0)) - 1;

	//return A < 0 ? a : ((~a) & lightHeightFullBit);


	//if (A <= 0 && B <= 0 && C <= 0)
	//	return 0;
	float D = max(B*(lightResWith - 1) + C, C);

	if (A <= 0 && D <= 0) return 0;

	if (A == 0 || D == 0) return lightHeightFullBit;

	//if (A == 0) thrust::swap(A, D);
	//if (D == 0) return A > 0 ? lightHeightFullBit : 0;

	T a = ((T)1 << (int)fmaxf(fminf(ceilf(-1 * D / A), lightResHeight), 0)) - 1;

	return A < 0 ? a : ((~a) & lightHeightFullBit);
	
}

template<typename T>
__device__ __forceinline T calLightRowRange(float B, float D)
{
	const int lightResWidth = c_fdStaticParams.light.lightResWidth;
	const T	  lightWidthFullBit = ((T)1 << lightResWidth) - 1;//getLightWidthBit<T>();

	//T a;
	//int j;
	//const int FD_LIGHT_RES_WIDTH = c_fdStaticParams.FD_LIGHT_RES_WIDTH;

	/*if (D == 0)
	{
		a = 0;
	}
	else{
		j = ceilf(-1.0f * D / B);
		j = min(j, FD_LIGHT_RES_WIDTH);
		j = max(0, j);
		a = ((T)1 << j) - 1;
	}

	if (B >= 0)
		return ~a;
	return a;*/

	//if (B <= 0 && D <= 0)
	//	return 0;

	////if (B == 0) thrust::swap(B, D);
	////if (D == 0) return B > 0 ? lightWidthFullBit : 0;


	////if (D == 0) return B > 0 ? lightWidthFullBit : 0;
	////if (B == 0) return D > 0 ? lightWidthFullBit : 0;

	//


	//T a = ((T)1 << (int)ceilf(fmaxf(0.0f, fminf((float)lightResWidth, -1 * D / B)))) - 1;
	//return B < 0 ? a : ~a;



	if (B <= 0 && D <= 0)
		return 0;

	if (B == 0 || D == 0)
		return lightWidthFullBit;
	//if (B == 0) thrust::swap(B, D);
	//if (D == 0) return B > 0 ? lightWidthFullBit : 0;

	T a = ((T)1 << (int)fmaxf(0, fminf(lightResWidth, ceilf(-1 * D / B)))) - 1;
	return B < 0 ? a : ~a;


}

template<typename T>
__device__ __forceinline T calLightRowShadowWithOutDepth(int i, float A0, float A1, float A2, float B0, float B1, float B2, float C0, float C1, float C2)
{
	const T lightWidthFulBit = getLightWidthBit<T>();


	T light_bit = lightWidthFulBit;
	float D;

	D = A0 * i + C0;
	light_bit &= calLightRowRange<T>(B0, D);

	D = A1 * i + C1;
	light_bit &= calLightRowRange<T>(B1, D);

	D = A2 * i + C2;
	light_bit &= calLightRowRange<T>(B2, D);

	/*D = A3 * i + C3;
	light_bit &= calLightRowRange(B3, D);*/

	light_bit &= lightWidthFulBit;
	return light_bit;
}

template<typename T>
__device__ __forceinline T calLightRowShadowWithDepth(int i, float A0, float A1, float A2, float A3, float B0, float B1, float B2, float B3, float C0, float C1, float C2, float C3)
{
	const T lightWidthFullBit = getLightWidthBit<T>();

	T light_bit = calLightRowShadowWithOutDepth<T>(i, A0, A1, A2, B0, B1, B2, C0, C1, C2);

	float D = A3 * i + C3;
	light_bit &= calLightRowRange<T>(B3, D);

	light_bit &= lightWidthFullBit;

	return light_bit;
}
template<typename T>
__device__ __forceinline void clearBitMask(volatile T * const addr, const int memSizeofT, const int threadNum, const int threadID)
{
	for (int i = threadID; i < memSizeofT; i += threadNum)
		addr[i] = (T)0;
	__syncthreads();
}



template<typename T>
// 这里使用定值进行优化__launch_bounds__ 》》》
__global__ void /*__launch_bounds__(20 * 32, 2)*/ shadowCal_CTA_kernel(const int validBinNum, const int sampleBitMaskBufferNum)
{
	// input;
	int * const binTriStart = (int *)c_fdStaticParams.binTriStart;
	int * const binTriEnd = (int *)c_fdStaticParams.binTriEnd;
	int * const pair_tri = (int *)c_fdStaticParams.bt_pair_tri;

	int * const binSampleStart = (int *)c_fdStaticParams.binPixelStart;
	int * const binSampleEnd = (int*)c_fdStaticParams.binPixelEnd;
	int * const pair_sample = (int *)c_fdStaticParams.pb_pair_pixel;

	const int	viewportWidth_LOG2_UP = c_fdStaticParams.viewportWidth_LOG2_UP;
	const int	viewportWidth = c_fdStaticParams.viewportWidth;

	int * const validBin = (int *)c_fdStaticParams.validBinBuffer;

	const float scenes_esp = c_fdStaticParams.scenes_esp;

	const int lightResWidth = c_fdStaticParams.light.lightResWidth;
	const int lightResHeight = c_fdStaticParams.light.lightResHeight;
	const T   lightWidthFullBit =  getLightWidthBit<T>();//((T)1 << lightResWidth ) - 1;//
	const T	  lightHeightFullBit = ((T)1 << lightResHeight) - 1;// getLightHeightBit<T>();
	const int lightResNum = c_fdStaticParams.light.lightResNum;
	const int lightBitMaskSizeT = c_fdStaticParams.light.lightBitMaskSizeT;

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
	////////////////////////////////////////////////////////////////////////////////////////
	
	//__shared__			T		s_sampleBitMask		[sampleBitMaskBufferNum][maxLightResHeight+1]; // +1 for test bitMask is all full; 
	//__shared__ volatile float	s_samplePosition	[3]						  [sampleBitMaskBufferNum]; // for avoid bank collisions;
	//__shared__ volatile float	s_sampleRectangle	[4]						  [sampleBitMaskBufferNum];
	//__shared__ volatile float	s_sampleZ			[sampleBitMaskBufferNum];
	//__shared__ volatile int		s_sampleWritePos	[sampleBitMaskBufferNum];
	__shared__ volatile int		s_sampleBitMaskNum;

	__shared__ volatile int		s_ourBin;
	__shared__ volatile int*	s_ourTri;
	__shared__ volatile int		s_ourTriNum;
	__shared__ volatile int*	s_ourSample;
	__shared__ volatile int		s_ourSampleNum;
	//

	extern __shared__   int _s[];
	volatile		T		* s_sampleBitMask = (T*)_s;
	volatile	float	* s_samplePositionAndZ = (float*)&s_sampleBitMask[sampleBitMaskBufferNum * lightBitMaskSizeT];
	volatile	float	* s_sampleRectangle = (float*)&s_samplePositionAndZ[sampleBitMaskBufferNum * 4];
//	volatile	float	* s_sampleZ = (float*)& s_sampleRectangle[sampleBitMaskBufferNum*4];
	volatile	int		* s_sampleWritePos = (int*)&s_sampleRectangle[sampleBitMaskBufferNum * 4];


	const int threadNumPerBlock = blockDim.x * blockDim.y;
	const int threadId = threadIdx.y * blockDim.x + threadIdx.x;


	for (;;)
	{
		__syncthreads();
		//ask a bin;
		if (threadId == 0)
		{
			// atomicAdd return old;
			int idx= atomicAdd(&g_fdAtomics.shadowComputerCounter, 1);
			if (idx >= validBinNum)
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

			if (threadId == 0)
				s_sampleBitMaskNum = min(sampleBitMaskBufferNum, s_ourSampleNum - sampleCounter);

			__syncthreads();

			clearBitMask<T>(s_sampleBitMask, s_sampleBitMaskNum * lightBitMaskSizeT, threadNumPerBlock, threadId);

			//if (threadNumPerBlock >= sampleBitMaskBufferNum)
			//{ // 线程数大于需要载入的sample 数目
			//	if (threadId < sampleBitMaskBufferNum && sampleCounter + threadId < s_ourSampleNum)
			//	{
			//		int sampleID = s_ourSample[sampleCounter + threadId];

			//		int sampleX = sampleID & ((1 << viewportWidth_LOG2_UP) - 1);
			//		int sampleY = sampleID >> viewportWidth_LOG2_UP;

			//		s_sampleWritePos[threadId] = sampleY * viewportWidth + sampleX;

			//		// 
			//		float4 temp;
			//		temp = tex2D(samplePositionTex, sampleX, sampleY);
			//		s_samplePosition[threadId] = make_float3(temp);

			//		temp = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 0, cudaBoundaryModeTrap);
			//		s_sampleRectangle[threadId].x = temp.x;
			//		s_sampleRectangle[threadId].y = temp.y;

			//		temp = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 1, cudaBoundaryModeTrap);
			//		s_sampleRectangle[threadId].z = temp.x;
			//		s_sampleRectangle[threadId].w = temp.y;

			//		s_sampleZ[threadId] = temp.z;

			//		// we can load previous mask? 
			//	//	for (int i = 0; i <= lightResHeight; i++) // lightResHeight + 1  // have bank conflict
			//	//		s_sampleBitMask[threadId][i] = (T)0;
			//	}
			//}
			//else
			{ // 线程数少于载入sample 数目，分批载入。

				for (int loadId = threadId; loadId < s_sampleBitMaskNum; loadId += threadNumPerBlock)
				{
					int sampleID = s_ourSample[sampleCounter + loadId];


					int sampleX = sampleID & ((1 << viewportWidth_LOG2_UP) - 1);
					int sampleY = sampleID >> viewportWidth_LOG2_UP;

					s_sampleWritePos[loadId] = sampleY * viewportWidth + sampleX;

					// 
					float4 temp;
					temp = tex2D(samplePositionTex, sampleX, sampleY);
					s_samplePositionAndZ[loadId<<2 | 0x00] = temp.x;
					s_samplePositionAndZ[loadId << 2 | 0x01] = temp.y;
					s_samplePositionAndZ[loadId << 2 | 0x02] = temp.z;

					temp = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 0, cudaBoundaryModeTrap);
					s_sampleRectangle[loadId<<2 | 0x00] = temp.x;
					s_sampleRectangle[loadId<<2 | 0x01] = temp.y;

					temp = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX*sizeof(float4), sampleY, 1, cudaBoundaryModeTrap);
					s_sampleRectangle[loadId<<2 | 0x02] = temp.x;
					s_sampleRectangle[loadId<<2 | 0x03] = temp.y;

					//s_sampleZ[loadId] = temp.z;
					s_samplePositionAndZ[loadId << 2 | 0x03] = temp.z;

					// we can load previous mask?
				//	for (int i = 0; i <= lightResHeight; i++) // lightResHeight + 1 
				//		s_sampleBitMask[loadId][i] = (T)0;
				}
			}
			__syncthreads();

			// load triangle, in warps, no sysncthreads;
			for (int triangleCounter = 0; triangleCounter + threadIdx.y * blockDim.x < s_ourTriNum; triangleCounter += threadNumPerBlock)
			{
				// cal
				// load my triangle;
				int mytriangleID = -1;
				if (triangleCounter + threadId < s_ourTriNum)
				{
					mytriangleID = s_ourTri[triangleCounter + threadId];

				}
#if _DEBUG
				int coveredSampleNum = 0;
#endif

				
				float3 v0 = make_float3(0, 0, 0);
				float3 v1 = make_float3(0, 0, 0);
				float3 v2 = make_float3(0, 0, 0);

				float3 AABBb = make_float3(FD_F32_MAX, FD_F32_MAX, FD_F32_MAX);
				float3 AABBe = make_float3(-FD_F32_MAX, -FD_F32_MAX, -FD_F32_MAX);

				//// triangle 缓存 占用 33个寄存器
				//float3 c_v0_a = make_float3(0, 0, 0);
				//float3 c_v0_b = make_float3(0, 0, 0);
				//float3 c_v0_c = make_float3(0, 0, 0);

				//float3 c_v1_a = make_float3(0, 0, 0);
				//float3 c_v1_b = make_float3(0, 0, 0);
				//float3 c_v1_c = make_float3(0, 0, 0);
			
				//float3 c_v2_a = make_float3(0, 0, 0);
				//float3 c_v2_b = make_float3(0, 0, 0);
				//float3 c_v2_c = make_float3(0, 0, 0);

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
					//float3 v0_v1 = v0 - v1;
					//float3 v1_v2 = v1 - v2;
					//float3 v2_v0 = v2 - v0;
					

					/*float3 L_v1 = upRightCornerPosition - v1;
					c_v1_a = cross(v0_v1, y_delt);
					c_v1_b = cross(v0_v1, x_delt);
					c_v1_c = cross(v0_v1, L_v1);

					float3 L_v2 = upRightCornerPosition - v2;
					c_v2_a = cross(v1_v2, y_delt);
					c_v2_b = cross(v1_v2, x_delt);
					c_v2_c = cross(v1_v2, L_v2);*/

					//float3 L_v0 = upRightCornerPosition - v0;
					/*c_v0_a = cross(v2_v0, y_delt);
					c_v0_b = cross(v2_v0, x_delt);
					c_v0_c = cross(v2_v0, L_v0);*/

					//

					//
					/*float3 v1_v0 = v1 - v0;
					float3 v2_v0 = v2 - v0;

					n = cross(v1_v0, v2_v0);*/

					float3 v1_v2 = v1 - v2;
					float3 v2_v0 = v2 - v0;
					n = cross(v1_v2, v2_v0);
					float3 L_v0 = upRightCornerPosition - v0;

					A3_o = dot(n, y_delt);
					B3_o = dot(n, x_delt);
					C3_o = dot(n, L_v0);
					

				}
				if (__any(mytriangleID != -1))
				{
					__AddCounter(g_fdAtomics.BTpairCounter);

					for (int sampleIdx = 0; sampleIdx < s_sampleBitMaskNum; sampleIdx++)
					{
						__AddCounter(g_fdAtomics.allTriangleSampleCounter);

						bool rectangleIsInaabb = AABBb.x <= s_sampleRectangle[sampleIdx << 2 | 0x00] && AABBb.y <= s_sampleRectangle[sampleIdx << 2 | 0x01] &&
							s_sampleRectangle[sampleIdx << 2 | 0x02] <= AABBe.x && s_sampleRectangle[sampleIdx << 2 | 0x03] <= AABBe.y && // triangle AABB covered all rectangle
							//AABBb.z <= s_sampleZ[sampleIdx]))													// triangle minZ  < sample Z  有一部分三角形位于灯和sample之间。
							AABBb.z <= s_samplePositionAndZ[sampleIdx << 2 | 0x03];


#if _DEBUG

						if (rectangleIsInaabb)
						{
							__AddCounter(g_fdAtomics.aabbCoverRectangleCounter);
							coveredSampleNum++;
						}
#endif


#if _DEBUG
						if (       
							__any(rectangleIsInaabb))
#else
						if (s_sampleBitMask[sampleIdx * lightBitMaskSizeT + lightResHeight] != lightHeightFullBit &&        // this sample is not in all light 's shadow
							__any(rectangleIsInaabb))
#endif// _DEBUG
						{

							
							/// cal shadow volume params A, B, C;
							float A0, B0, C0, A1, B1, C1, A2, B2, C2, A3, B3, C3;
							float3 S = *(float3*)(&s_samplePositionAndZ[sampleIdx <<2]);

							//float temp;

							//makeShadowVolume(A0, B0, C0, A1, B1, C1, A2, B2, C2, temp, v0, v1, v2, n, S);

							float3 v0_v1 = v0 - v1;
							float3 v1_v2 = v1 - v2;
							float3 v2_v0 = v2 - v0;
							//float3 n = -1 * cross(v2_v0, v1_v2);

							float3 L_v1 = upRightCornerPosition - v1;
							float3 L_v2 = upRightCornerPosition - v2;
							float3 L_v0 = upRightCornerPosition - v0;

							float3 s_v1 = S - v1;
							A0 = dot(cross(v0_v1, y_delt), s_v1);
							B0 = dot(cross(v0_v1, x_delt), s_v1);
							C0 = dot(cross(v0_v1, L_v1), s_v1);

							float3 s_v2 = S - v2;
							A1 = dot(cross(v1_v2, y_delt), s_v2);
							B1 = dot(cross(v1_v2, x_delt), s_v2);
							C1 = dot(cross(v1_v2, L_v2), s_v2);

							float3 s_v0 = S - v0;
							A2 = dot(cross(v2_v0, y_delt), s_v0);
							B2 = dot(cross(v2_v0, x_delt), s_v0);
							C2 = dot(cross(v2_v0, L_v0), s_v0);


							/*float3 v0_s = v0 - S;
							float3 v1_s = v1 - S;
							float3 v2_s = v2 - S;

							float3 n0 = cross(v0_s, v1_s);
							float3 L_v1 = upRightCornerPosition - v1;
							A0 = dot(n0, y_delt);
							B0 = dot(n0, x_delt);
							C0 = dot(n0, L_v1);

							float3 n1 = cross(v1_s, v2_s);
							float3 L_v2 = upRightCornerPosition - v2;
							A1 = dot(n1, y_delt);
							B1 = dot(n1, x_delt);
							C1 = dot(n1, L_v2);

							float3 n2 = cross(v2_s, v0_s);
							float3 L_v0 = upRightCornerPosition - v0;
							A2 = dot(n2, y_delt);
							B2 = dot(n2, x_delt);
							C2 = dot(n2, L_v0);*/

							A3 = A3_o;
							B3 = B3_o;
							C3 = C3_o;

							//float3 n = 
							float temp = dot(n, s_v0);
							
							//float temp = -1.0f * dot(n, v2_s);
							
							T initHeightBit = lightHeightFullBit;
							// 目的： 消除S所在三角形挡住S的情况。
							// 这里会存在误差：
							// optix 中使用S到三角形交点处的距离做判断。
							//因为我们没有办法对每一条光线都算出S到交点距离，只能使用S到三角形所在平面的距离代替，（我们使用直角三角形的直角边代替了斜边）我们得到的距离比optix要小。

							// 增加判断
							if (!rectangleIsInaabb || fabsf(temp) < scenes_esp && test3DPointInTriangle(v0, v1, v2, S))
								initHeightBit = 0;
							if ( temp > 0)
							{
								A0 *= -1.0f;
								B0 *= -1.0f;
								C0 *= -1.0f;

								A1 *= -1.0f;
								B1 *= -1.0f;
								C1 *= -1.0f;

								A2 *= -1.0f;
								B2 *= -1.0f;
								C2 *= -1.0f;

								A3 *= -1.0f;
								B3 *= -1.0f;
								C3 *= -1.0f;
							}
							
#if _DEBUG

							T light_bit_y = initHeightBit;
							T lightRowFlag = 0;
#else
							
							// find light height range; 
							T light_bit_y = initHeightBit;
							volatile T &lightRowFlag = s_sampleBitMask[sampleIdx * lightBitMaskSizeT + lightResHeight];
#endif// _DEBUG


//#if _DEBUG
//							if (aabbCovered == false)
//								light_bit_y = 0;
//#endif

							light_bit_y &= calLightHeightRange<T>(A0, B0, C0);
							light_bit_y = (light_bit_y | lightRowFlag) ^ lightRowFlag;
							if (__all(light_bit_y == 0)) // no light ;
								continue;

							light_bit_y &= calLightHeightRange<T>(A1, B1, C1);
							//// 去除已经满了的行；
							light_bit_y = (light_bit_y | lightRowFlag) ^ lightRowFlag;
							if (__all(light_bit_y == 0)) // no light ;
								continue;

							light_bit_y &= calLightHeightRange<T>(A2, B2, C2);
							//// 去除已经满了的行；
							light_bit_y = (light_bit_y | lightRowFlag) ^ lightRowFlag;
							if (__all(light_bit_y == 0)) // no light ;
								continue;
							
							light_bit_y &= calLightHeightRange<T>(A3, B3, C3);
							//// 去除已经满了的行；
							light_bit_y = (light_bit_y | lightRowFlag) ^ lightRowFlag;
							if (__all(light_bit_y == 0)) // no light ;
								continue;
								

#if _DEBUG	
							if (light_bit_y != 0)
								__AddCounter(g_fdAtomics.shadowVolumeCoverCounter);
#endif

							//if ((C0 <= 0 || C1 <= 0 || C2 <= 0 || C3 <= 0))
							//	light_bit_y = 0;

							//if (__all((light_bit_y | s_sampleBitMask[sampleIdx * lightBitMaskSizeT + lightResHeight]) == s_sampleBitMask[sampleIdx * lightBitMaskSizeT + lightResHeight])) // all lights triangles affected are already in shadow;
							//	continue;

							
							
							
							// reduce all range ;
							int myHeightSize = CountNumof1<T>(light_bit_y);
							int warpHeightSize = 1<<20; // 当只有一个三角形时， 标识不使用合并求解。

							T warpHeightRange;
							if (__popc(__ballot(light_bit_y != 0)) > 1) // more than one triangle affect this sample
							{
								warpHeightRange = butterflyReduceOr<T>(light_bit_y);
								warpHeightSize = CountNumof1<T>(warpHeightRange);
							}
#if _DEBUG
							bool useFul = false;
#endif

							// if range are closer; cal together; use one atomicOr;
							if (__any(warpHeightSize * 3 <=  myHeightSize *4) )
							{
								if (__all(light_bit_y == 0 || AABBe.z < s_samplePositionAndZ[sampleIdx<<2|0x03]))//s_sampleZ[sampleIdx]))// not need depth test
								{
								 //因为warpHeightRange> 0 i 初始值必然>= 0;
									
									for (int i = FFS<T>(warpHeightRange)-1; warpHeightRange; warpHeightRange ^= ((T)1 << i), i = FFS<T>(warpHeightRange) -1)
									{
										// this row is not in range or this row is all in shadow; skip
									//	if (warpHeightRange & 0x01 == 0 || s_sampleBitMask[sampleIdx* lightBitMaskSizeT + i] == lightWidthFullBit)
										//	continue;

										T light_bit = (light_bit_y & (T)1<<i)==0?0:  calLightRowShadowWithOutDepth<T>(i, A0, A1, A2, B0, B1, B2, C0, C1, C2);

#if _DEBUG
										if (light_bit)
											useFul = true;
#endif

										// reduce all
										light_bit = butterflyReduceOr<T>(light_bit);
										if (threadIdx.x == 0)
										{
											writeLightBitToShareMem<T>(&(s_sampleBitMask[sampleIdx * lightBitMaskSizeT]), i, light_bit);
										}
									}

								}
								else
								{ // need depth test;
									for (int i = FFS<T>(warpHeightRange)-1; warpHeightRange; warpHeightRange ^= ((T)1 << i), i = FFS<T>(warpHeightRange) -1)
									{
										// this row is not in range or this row is all in shadow; skip
										//if (warpHeightRange & 0x01 == 0 || s_sampleBitMask[sampleIdx * lightBitMaskSizeT + i] == lightWidthFullBit)
											//continue;

										T light_bit = (light_bit_y & (T)1 << i) == 0 ? 0 : calLightRowShadowWithDepth<T>(i, A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3);

#if _DEBUG
										if (light_bit)
											useFul = true;
#endif

										// reduce all
										light_bit = butterflyReduceOr<T>(light_bit);
										if (threadIdx.x == 0)
										{
											writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx * lightBitMaskSizeT], i, light_bit);
										}
									}
								}
							}
							else// else cal respective; 
							{// <!>for simple just else;
								 //for depth test;
								if (__all(light_bit_y == 0 || AABBe.z < s_samplePositionAndZ[sampleIdx<<2|0x03]))//s_sampleZ[sampleIdx]))// not need depth test
								{
									if (light_bit_y != 0)
									{
										int i = max(FFS<T>(light_bit_y)-1, 0);
										light_bit_y >>= i;
										for (; light_bit_y; light_bit_y >>= 1, i++) // cal all light;
										{
											// this row is all in shadow; skip
											//if (s_sampleBitMask[sampleIdx * lightBitMaskSizeT + i] == lightWidthFullBit)
												//continue;
											T light_bit = calLightRowShadowWithOutDepth<T>(i, A0, A1, A2, B0, B1, B2, C0, C1, C2);

#if _DEBUG
											if (light_bit)
												useFul = true;
#endif

											//T light_bit = 0;
											//for (int j = 0; j < lightResWidth; j++)
											//	if (A0*i + B0*j + C0 > 0 && A1*i + B1*j + C1 > 0 && A2*i + B2*j + C2 > 0)
											//		light_bit |= 1 << j;
											
											writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx * lightBitMaskSizeT], i, light_bit);
										}
									}
								}
								else
								{// do depth test;
									if (light_bit_y != 0)
									{
										int i = max(FFS<T>(light_bit_y)-1, 0);
										light_bit_y >>= i;
										for (; light_bit_y; light_bit_y >>= 1, i++) // cal all light;
										{
											// this row is all in shadow; skip
											//if (s_sampleBitMask[sampleIdx * lightBitMaskSizeT + i] == lightWidthFullBit)
												//continue;

											T light_bit = calLightRowShadowWithDepth<T>(i, A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3);

#if _DEBUG
											if (light_bit)
												useFul = true;
												
#endif


											//T light_bit = 0;
											//for (int j = 0; j < lightResWidth; j++)
											//	if (A0*i + B0*j + C0 > 0 && A1*i + B1*j + C1 > 0 && A2*i + B2*j + C2 > 0 && A3*i + B3*j + C3 > 0)
											//		light_bit |= 1 << j;

										
											writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx * lightBitMaskSizeT], i, light_bit);

											// more row lights;
										}
									}
									//
								}

								//  
							}

#if _DEBUG
							if (useFul)
								__AddCounter(g_fdAtomics.triangleBlockSampleCounter);
#endif
							/*
							{
								T light_bit;
								for (int i = 0; i < lightResHeight; i++){
									light_bit = 0;
									for (int j = 0; j < lightResWidth; j++)
									{
										if (A0 * i + B0*j + C0 > 0 &&
											A1 * i + B1*j + C1 > 0 &&
											A2 * i + B2*j + C2 > 0 &&
											A3 * i + B3*j + C3 > 0  )
											light_bit |= ((T)1 << j);
									}
									writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx * lightBitMaskSizeT], i, light_bit&initHeightBit);
								}
							}
							*/
							
/*
							if (light_bit_y != 0)
							{
								for (int i = 0; i < lightResHeight; i++)
								{
									if (s_sampleBitMask[sampleIdx * lightBitMaskSizeT + lightResHeight] & (1 << i))
										continue;
									for (int j = 0; j < lightResWidth; j++)
									{
										float3 S = *(float3*)(&s_samplePositionAndZ[sampleIdx << 2]);
										//float3 S = make_float3(s_samplePositionAndZ[sampleIdx << 2 | 0x00], s_samplePositionAndZ[sampleIdx << 2 | 0x01], s_samplePositionAndZ[sampleIdx << 2 | 0x02]);
										if (s_sampleBitMask[sampleIdx* lightBitMaskSizeT + i] & (1 << j))
											continue;
										float3 L = c_fdStaticParams.light.lightPos[i*lightResWidth + j] - S;
										float dist = sqrt(dot(L, L));
										float3 rayDirection = L / dist;
										float  rayTmin = 1e-3;

										if (intersect_triangle_branchless(rayDirection, S, rayTmin, dist, v0, v1, v2))
										{
											writeLightBitToShareMem<T>(&s_sampleBitMask[sampleIdx* lightBitMaskSizeT], i, (T)1 << j);
										}
									}
								}

							}
							
*/

							// end of need to cal;
						}
						
						// more sample need to cal;
					}

#if _DEBUG
					if (coveredSampleNum > 0)
						__AddCounter(g_fdAtomics.validBTpairCounter);
#endif
					// end of have triangle;
				}

				// more triangle
			}

			__syncthreads();
			// store sampleMask to global;
			//if (threadNumPerBlock >= sampleBitMaskBufferNum)
			//{ // 线程数大于需要载入的sample 数目
			//	if (threadId < s_sampleBitMaskNum)
			//	{
			//		// for simple we just computer shadow once;
			//		int result = 0;
			//		for (int i = 0; i < lightResHeight; i++)
			//		{
			//			result += CountNumof1<T>(s_sampleBitMask[threadId * lightBitMaskSizeT + i]);
			//		}
			//		// write out
			//		shadowResult[s_sampleWritePos[threadId]] = (float)result / (float)lightResNum;
			//		//shadowResult[s_sampleWritePos[threadId]] = 1.0f;
			//	}
			//}
			//else
			{// 线程数少于载入sample 数目，分批写出。
				for (int storeId = threadId; storeId < s_sampleBitMaskNum; storeId += threadNumPerBlock)
				{
					int result = 0;
					for (int i = 0; i < lightResHeight; i++)
					{
						result += CountNumof1<T>(s_sampleBitMask[storeId * lightBitMaskSizeT + i]);
					}
					shadowResult[s_sampleWritePos[storeId]] = (float)result / (float)lightResNum;

					//shadowResult[s_sampleWritePos[storeId]] = 1.0f;
				}
			}
			// more samples

		}
		//end of this bin


	}// loop forever

}

void setBankSize()
{
	cudaFuncSetSharedMemConfig(shadowCal_CTA_kernel<U32>, cudaSharedMemBankSizeFourByte);
	cudaFuncSetSharedMemConfig(shadowCal_CTA_kernel<U64>, cudaSharedMemBankSizeEightByte);
}
void shadowCal()
{
	
	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("shadow cal start!\n");
	// clear global counter
	checkCudaErrors(cudaMemset(g_fdAtomics_addr, 0, sizeof(FDAtomics)));

	int sampleNumPerBlock, warpNumPerBlock, gridSizeX, sharedMemPerBlockSizeByte;
	prepareBin(sampleNumPerBlock, sharedMemPerBlockSizeByte, warpNumPerBlock, gridSizeX );
	
	
	dim3 block(32, warpNumPerBlock);
	dim3 grid(gridSizeX);
	if (h_fdStaticParams.light.lightResWidth <= 32)
		shadowCal_CTA_kernel<U32> << <grid, block, sharedMemPerBlockSizeByte >> >(m_validBinNum, sampleNumPerBlock);
	else
	{
		shadowCal_CTA_kernel<U64> << <grid, block, sharedMemPerBlockSizeByte >> >(m_validBinNum, sampleNumPerBlock);
	}

	cudaDeviceSynchronize();

	

#if _DEBUG
	FDAtomics temp;
	checkCudaErrors(cudaMemcpy(&temp, g_fdAtomics_addr, sizeof(FDAtomics), cudaMemcpyDeviceToHost));
	printf("allTriangleSampleCounter : %I64u\n ", temp.allTriangleSampleCounter);
	printf("aabb cover rectangle: %I64u, %lf%%\n", temp.aabbCoverRectangleCounter, (double)temp.aabbCoverRectangleCounter / (double)temp.allTriangleSampleCounter * 100.0f);
	printf("shadow volume cover is : %I64u, %lf%%\n", temp.shadowVolumeCoverCounter, (double)temp.shadowVolumeCoverCounter / (double)temp.allTriangleSampleCounter * 100.0f);
	printf("triangle block sample counter is : %I64u, %lf%%, %lf%%\n", temp.triangleBlockSampleCounter, (double)temp.triangleBlockSampleCounter / (double)temp.allTriangleSampleCounter * 100.0f, (double)temp.triangleBlockSampleCounter / temp.aabbCoverRectangleCounter*100);
	printf("bt pair num is %I64u\n", temp.BTpairCounter);
	printf("valid bt pair num is %I64u, %lf%%\n", temp.validBTpairCounter, (double)temp.validBTpairCounter / (double)temp.BTpairCounter* 100);

	printf("aabb cover rectangle: %I64u\n", temp.aabbCoverRectangleCounter);// , (double)temp.aabbCoverRectangleCounter / (double)temp.allTriangleSampleCounter * 100.0f);
	printf("triangle block sample counter is : %I64u, %lf%%\n", temp.triangleBlockSampleCounter, (double)temp.triangleBlockSampleCounter / (double)temp.aabbCoverRectangleCounter * 100.0f);// , (double)temp.triangleBlockSampleCounter / temp.aabbCoverRectangleCounter * 100);

#endif

	my_debug(MY_DEBUG_SECTION_SHADOW_CAL, 1)("shadow cal end!\n");

//	// cal
//	{
//		dim3 block(32, FD_COMPUTE_SHADOW_WARPS); // <!> register limit warps; threads num >= sampleBitMaskBufferNum
//		dim3 grid(m_CTA_num); // <share memory per block> <share memory per SM>and <SM num> limit;
//		//if (m_lightResWidth <= 32 && m_lightResHeight <= 32){
//		//	// 32*32
//		//	shadowCal_CTA_kernel<uint, 32, 256> << <grid, block >> >(m_validBinNum);
//		//}
//		//else{
//		//	// 64*64
//		//	if (m_lightResWidth <= 64 && m_lightResHeight <= 64)
//		//		shadowCal_CTA_kernel<U64, 64, 76> << <grid, block >> >(m_validBinNum);
//		//	else{
//		//		// lightRes too large;
//		//	}
//		//}
//
//#ifdef FD_SC_USE_U32_LINE	
//		shadowCal_CTA_kernel<uint, FD_LIGHT_RES_HEIGHT, FD_SC_SHARE_MEMORY_MAX_SAMPLE_NUM, FD_COMPUTE_SHADOW_WARPS*32> << <grid, block >> >(m_validBinNum);
//#else
//		shadowCal_CTA_kernel<U64, FD_LIGHT_RES_HEIGHT, FD_SC_SHARE_MEMORY_MAX_SAMPLE_NUM, FD_COMPUTE_SHADOW_WARPS*32> << <grid, block >> >(m_validBinNum);
//#endif
//			
//	}
	

}