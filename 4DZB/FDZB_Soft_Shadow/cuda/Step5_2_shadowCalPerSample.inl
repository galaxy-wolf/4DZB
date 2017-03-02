
__device__ __forceinline void getPlaneDiscriminant(
	float &A, float &B, float &C,
	const float3 &v0, const float3 &v1, const float3 &Q,
	const float3 &t0, const float3 &t1, const float3 &L0_Q)
{
	float3 n = cross(v1 - Q, v0 - v1);
	
	A = dot(n, t0);
	B = dot(n, t1);
	C = dot(n, L0_Q);
}

template<typename T>
__device__ __forceinline void getRowMaskForPlaneDiscriminant(
	T& rowMask,
	const float &D, const float &B, const unsigned int & LightsampleNumPerRowSubOne, const T& FULL)
{
	bool f1 = (D >= 0.0f);
	bool f2 = (B * LightsampleNumPerRowSubOne + D >= 0.0f);

	if (f1 && f2)
		rowMask = FULL;
	else if (!f1 && !f2)
		rowMask = 0;
	else {
		rowMask = 1;
		rowMask <<= (int)ceilf(-1 * D / B);
		--rowMask;
		if (!f1)
			rowMask = ~rowMask;
	}
}


// ����Դ�ֱ��� 32x32����Դ������������
template<typename T>
__global__ void shadowCal_per_sample_kernel(const unsigned int threadNum)
{
	// input;
	int * const binTriStart = (int *)c_fdStaticParams.binTriStart;
	int * const binTriEnd = (int *)c_fdStaticParams.binTriEnd;
	int * const tb_pair_tri = (int *)c_fdStaticParams.bt_pair_tri;

	int * const pb_pair_sample = (int *)c_fdStaticParams.pb_pair_pixel;
	int * const pb_pair_bin = (int *)c_fdStaticParams.pb_pair_bin;

	const int	viewportWidth_LOG2_UP = c_fdStaticParams.viewportWidth_LOG2_UP;
	const int	viewportWidth = c_fdStaticParams.viewportWidth;

	const float scenes_esp = c_fdStaticParams.scenes_esp;

	const int lightResWidth = c_fdStaticParams.light.lightResWidth;
	const int lightResHeight = c_fdStaticParams.light.lightResHeight;

	const unsigned int lightResWidthSubOne = lightResWidth - 1;
	const unsigned int lightResHeightSubOne = lightResHeight - 1;

	const T   lightWidthFullBit = getLightWidthBit<T>();//((T)1 << lightResWidth ) - 1;//
	const T	  lightHeightFullBit = lightWidthFullBit;
	const int lightResNum = c_fdStaticParams.light.lightResNum;

	const cudaSurfaceObject_t sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;

	//light
	
	
	
	// output;
	float * const shadowResult = (float*)c_fdStaticParams.shadowValue;

	// register buffer

	extern __shared__   int _s[];
	U32 * mask = (U32*)(&_s[(threadIdx.y * blockDim.x) * lightResHeight + threadIdx.x]);
	//U32 mask[32] = { 0 };
	U32 fullRowInResultMask = 0;

	for (int i = 0; i < lightResHeight; ++i)
		mask[i * 32] = 0;

	const int threadID = blockIdx.x * blockDim.x  *blockDim.y  + threadIdx.y * blockDim.x + threadIdx.x;

	if (threadID >= threadNum)
		return;

	const int sampleID = pb_pair_sample[threadID];
	const int binID = pb_pair_bin[threadID];

	// load sample; ��ǰsample ID Ϊ sampleID
	int sampleX = sampleID & ((1 << viewportWidth_LOG2_UP) - 1);
	int sampleY = sampleID >> viewportWidth_LOG2_UP;
	float3 Q = make_float3(tex2D(samplePositionTex, sampleX, sampleY));

	// @�޸Ĵ洢��ʽ��AABB ���ڵ�һ�㣬Z���ڵڶ��㡣
	float4 sampleProj;
	float sampleProjZ;
	float4 temp = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX * sizeof(float4), sampleY, 0, cudaBoundaryModeTrap);
	sampleProj.x = temp.x;
	sampleProj.y = temp.y;

	temp = surf2DLayeredread<float4>(sampleRectangleSurf, sampleX * sizeof(float4), sampleY, 1, cudaBoundaryModeTrap);
	sampleProj.z = temp.x;
	sampleProj.w = temp.y;
	sampleProjZ = temp.z;

	const int triStart = binTriStart[binID];
	const int triEnd = binTriEnd[binID];

	for (int triIdx = triStart; triIdx < triEnd && fullRowInResultMask != lightHeightFullBit; ++triIdx)
	{
		int triID = tb_pair_tri[triIdx];

		// load ������; ��ǰ������ ID Ϊ Tid

		float3 v0, v1, v2, triProjAABBb, triProjAABBe;
		loadTriangle(v0, v1, v2, triProjAABBb, triProjAABBe, triID);

		// ����޳�, �����޳�
		if (
			!(triProjAABBb.x <= sampleProj.x && triProjAABBb.y <= sampleProj.y &&
				sampleProj.z <= triProjAABBe.x && sampleProj.w <= triProjAABBe.y && // ������ͶӰaabb ��ȫ���� ���������ͶӰ
			triProjAABBb.z <= sampleProjZ)
			)
		{
			continue;
		}

		// 
		float3 n = cross((v1 - v0), (v2 - v1));

		float dir = dot(Q - v0, n);

		if (fabsf(dir) < scenes_esp && test3DPointInTriangle(v0, v1, v2, Q))
			continue;

		float3 L0 = c_fdStaticParams.light.upRightCornerPosition;
		float3 t0 = c_fdStaticParams.light.x_delt;
		float3 t1 = c_fdStaticParams.light.y_delt;

		float3 L0_Q = L0 - Q;

		if (dir < 0)
		{
			t0 *= -1;
			t1 *= -1;
			L0_Q *= -1;
		}

		// �б�ʽ����
		float A0, A1, A2, B0, B1, B2, C0, C1, C2;

		// QV0V1
		getPlaneDiscriminant(
			A0, B0, C0,
			v0, v1, Q,
			t0, t1, L0_Q
		);

		// QV1V2
		getPlaneDiscriminant(
			A1, B1, C1,
			v1, v2, Q,
			t0, t1, L0_Q
		);

		// QV2V0
		getPlaneDiscriminant(
			A2, B2, C2,
			v2, v0, Q,
			t0, t1, L0_Q
		);



		// �ظ��ڵ��޳�
		T N0, N1, N2;
		getRowMaskForPlaneDiscriminant(
			N0,
			(B0 > 0 ? B0*lightResWidthSubOne : 0) + C0, A0, lightResHeightSubOne, lightHeightFullBit);

		getRowMaskForPlaneDiscriminant(
			N1,
			(B1 > 0 ? B1*lightResWidthSubOne : 0) + C1, A1, lightResHeightSubOne, lightHeightFullBit);

		getRowMaskForPlaneDiscriminant(
			N2,
			(B2 > 0 ? B2*lightResWidthSubOne : 0) + C2, A2, lightResHeightSubOne, lightHeightFullBit);

		// ���ܲ����ڵ������� 1�� һ���������ڵ�������0��
		T validrowMask = (N0 & N1 & N2);

		// ȥ���Ѿ��ڵ�����
		// ������Ҫ��⣬��֮ǰû����ȫ�����С�
		//	fullRowInResultMask   validrowMask    ���
		//       0                    1           1
		//       0                    0           0
		//       1                    1           0
		//       1                    0           0
		validrowMask = (~fullRowInResultMask) & validrowMask;


		// ������⣬ע�ⲻ���������ġ�
		for (int rowID = FFS<T>(validrowMask) - 1; validrowMask; validrowMask ^= ((T)1 << rowID), rowID = FFS<T>(validrowMask) - 1)
		{
			T M0, M1, M2;
			getRowMaskForPlaneDiscriminant(
				M0,
				A0*rowID + C0, B0, lightResWidthSubOne, lightWidthFullBit);

			getRowMaskForPlaneDiscriminant(
				M1,
				A1*rowID + C1, B1, lightResWidthSubOne, lightWidthFullBit);
			getRowMaskForPlaneDiscriminant(
				M2,
				A2*rowID + C2, B2, lightResWidthSubOne, lightWidthFullBit);

			mask[rowID*32] |= M0 & M1 & M2;

			if (mask[rowID*32] == lightWidthFullBit)
				fullRowInResultMask |= ((T)1 << rowID);
		}
	}


	int result = 0;
	for (int i = 0; i < lightResHeight; i++)
	{
		result += CountNumof1<T>(mask[i*32]);
	}
	shadowResult[sampleID] = (float)result / (float)lightResNum;


}

void shadowCal_perSample()
{
	cudaFuncSetCacheConfig((void*)shadowCal_per_sample_kernel<U32>, cudaFuncCachePreferShared);
	cudaFuncSetSharedMemConfig(shadowCal_CTA_kernel<U32>, cudaSharedMemBankSizeFourByte);
	cudaFuncSetSharedMemConfig(shadowCal_CTA_kernel<U64>, cudaSharedMemBankSizeEightByte);

	// clear global counter
	checkCudaErrors(cudaMemset(g_fdAtomics_addr, 0, sizeof(FDAtomics)));

	const unsigned int validSampleNum = m_validSampleNum;
	
	dim3 block(32, 16);
	dim3 grid(iDiviUp(validSampleNum, block.x*block.y), 1);
	U32 sharedMemSize = h_fdStaticParams.light.lightResHeight * sizeof(U32) * block.x*block.y;

	if (h_fdStaticParams.light.lightResWidth <= 32 && h_fdStaticParams.light.lightResHeight <=32 && h_fdStaticParams.light.lightResWidth == h_fdStaticParams.light.lightResHeight)
		shadowCal_per_sample_kernel<U32> << <grid, block, sharedMemSize>> >(validSampleNum);
	else
		fprintf(stderr, "����Դ�����ֱ���Ϊ32x32,�ҹ�Դ�ֱ�������==������ ���õĹ�Դ������Ϊ%dx%d\n", h_fdStaticParams.light.lightResWidth, h_fdStaticParams.light.lightResHeight);

	

	cudaDeviceSynchronize();
}