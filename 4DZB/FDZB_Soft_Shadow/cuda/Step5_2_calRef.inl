__device__ bool intersect_triangle_branchless(const float3&    rayDirection,
	const float3& rayOrigin,
	const float&  rayTmin,
	const float&  rayTmax,
	const float3& p0,
	const float3& p1,
	const float3& p2
	)
{
	float3 n;
	float  t;
	float  beta;
	float  gamma;

	const float3 e0 = p1 - p0;
	const float3 e1 = p0 - p2;
	n = cross(e1, e0);

	const float3 e2 = (1.0f / dot(n, rayDirection)) * (p0 - rayOrigin);
	const float3 i = cross(rayDirection, e2);

	beta = dot(i, e1);
	gamma = dot(i, e0);
	t = dot(n, e2);

	return ((t<rayTmax) & (t>rayTmin) & (beta >= 0.0f) & (gamma >= 0.0f) & (beta + gamma <= 1));
}



__global__ void refCal_kernel()
{
	// input;
	int viewportX = c_fdStaticParams.viewportWidth;
	int viewportY = c_fdStaticParams.viewportHeight;
	int viewportX_LOG2UP = c_fdStaticParams.viewportWidth_LOG2_UP;
	int binWidth = c_fdLightPlaneParams.widthBins;
	int binWidth_LOG2 = c_fdLightPlaneParams.widthBins_LOG2;
	int binHeight = c_fdLightPlaneParams.heightBins;
	int INVALID_BIN = c_fdLightPlaneParams.numBins;
	float3 lightPlaneB = c_fdLightPlaneParams.begin;
	float3 lightPlaneE = c_fdLightPlaneParams.end;
	float2 lightPlaneFactor = c_fdLightPlaneParams.factor;
	const int lightNum = c_fdStaticParams.light.lightResNum;

	cudaSurfaceObject_t sampleRectangleSurf = c_fdStaticParams.sampleRectangleSurf;

	float3* lightPos = c_REF_CAL_lightPos;

	int * const binTriStart = (int *)c_fdStaticParams.binTriStart;
	int * const binTriEnd = (int *)c_fdStaticParams.binTriEnd;
	int * const pair_tri = (int *)c_fdStaticParams.bt_pair_tri;



	// output;
	float * const shadowResult = (float*)c_fdStaticParams.shadowValue;


	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < viewportX && y < viewportY)
	{
		float2 rectB, rectE;
		float sampleZ;
		float4 temp;
		temp = surf2DLayeredread<float4>(sampleRectangleSurf, x*sizeof(float4), y, 0, cudaBoundaryModeTrap);
		rectB = make_float2(temp);
		temp = surf2DLayeredread<float4>(sampleRectangleSurf, x*sizeof(float4), y, 1, cudaBoundaryModeTrap);
		rectE = make_float2(temp);
		sampleZ = temp.z;

		if ((rectB.x > rectE.x || rectB.y > rectE.y) // 该像素点rectangle不存在，可能是像素是屏幕的背景。
			|| (rectB.x < lightPlaneB.x || rectB.y <lightPlaneB.y || rectE.x > lightPlaneE.x || rectE.y > lightPlaneE.y) // 该矩形没有全部在light plane 中，说明没有三角形能遮挡该像素
			)
		{
			// 此时阴影值为0， 不需要进一步计算；
			return;
		}
		
		float4 samplePos = tex2D(samplePositionTex, x, y);

		float2 center = (rectE + rectB) / 2;
		int2 mybin;
		mybin.x = floorf((center.x - lightPlaneB.x) * lightPlaneFactor.x);
		mybin.y = floorf((center.y - lightPlaneB.y) * lightPlaneFactor.y);
		int binID = mybin.y << binWidth_LOG2 | mybin.x;
		int triStart = binTriStart[binID];
		int triEnd = binTriEnd[binID];

		uint shadowBits[32] = { 0, 0 };
		
		for (int i = triStart; i < triEnd; i++)
		{
			int tid = pair_tri[i];
			float3 AABBb;
			float3 AABBe;
			float3 p0, p1, p2;
			
			loadTriangle(p0, p1, p2, AABBb, AABBe, tid);

			if (!(rectE.x <= AABBe.x && rectE.y <= AABBe.y && rectB.x >= AABBb.x && rectB.y >= AABBb.y))
				continue;

			if (AABBb.z - 1e-3  > sampleZ)
				continue;

			
			for (int i = 0; i < lightNum; i++)
			{
				if (shadowBits[i >> 5] & 1 << (i & 0x1f))
					continue;
				float3 L = lightPos[i] - make_float3(samplePos);
				float dist = sqrt(dot(L, L));
				float3 rayDirection = L/dist;
				float  rayTmin = 1e-3;
				
				if (intersect_triangle_branchless(rayDirection, make_float3(samplePos), rayTmin, dist, p0, p1, p2))
				{
					shadowBits[i >> 5] |= 1<<( i & 0x1f);
				}
			}
		}
		int result = 0;
		for (int i = 0; i < lightNum; i++)
			if (shadowBits[i >> 5] & 1 << (i & 0x1f))
				result++;
		shadowResult[y*viewportX + x] = (float)result / lightNum;
	}
}

void refCal()
{
	dim3 block(16, 16);
	dim3 grid(iDiviUp(m_viewWidth, block.x), iDiviUp(m_viewHeight, block.y));

	refCal_kernel << <grid, block >> >();

	cudaThreadSynchronize();

	getLastCudaError("refcal");
}

