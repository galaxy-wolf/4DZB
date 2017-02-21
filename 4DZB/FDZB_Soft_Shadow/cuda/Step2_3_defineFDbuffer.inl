#include "Step2_1_samplerRectangleAABB.inl"
#include "Step2_2_modelRectangleAABB.inl"


bool lightPlaneParamCal()
{
	//////////////////////////////////////////////
	// step 1    get far plane project params;
	//////////////////////////////////////////////

	// for use far plane project, we need maxZ for sample;
	// for simple triangle setup(no split triangle), we keep maxZ of sample is bigger than maxZ of triangle;
	// cal maxZ of sample;
	float sampleMaxZ;
	if (sampleMaxZcal(sampleMaxZ))
		return 1;
	// cal maxZ of triangle ;
	float triangleMaxZ;
	if (triangleMaxZcal(triangleMaxZ))
		return 1;

	float f = fmax(sampleMaxZ, triangleMaxZ);

	h_fdRectangleSubConstant.x /= f;
	h_fdRectangleSubConstant.y /= f;

	/// set rectangle param subConstant;
	setRectangleSubConstant();
	my_debug(MY_DEBUG_SECTION_SETUP, 1)("sampler max z is : %f triangle max z is ：%f\n", sampleMaxZ, triangleMaxZ);
	my_debug(MY_DEBUG_SECTION_SETUP, 1)("rectangle sub constant is : x %f y ：%f\n", h_fdRectangleSubConstant.x, h_fdRectangleSubConstant.y);



	//////////////////////////////////////////////
	// step 2, 对 sample 和 vertices AABB 求交 得到light plane 的范围
	//////////////////////////////////////////////


	// first cal rectangle aabb
	if (sampleRectangleAABBcal()) // 没有像素
		return 1;
	verticesRectangleAABBcal();

	// cal and set light plane params
	if (modelRectangleAABB[0].x > modelRectangleAABB[1].x || modelRectangleAABB[0].y > modelRectangleAABB[1].y ||
		sampleRectangleAABB[0].x > sampleRectangleAABB[1].x || sampleRectangleAABB[0].y > sampleRectangleAABB[1].y) //没有像素或者模型
		return 1;

	h_fdLightPlaneParams.begin.x = fmax(sampleRectangleAABB[0].x, modelRectangleAABB[0].x);
	h_fdLightPlaneParams.begin.y = fmax(sampleRectangleAABB[0].y, modelRectangleAABB[0].y);
	h_fdLightPlaneParams.end.x = fmin(sampleRectangleAABB[1].x, modelRectangleAABB[1].x);
	h_fdLightPlaneParams.end.y = fmin(sampleRectangleAABB[1].y, modelRectangleAABB[1].y);

	h_fdLightPlaneParams.begin.z = modelRectangleAABB[0].z;
	h_fdLightPlaneParams.end.z = sampleRectangleAABB[1].z;
	if (h_fdLightPlaneParams.begin.x >= h_fdLightPlaneParams.end.x || h_fdLightPlaneParams.begin.y >= h_fdLightPlaneParams.end.y) // 没有相交的，没有影子
		return 1;
	// <!>
	//need ?? 
	/*float deltX = 1e-4f * (h_fdLightPlaneParams.end.x - h_fdLightPlaneParams.begin.x);
	float deltY = 1e-4f * (h_fdLightPlaneParams.end.y - h_fdLightPlaneParams.begin.x);
	h_fdLightPlaneParams.begin.x -= deltX;
	h_fdLightPlaneParams.begin.y -= deltY;
	h_fdLightPlaneParams.end.x += deltX;
	h_fdLightPlaneParams.end.y += deltY;*/


	// not sure m_binNum, 
	//h_fdLightPlaneParams.factor.x = m_binNum / (h_fdLightPlaneParams.end.x - h_fdLightPlaneParams.begin.x);
	//h_fdLightPlaneParams.factor.y = m_binNum / (h_fdLightPlaneParams.end.y - h_fdLightPlaneParams.begin.y);

	my_debug(MY_DEBUG_SECTION_SETUP, 1)("light plane range is : x %f~ %f y ：%f~%f.\n\t size is x: %f y: %f\n", h_fdLightPlaneParams.begin.x, h_fdLightPlaneParams.end.x,
		h_fdLightPlaneParams.begin.y, h_fdLightPlaneParams.end.y, h_fdLightPlaneParams.end.x - h_fdLightPlaneParams.begin.x, h_fdLightPlaneParams.end.y - h_fdLightPlaneParams.begin.y);

	// set light plane range for setupTriangleVertex;
	setLightPlaneParams();

	//////////////////////////////////////////////
	// step 3, 根据三角形投影大小，确定light plane 上格子的大小。
	//////////////////////////////////////////////

	if (setupTriangleVertex())// 没有三角形；
		return 1;

	float2 LP_size = make_float2(h_fdLightPlaneParams.end) - make_float2(h_fdLightPlaneParams.begin);
	float2 AT_size;
	AT_size.x = modelRectangleAABB[0].w;
	AT_size.y = modelRectangleAABB[1].w;
	float M = 20.0f;//sqrtf(10.0f);// sqrtf(FD_MAX_BT_PIAR_Num / m_validTriangleNum);
	int m_LP_GridSizeWidth = FD_MAX_LP_GRID_SIZE;// min((int)(LP_size.x / AT_size.x * M), FD_MAX_LP_GRID_SIZE);
	int m_LP_GridSizeHeight = FD_MAX_LP_GRID_SIZE;// min((int)(LP_size.y / AT_size.y * M), FD_MAX_LP_GRID_SIZE);

	h_fdLightPlaneParams.widthBins = m_LP_GridSizeWidth;
	h_fdLightPlaneParams.heightBins = m_LP_GridSizeHeight;
	h_fdLightPlaneParams.widthBins_LOG2 = ceilf(log2f(h_fdLightPlaneParams.widthBins));
	h_fdLightPlaneParams.numBins = h_fdLightPlaneParams.widthBins * h_fdLightPlaneParams.heightBins;
	h_fdLightPlaneParams.factor.x = h_fdLightPlaneParams.widthBins / (h_fdLightPlaneParams.end.x - h_fdLightPlaneParams.begin.x);
	h_fdLightPlaneParams.factor.y = h_fdLightPlaneParams.heightBins / (h_fdLightPlaneParams.end.y - h_fdLightPlaneParams.begin.y);

	// set all light plane params;
	setLightPlaneParams();

	my_debug(MY_DEBUG_SECTION_SETUP, 1)("width bin is %d height bin is %d\n", m_LP_GridSizeWidth, m_LP_GridSizeHeight);
	return 0;
}

// auto set LP_GRID_SIZE;

