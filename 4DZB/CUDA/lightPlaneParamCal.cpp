#ifndef __LIGHT_PLANE_PARAM_CAL_CUH__
#define __LIGHT_PLANE_PARAM_CAL_CUH__
#include "lightPlaneParamCal.h"
bool lightPlaneParamCal()
{
	// first cal rectangle aabb
	sampleRectangleAABBcal();
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
	if (h_fdLightPlaneParams.begin.x > h_fdLightPlaneParams.end.x || h_fdLightPlaneParams.begin.y > h_fdLightPlaneParams.end.y) // 没有相交的，没有影子
		return 1;
	// <!>
	//need ??
	/*float deltX = 1e-4f * (h_fdLightPlaneParams.end.x - h_fdLightPlaneParams.begin.x);
	float deltY = 1e-4f * (h_fdLightPlaneParams.end.y - h_fdLightPlaneParams.begin.x);
	h_fdLightPlaneParams.begin.x -= deltX;
	h_fdLightPlaneParams.begin.y -= deltY;
	h_fdLightPlaneParams.end.x += deltX;
	h_fdLightPlaneParams.end.y += deltY;*/

	h_fdLightPlaneParams.factor.x = m_binNum / (h_fdLightPlaneParams.end.x - h_fdLightPlaneParams.begin.x);
	h_fdLightPlaneParams.factor.y = m_binNum / (h_fdLightPlaneParams.end.y - h_fdLightPlaneParams.begin.y);

	setLightPlaneParams();
	

	my_debug(MY_DEBUG_SECTION_SETUP, 1)("light plane range is : x %f~ %f y ：%f~%f\n", h_fdLightPlaneParams.begin.x, h_fdLightPlaneParams.end.x,
		h_fdLightPlaneParams.begin.y, h_fdLightPlaneParams.end.y);

	return 0;
}

#endif// __LIGHT_PLANE_PARAM_CAL_CUH__