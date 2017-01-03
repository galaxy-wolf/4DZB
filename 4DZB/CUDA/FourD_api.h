#pragma  once
#ifndef __FOURD_API_H__
#define __FOURD_API_H__
#include <string>
namespace FD{

	// init
	void FDinit();

	// 开启反面剔除;
	void FDsetBackfaceCull(bool enable);

	// 添加模型，
	void FDaddModel(GLuint modelVBO, GLuint modelIBO, size_t vertexSizeBytes, std::string modelName);

	// 设置采样点纹理
	void FDsetSampleGLTex(GLuint samplePositionTex, GLuint sampleNormalTex);

	// 设置输出buffer
	void FDsetOutputGLBuffer(GLuint resultPBO);

	// 当纹理大小发生改变时，需要重新注册
	void FDregisterGLres();

	//程序结束解除注册
	void FDunregisterGLres();

	// 设置bin参数
	int FDsetBinNum(size_t binNum);

	// 设置视口大小
	void FDsetView(int viewWidth, int viewHeight);

	// 设置场景中三角形数量信息
	void FDsetSceneParams(size_t maxTriangleNum, size_t maxVerticesNum);

	void FDsetLightRes(int lightResWidth, int lightResHeight);

	// 设置灯参数
	void FDsetAreaLight(AreaLightDes lightDes);

	// 开始阴影计算
	void FDLaunch();



}

#endif// __FOURD_API_H__