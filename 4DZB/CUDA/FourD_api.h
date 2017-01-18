#pragma  once
#ifndef __FOURD_API_H__
#define __FOURD_API_H__
#include <string>
namespace FD{

	// init
	void FDinit();

	// ���������޳�;
	void FDsetBackfaceCull(bool enable);

	// ���ģ�ͣ�
	void FDaddModel(GLuint modelVBO, GLuint modelIBO, size_t vertexSizeBytes, std::string modelName);

	// ���ò���������
	void FDsetSampleGLTex(GLuint samplePositionTex, GLuint sampleNormalTex);

	// �������buffer
	void FDsetOutputGLBuffer(GLuint resultPBO);

	// �������С�����ı�ʱ����Ҫ����ע��
	void FDregisterGLres();

	//����������ע��
	void FDunregisterGLres();

	// ����bin����
	int FDsetBinNum(size_t binNum);

	// �����ӿڴ�С
	void FDsetView(int viewWidth, int viewHeight);

	// ���ó�����������������Ϣ
	void FDsetSceneParams(size_t maxTriangleNum, size_t maxVerticesNum);

	void FDsetLightRes(int lightResWidth, int lightResHeight);

	// ���õƲ���
	void FDsetAreaLight(AreaLightDes lightDes);

	// ��ʼ��Ӱ����
	void FDLaunch();



}

#endif// __FOURD_API_H__