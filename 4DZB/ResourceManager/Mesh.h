#pragma once
#include <string>
#include <vector>

#include <GL/glew.h>
#include <GL/glut.h>

#include "..\CG_MATH\vector3.h"
#include "..\CG_MATH\Matrix3x4.h"
#include "glm.h"

using namespace std;
using namespace CG_MATH;

// һ��mesh��������Ϣ
class Mesh
{
public:

	// ���캯��
	Mesh(const string& path);
	~Mesh();

	// ��ֹ�����Ϳ�������
	Mesh(const Mesh &m) = delete;
	Mesh& operator=(Mesh &m) = delete;

	// Mesh ���Ա��ƶ���
	// ɾ��Ĭ�Ͽ������캯���� Ҳ��ɾ���ƶ����캯�������������ƶ����캯��.
	Mesh(Mesh &&m) = default;

	//


	//���б���
	vector<float> m_vertices;
	vector<GLMmaterial> m_materials;

	vector<vector<GLuint>> m_groupIndices;
	vector<int> m_groupMaterialID;

	vector3 m_AABBmin, m_AABBmax;
	vector3 m_modelCenter;

	Matrix3x4 m_ObjectToWorldMatrix;

	// GPU Buffer

	GLuint m_VBO;
	vector<GLuint> m_groupIBO;

	// ˽�к���
private:
	void compileMesh(const GLMmodel * model);
	void createGPUbuffer();

};