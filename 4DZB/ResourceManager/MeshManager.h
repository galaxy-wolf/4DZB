#pragma once
#include <string>
#include <vector>
#include <GL/glew.h>
#include <GL/glut.h>
#include "..\CG_MATH\vector3.h"
#include "..\CG_MATH\Matrix3x4.h"
#include "glm.h"

#include "Mesh.h"

using namespace std;
using namespace CG_MATH;


class Mesh;

// �������е�mesh�� 
// ������mesh �ϲ���һ����mesh�� ����cuda����
//
// ʹ�õ���ģʽ
class MeshManager
{

public:

	// ��������
	static MeshManager &getInstance() {

		static MeshManager Instance;
		return Instance;
	}

	// ��ֹ�����͹��캯��
	MeshManager(const MeshManager &m) = delete;
	MeshManager & operator=(const MeshManager &m) = delete;

	void addMesh(const string& path);
	void createGPUbufferForCUDA();

	// ��������
	// position, normal, texture coordinate
	const int m_pSize = 3;
	const int m_nSize = 3;
	const int m_tcSize = 2;

	const int m_pOffset = 0;
	const int m_nOffset = m_pOffset + m_pSize;
	const int m_tcOffset = m_nOffset + m_nSize;

	const int m_vtxSize = m_pSize + m_nSize + m_tcSize;

	vector<Mesh> m_meshes;

		// CUDA ��ȡ��������ģ�� 

	GLuint m_VBO;
	GLuint m_IBO;

	int m_verticesNum;
	int m_indicesNum;

		//ÿ�� mesh��Scene VBO �е���ʼλ�� 
	vector<GLuint> m_meshVertexStart;

private:
	
	
	// ˽�в���

	MeshManager();
	~MeshManager();
	
	// ˽�г�Ա


};

