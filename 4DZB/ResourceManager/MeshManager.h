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

// 管理场景中的mesh， 
// 将所有mesh 合并成一个大mesh， 方便cuda操作
//
// 使用单例模式
class MeshManager
{

public:

	// 公共操作
	static MeshManager &getInstance() {

		static MeshManager Instance;
		return Instance;
	}

	// 禁止拷贝和构造函数
	MeshManager(const MeshManager &m) = delete;
	MeshManager & operator=(const MeshManager &m) = delete;

	void addMesh(const string& path);
	void createGPUbufferForCUDA();

	// 公共数据
	// position, normal, texture coordinate
	const int m_pSize = 3;
	const int m_nSize = 3;
	const int m_tcSize = 2;

	const int m_pOffset = 0;
	const int m_nOffset = m_pOffset + m_pSize;
	const int m_tcOffset = m_nOffset + m_nSize;

	const int m_vtxSize = m_pSize + m_nSize + m_tcSize;

	vector<Mesh> m_meshes;

		// CUDA 读取整个场景模型 

	GLuint m_VBO;
	GLuint m_IBO;

	int m_verticesNum;
	int m_indicesNum;

		//每个 mesh在Scene VBO 中的起始位置 
	vector<GLuint> m_meshVertexStart;

private:
	
	
	// 私有操作

	MeshManager();
	~MeshManager();
	
	// 私有成员


};

