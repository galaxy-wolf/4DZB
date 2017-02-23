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

// 一个mesh的所有信息
class Mesh
{
public:

	// 构造函数
	Mesh(const string& path);
	~Mesh();

	// 禁止拷贝和拷贝构造
	Mesh(const Mesh &m) = delete;
	Mesh& operator=(Mesh &m) = delete;

	// Mesh 可以被移动，
	// 删除默认拷贝构造函数后， 也会删除移动构造函数，这里启用移动构造函数.
	Mesh(Mesh &&m) noexcept;

	//


	//公有变量


	// m_vertices 存储方式为：
	//|====================================
	//|		     | 3 float position       |
	//|		     --------------------------
	//| vertex 0 | 3 float normal         |
	//|          --------------------------
	//|          | 2 float textcoordinate |
	//|====================================
	//|		     | 3 float position       |
	//|		     --------------------------
	//| vertex 1 | 3 float normal         |
	//|          --------------------------
	//|          | 2 float textcoordinate |
	//|====================================
	// ....
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

	// 私有函数
private:
	void compileMesh(const GLMmodel * model);
	void createGPUbuffer();

};