#include "MeshManager.h"

MeshManager::MeshManager()
	:m_VBO(0), m_IBO(0), m_verticesNum(0), m_indicesNum(0)
{
}


MeshManager::~MeshManager()
{
	glDeleteBuffers(1, &m_VBO);
	glDeleteBuffers(1, &m_IBO);
}

void MeshManager::addMesh(const string &path)
{


	// 模型顶点在scene VBO 中的开始位置
	m_meshes.emplace_back(path);

	// 计算Vertex 起始位置

	if (m_meshes.size() == 1) {
		m_meshVertexStart.push_back(0);
	}
	else {

		int last = *(m_meshVertexStart.end() - 1);
		m_meshVertexStart.push_back(last + (m_meshes.end() - 2)->m_vertices.size() / 8);
	}
}

void MeshManager::createGPUbufferForCUDA()
{
	vector<float> vertices;
	vector<float> indices;

	// 将所有模型的vertex 和 indices 合并到一起

	for (int i = 0; i < m_meshes.size(); ++i) {

		const Mesh &m = m_meshes[i];

		vertices.insert(vertices.end(), m.m_vertices.cbegin(), m.m_vertices.cend());
		
		
		// for each group;
		for (int j = 0; j < m.m_groupIndices.size(); ++j) {
			
			// for each indices
			for (int k = 0; k < m.m_groupIndices[j].size(); ++k) {
				indices.push_back(m.m_groupIndices[j][k] + m_meshVertexStart[i]);
			}
		}
	}

	// 记录buffer size
	m_verticesNum = vertices.size() / 8;
	m_indicesNum = indices.size();



	// 创建GUP buffer

	// VBO
	glGenBuffers(1, &m_VBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	glBufferData(GL_ARRAY_BUFFER,
		vertices.size() * sizeof(float),
		&vertices[0],
		GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// IBO
	glGenBuffers(1, &m_IBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		indices.size() * sizeof(GLuint),
		&indices[0],
		GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}



