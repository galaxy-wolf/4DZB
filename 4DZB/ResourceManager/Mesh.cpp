#include "Mesh.h"

#include <map>
#include <tuple>

using namespace std;

typedef tuple<GLuint, GLuint, GLuint> triple;

Mesh::Mesh(const string& path)
	:m_AABBmin(kZeroVector), m_AABBmax(kZeroVector), m_modelCenter(kZeroVector), m_VBO(0)
{

	m_ObjectToWorldMatrix.identity();

	GLMmodel * model;

	// 使用glm load mesh

	model = glmReadOBJ(path.c_str());

	if (!model) {

		fprintf(stderr, "Can not load file: %s\n", path.c_str());
		exit(-1);
	}


	// 整理mesh 数据， 使得其按照索引方式存储, 并将结果写在m_vertices, m_indices, m_materials
	compileMesh(model);



	// 删除glm mesh

	glmDelete(model);


	// 创建GPU buffer
	createGPUbuffer();

}

Mesh::~Mesh()
{
	//http://docs.gl/gl3/glDeleteBuffers
	// glDeleteBuffers silently ignores 0's and names that do not correspond to existing buffer objects.
	glDeleteBuffers(1, &m_VBO);

	if (m_groupIBO.size() > 0)
		glDeleteBuffers(m_groupIBO.size(), &m_groupIBO[0]);
}


void Mesh::compileMesh(const GLMmodel * model)
{

	float *positions = model->vertices;
	float *normals = model->normals;
	float *texCoords = model->texcoords;

	GLMgroup *group = model->groups;
	GLMtriangle *triangle = model->triangles;

	map<triple, int> tripleToIdx;


	// vertex

	for (; group; group = group->next) {

		if (group->numtriangles <= 0)
			continue;

		m_groupMaterialID.push_back(group->material);
		m_groupIndices.push_back(vector<GLuint>());

		for (int i = 0; i < group->numtriangles; ++i) {
			const unsigned int &tid = group->triangles[i];
			const GLMtriangle &curTri = triangle[tid];

			for (int j = 0; j < 3; ++j) {

				triple key{ curTri.vindices[j], curTri.nindices[j], curTri.tindices[j] };

				int value;

				if (!tripleToIdx.count(key)) {


					//没有找到相同的顶点， 插入一个新的顶点。
					int index;

					// 编译后的点 默认一个点包括3个float的position， 3个float的normal， 2个float的纹理坐标
					// 并且依次排列
					vector<float> vertex(8, 0.0f);
					if (positions != NULL) {

						index = curTri.vindices[j];

						vertex[0] = positions[index * 3 + 0];
						vertex[1] = positions[index * 3 + 1];
						vertex[2] = positions[index * 3 + 2];
					}

					if (normals != NULL) {

						index = curTri.nindices[j];

						vertex[3] = normals[index * 3 + 0];
						vertex[4] = normals[index * 3 + 1];
						vertex[5] = normals[index * 3 + 2];
					}

					if (texCoords != NULL) {

						index = curTri.tindices[j];

						vertex[6] = texCoords[index * 2 + 0];
						vertex[7] = texCoords[index * 2 + 1];
					}


					m_vertices.insert(m_vertices.end(), vertex.begin(), vertex.end());

					value = (m_vertices.size() / 8) - 1;
					tripleToIdx[key] = value;
				}
				else {
					// 找到相同的顶点，

					value = tripleToIdx[key];
				}

				m_groupIndices[m_groupIndices.size() - 1].push_back(value);
			}
		}

	}

	// materials;
	for (int i = 0; i < model->nummaterials; ++i)
		m_materials.push_back(model->materials[i]);

	// AABB
	glmBoundingBox(model, &m_AABBmin.x, &m_AABBmax.x);
	m_modelCenter = (m_AABBmin + m_AABBmax)*0.5f;


}

void Mesh::createGPUbuffer()
{

	// VBO
	glGenBuffers(1, &m_VBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	glBufferData(GL_ARRAY_BUFFER,
		m_vertices.size() * sizeof(float),
		&m_vertices[0],
		GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// IBO
	m_groupIBO.resize(m_groupIndices.size());

	glGenBuffers(m_groupIBO.size(), &m_groupIBO[0]);

	for (int i = 0; i < m_groupIBO.size(); ++i) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_groupIBO[i]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,
			m_groupIndices[i].size() * sizeof(GLuint),
			&m_groupIndices[i][0],
			GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
}
