#include "Render.h"
#include "..\util\ColumnsMajorMatrix4x4.h"


#include "..\util\Debug.h"

GLuint create2DColorTexture() {
	GLuint tex;
	glGenTextures(1, &tex);

	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, 32, 32, 0, GL_RGBA, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}


GLuint create2DDepthTexture() {

	GLuint tex;

	glGenTextures(1, &tex);

	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	float BorderColor[4] = { std::numeric_limits<float>::max() };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, BorderColor);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8_EXT, 32, 32, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}

void Render::resize(int width, int height)
{
	glBindTexture(GL_TEXTURE_2D, m_worldPositionTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, m_colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, m_normalTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, m_DepthTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
}



Render::Render()
	:m_cameraPos(0, 0, 0), m_lightPos(0, 0, 0)
{

	// init Shader;
	
	
	m_shaderManager = new GLSLShaderManager(m_shaderDir);

	if (!m_shaderManager->startup()) {
		fprintf(stderr, "%s %d: failed to create shaderManager", __FILE__, __LINE__);
		exit(-1);
	}

	// 根据enum 的值决定load 顺序， 不能随意改变。
	m_shaderManager->loadProgram("Phone", 0, 0, 0, "Phone");
	/*m_shaderManager->loadProgram("defer", 0, 0, 0, "defer");
	m_shaderManager->loadProgram("Light", 0, 0, 0, "Light");
	m_shaderManager->loadProgram("PointSprites", 0, 0, 0, "PointSprites"); */



	// init FBO

	m_FBO = new FramebufferObject();

	m_worldPositionTex = create2DColorTexture();
	m_colorTex = create2DColorTexture();
	m_normalTex = create2DColorTexture();

	m_DepthTex = create2DDepthTexture();

	m_FBO->Bind();
	m_FBO->AttachTexture(GL_TEXTURE_2D, m_worldPositionTex, GL_COLOR_ATTACHMENT0);
	m_FBO->AttachTexture(GL_TEXTURE_2D, m_colorTex, GL_COLOR_ATTACHMENT1);
	m_FBO->AttachTexture(GL_TEXTURE_2D, m_normalTex, GL_COLOR_ATTACHMENT2);
	m_FBO->AttachTexture(GL_TEXTURE_2D, m_DepthTex, GL_DEPTH_ATTACHMENT);

	if (!m_FBO->IsValid()) {

		fprintf(stderr, "%s %d: Render init error: create FBO failed\n", __FILE__, __LINE__);
		exit(-1);
	}


}




Render::~Render()
{
	m_FBO->Disable();
	delete m_FBO;

	m_shaderManager->shutdown();
	delete m_shaderManager;
}


void Render::DrawAxis()
{
	// 使用固定管线
	glUseProgram(0);
	
	glMatrixMode(GL_MODELVIEW);

	glPushMatrix();

	glScalef(10.0f, 10.0f, 10.0f);
	//glTranslatef(0.0f, 0.2f, -.5f);


	//x 轴
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(3.0f, 0.0f, 0.0f);
	glEnd();

	//y 轴
	glColor3f(0.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 3.0f, 0.0f);
	glEnd();

	//z 轴
	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 3.0f);
	glEnd();

	glFlush();
	glPopMatrix();

}

void Render::DrawModel()
{
	glUseProgram(m_shaderManager->program[Phone].handle);

	MeshManager & meshM = MeshManager::getInstance();

	// 设置矩阵
	Matrix4x4 vp = m_projectMatrix*m_viewMatrix;

	CHECK_ERRORS();
	glUniformMatrix4fv(
		m_shaderManager->program[Phone].getUniformLocation("ViewAndPerspectiveMatrix"),
		1, 
		GL_TRUE, 
		&vp.m11
	);

	// 设置光照计算相关参数

	glUniform3f(
		m_shaderManager->program[Phone].getUniformLocation("cameraPos"),
		m_cameraPos.x,
		m_cameraPos.y,
		m_cameraPos.z);

	glUniform3f(m_shaderManager->program[Phone].getUniformLocation("Light.Position"), m_lightPos.x, m_lightPos.y, m_lightPos.z);
	glUniform3f(m_shaderManager->program[Phone].getUniformLocation("Light.La"), m_lightLa.r, m_lightLa.g, m_lightLa.b);
	glUniform3f(m_shaderManager->program[Phone].getUniformLocation("Light.Ld"), m_lightLd.r, m_lightLd.g, m_lightLd.b);
	glUniform3f(m_shaderManager->program[Phone].getUniformLocation("Light.Ls"), m_lightLs.r, m_lightLs.g, m_lightLs.b);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);


	// for each mesh;
	for (int i = 0; i < meshM.m_meshes.size(); ++i) {
		
		Mesh& mesh = meshM.m_meshes[i];
		
		// 设置model 矩阵

		glUniformMatrix4fv(m_shaderManager->program[Phone].getUniformLocation("modelMatrix"),
			1,
			GL_FALSE, 
			&(static_cast<ColumnsMajorMatrix4x4>(mesh.m_ObjectToWorldMatrix)).m[0][0]
		);

		//绑定VBO
		glBindBuffer(GL_ARRAY_BUFFER, mesh.m_VBO);
		glVertexAttribPointer(0,
			meshM.m_pSize,
			GL_FLOAT,
			GL_FALSE,
			meshM.m_vtxSize * sizeof(float),
			(void*)(meshM.m_pOffset * sizeof(float)));

		glVertexAttribPointer(1,
			meshM.m_tcSize,
			GL_FLOAT,
			GL_FALSE,
			meshM.m_vtxSize * sizeof(float),
			(void*)(meshM.m_tcOffset * sizeof(float)));


		glVertexAttribPointer(2,
			meshM.m_nSize,
			GL_FLOAT,
			GL_FALSE,
			meshM.m_vtxSize * sizeof(float),
			(void*)(meshM.m_nOffset * sizeof(float)));

		// for each group;
		for (int j = 0; j < mesh.m_groupIndices.size(); ++j) {
			
			// 绑定纹理
			BindMaterial(Phone, mesh.m_materials[mesh.m_groupMaterialID[j]]);
			

			// 绑定IBO 
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.m_groupIBO[j]);
			glDrawElements(GL_TRIANGLES, mesh.m_groupIndices[j].size(), GL_UNSIGNED_INT, 0);
			
		}
	}

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

}


inline void Render::BindTextrue(
	GLuint programID,
	int texUnit,
	const char* colorNameInShader, const float * color,
	const char* texNameInShader, const char* texPath) {
	
	TextureManager& texM = TextureManager::getInstance();
	const float invalidColor[3] = { NAN };

	int texID;
	if ((texID = texM.getTexID(texPath)) > 0) {
		glActiveTexture(GL_TEXTURE0 + texUnit);
		glBindTexture(GL_TEXTURE_2D, texID);
		glUniform1i(
			m_shaderManager->program[programID].getUniformLocation(texNameInShader),
			texUnit);

		// 表示使用纹理
		glUniform3fv(
			m_shaderManager->program[programID].getUniformLocation(colorNameInShader),
			1,
			invalidColor);
	}
	else {
		glUniform3fv(
			m_shaderManager->program[programID].getUniformLocation(colorNameInShader),
			1,
			color);
	}

}


void Render::BindMaterial(GLuint programID, const GLMmaterial& material)
{
	BindTextrue(programID, 0, "Material.Ka", material.ambient, "Material.ambientTex", material.ambient_map);
	BindTextrue(programID, 1, "Material.Kd", material.diffuse, "Material.diffuseTex", material.diffuse_map);
	BindTextrue(programID, 2, "Material.Ks", material.specular, "Material.specTex", material.specular_map);
	
	glUniform1f(m_shaderManager->program[programID].getUniformLocation("Material.Shininess"), material.shininess);

}


void Render::pass0()
{

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glEnable(GL_DEPTH_TEST);
	

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(&(static_cast<ColumnsMajorMatrix4x4>(m_viewMatrix)).m[0][0]);
	
	DrawAxis();

	CHECK_ERRORS();

	DrawModel();


	CHECK_ERRORS();
}