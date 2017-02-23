#include "Render.h"

#include "..\util\ColumnsMajorMatrix4x4.h"
#include "..\util\Debug.h"

#include "..\FDZB_Soft_Shadow\api.h"

#include "..\CG_MATH\RotationMatrix.h"

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

GLuint create2DShadowMapTex() {

	GLuint tex;

	glGenTextures(1, &tex);

	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, 32, 32, 0, GL_LUMINANCE, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	return tex;
}

void Render::resize(int width, int height)
{
	m_wndWidth = width;
	m_wndHeight = height;

	glBindTexture(GL_TEXTURE_2D, m_worldPositionTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, m_colorTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, m_normalTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, m_DepthTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

	glBindTexture(GL_TEXTURE_2D, m_shadowMapTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, GL_LUMINANCE, GL_FLOAT, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_shadowMapPBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*sizeof(float), 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	FD::FDsetView(width, height);

}



Render::Render()
	:m_cameraPos(0, 0, 0), 
	m_lightPos(0, 0, 0),
	m_lightLa(0.0f), 
	m_lightLd(0.0f),
	m_lightLs(0.0f),
	m_wndWidth(512), 
	m_wndHeight(512),
	m_lightCorner0(0, 0, 0),
	m_lightCorner1(0, 0, 0),
	m_lightCorner2(0, 0, 0),
	m_lightCorner3(0, 0, 0)
{

	//
	m_viewMatrix.identity();
	m_projectMatrix.identity();
	//m_inertiaToLightMatrix.identity();

	m_lightWidth = m_lightHeight = 0;



	// init Shader;
	m_shaderManager = new GLSLShaderManager(m_shaderDir);

	if (!m_shaderManager->startup()) {
		fprintf(stderr, "%s %d: failed to create shaderManager", __FILE__, __LINE__);
		exit(-1);
	}

	// 根据enum 的值决定load 顺序， 不能随意改变。
	m_shaderManager->loadProgram("Phone", 0, 0, 0, "Phone");
	m_shaderManager->loadProgram("defer", 0, 0, 0, "defer");
	//m_shaderManager->loadProgram("Light", 0, 0, 0, "Light");
	//m_shaderManager->loadProgram("PointSprites", 0, 0, 0, "PointSprites"); 



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

	// init shadow map pbo
	glGenBuffers(1, &m_shadowMapPBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_shadowMapPBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 32, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// init shadow map tex
	m_shadowMapTex = create2DShadowMapTex();


	// init FDZB soft shadow
	FD::FDinit();
	FD::FDsetBackfaceCull(false);
	FD::FDsetSceneParams(1 << 21, 1 << 21);
	FD::FDsetScenesEsp(1e-3);

	FD::FDsetSampleGLTex(m_worldPositionTex, m_normalTex);
	FD::FDsetOutputGLBuffer(m_shadowMapPBO);
	
	resize(m_wndWidth, m_wndHeight);

	
	// init screen width, height
	CHECK_ERRORS();
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

void Render::DrawRectLight()
{
	glUseProgram(0);
	glColor3f(m_lightLa.r + m_lightLd.r, m_lightLa.g + m_lightLd.g, m_lightLa.b + m_lightLd.b);

	if (m_lightWidth < 0.01f || m_lightHeight < 0.01f)
	{
		glPointSize(2);

		glBegin(GL_POINTS);
		glVertex3f(m_lightPos.x, m_lightPos.y, m_lightPos.z);
		glEnd();
	}
	else
	{
		glBegin(GL_QUADS);
		glVertex3f(m_lightCorner0.x, m_lightCorner0.y, m_lightCorner0.z);
		glVertex3f(m_lightCorner1.x, m_lightCorner1.y, m_lightCorner1.z);
		glVertex3f(m_lightCorner2.x, m_lightCorner2.y, m_lightCorner2.z);
		glVertex3f(m_lightCorner3.x, m_lightCorner3.y, m_lightCorner3.z);
		glEnd();
	}
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

#if 0 // 在屏幕上直接显示
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
#else // defer shading

	m_FBO->Bind();
	static const GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, drawBuffers);
	glViewport(0, 0, m_wndWidth, m_wndHeight);

	glEnable(GL_DEPTH_TEST);

	static const float NaN = std::numeric_limits<float>::quiet_NaN();

	glClearColor(NaN, NaN, NaN, NaN);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	DrawModel();

#endif
}

void Render::pass1()
{

	// 计算光源参数

	AreaLightDes lightDes;
	lightDes.lightResWidth = lightDes.lightResHeight = 2;

	lightDes.position = glm::vec3(m_lightPos.x, m_lightPos.y, m_lightPos.z);
	lightDes.rightRadius = m_lightWidth / 2.0f;
	lightDes.topRadius = m_lightHeight / 2.0f;

	CG_MATH::RotationMatrix r;
	r.setup(m_lightDir);
	lightDes.upDir = glm::vec3(r.m21, r.m22, r.m23);
	lightDes.viewDir = glm::vec3(-r.m31, -r.m32, -r.m33);
	vector3 x = vector3(r.m11, r.m12, r.m13) * lightDes.rightRadius;
	vector3 y = vector3(r.m21, r.m22, r.m23) * lightDes.topRadius;
	m_lightCorner0 = m_lightPos + x + y;
	m_lightCorner1 = m_lightPos - x + y;
	m_lightCorner2 = m_lightPos - x - y;
	m_lightCorner3 = m_lightPos + x - y;

	FD::FDsetAreaLight(lightDes);

// 开始cuda 运算


	double cuda_time;
	glFinish();

	// cal shadow
	cuda_time = FD::FDLaunch();


	glFinish();
	static double avg_time = 0;
	static int frame_cnt = 0;

	avg_time += cuda_time;
	frame_cnt++;
	if (frame_cnt >= 10)
	{
		avg_time /= frame_cnt;
		printf("cuda time is %.2lfms\n", avg_time);
		frame_cnt = 0;
		avg_time = 0;
	}

	// 读取结果到shadow map tex
	glPixelStoref(GL_UNPACK_ALIGNMENT, 1);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_shadowMapPBO);
	glBindTexture(GL_TEXTURE_2D, m_shadowMapTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_wndWidth, m_wndHeight, GL_LUMINANCE, GL_FLOAT, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void Render::pass2()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_wndWidth, m_wndHeight);

	
	// black background;
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	// defer shading
	glUseProgram(m_shaderManager->program[DEFER].handle);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_shadowMapTex);
	glUniform1i(m_shaderManager->program[DEFER].getUniformLocation("shadowMap"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, m_colorTex);
	glUniform1i(m_shaderManager->program[DEFER].getUniformLocation("diffuseAndSpecTex"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, m_normalTex);
	glUniform1i(m_shaderManager->program[DEFER].getUniformLocation("ambientColorTex"), 2);

	glBegin(GL_QUADS);
	glVertex3f(-1.0f, 1.0f, 0.0f);;
	glVertex3f(1.0f, 1.0f, 0.0f);
	glVertex3f(1.0f, -1.0f, 0.0f);
	glVertex3f(-1.0f, -1.0f, 0.0f);
	glEnd();


	// Draw light, axis, etc.

	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_FBO->GetID());
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // Write to default framebuffer
	glBlitFramebuffer(
		0, 0, m_wndWidth, m_wndHeight, 0, 0, m_wndWidth, m_wndHeight, GL_DEPTH_BUFFER_BIT, GL_NEAREST
	);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(&(static_cast<ColumnsMajorMatrix4x4>(m_viewMatrix)).m[0][0]);

	DrawRectLight();
	DrawAxis();
}
