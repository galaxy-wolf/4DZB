#pragma once

#include <GL/glew.h>
#include <GL/glut.h>

#include <string>

#include "..\ResourceManager\shaderLoader/shader_glsl.h"
#include "..\ResourceManager\TextureManger.h"
#include "..\ResourceManager\MeshManager.h"
#include "..\Render\framebufferObject.h"
#include "..\CG_MATH\Matrix4x4.h"
#include "..\util\Color.h"


// 单例模式
class Render
{
public:

	//公共操作

	static Render& getInstance() {
		
		static Render Instance;
		return Instance;
	}

	Render(Render &r) = delete;
	Render& operator= (Render &r) = delete;

	void resize(int width, int height);

	void pass0();
	void pass1();
	void pass2();

	// 公共数据

	Matrix3x4 m_viewMatrix;
	Matrix4x4 m_projectMatrix;
	vector3 m_cameraPos;
	vector3 m_lightPos;

	Color3f m_lightLa;
	Color3f m_lightLd;
	Color3f m_lightLs;

private:
	
	//私有操作
	
	Render();
	~Render();

	void DrawAxis();
	void DrawModel();
	void BindMaterial(GLuint programID, const GLMmaterial& material);
	void BindTextrue(
		GLuint programID,
		int texUnit,
		const char* colorNameInShader, const float * color,
		const char* texNameInShader, const char* texPath);
	// 私有数据

	GLSLShaderManager * m_shaderManager;
	FramebufferObject * m_FBO;

	GLuint m_worldPositionTex, m_colorTex, m_normalTex, m_DepthTex;

	enum shaderID{Phone=0};

	const char* m_shaderDir = "./Resources/shader/";

};



