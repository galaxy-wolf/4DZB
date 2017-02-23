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

#include "..\CG_MATH\EulerAngles.h"

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


	void pass0();
	void pass1();
	void pass2();

	// 公共数据


	GLuint m_wndWidth, m_wndHeight;

	Matrix3x4 m_viewMatrix;
	Matrix4x4 m_projectMatrix;
	vector3 m_cameraPos;
	

	vector3 m_lightPos;
	float m_lightWidth;
	float m_lightHeight;
	Color3f m_lightLa;
	Color3f m_lightLd;
	Color3f m_lightLs;
	EulerAngles m_lightDir;
	vector3 m_lightCorner0, m_lightCorner1, m_lightCorner2, m_lightCorner3;

	//Matrix4x4 m_inertiaToLightMatrix;

private:
	
	//私有操作
	
	Render();
	~Render();

	void DrawAxis();
	void DrawRectLight();
	void DrawModel();
	void BindMaterial(GLuint programID, const GLMmaterial& material);
	void BindTextrue(
		GLuint programID,
		int texUnit,
		const char* colorNameInShader, const float * color,
		const char* texNameInShader, const char* texPath);


	void resize(int width, int height);

	// 私有数据


	GLSLShaderManager * m_shaderManager;
	FramebufferObject * m_FBO;

	GLuint m_worldPositionTex, m_colorTex, m_normalTex, m_DepthTex;
	GLuint m_shadowMapPBO, m_shadowMapTex;

	enum shaderID{Phone=0, DEFER};

	const char* m_shaderDir = "./Resources/shader/";

};



