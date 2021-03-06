
#include <GL/glew.h>
#include <GL/glut.h>

#include "Render\Render.h"
#include "ResourceManager\MeshManager.h"
#include "CG_MATH\FPScamera.h"
#include "Controller\Controller.h"
#include "util\ColumnsMajorMatrix4x4.h"
#include "util\Color.h"

#include <glm\glm.hpp>
#include <glm/gtx/transform.hpp>
#include "FDZB_Soft_Shadow\api.h"

using namespace std;
using namespace CG_MATH;

void init();
void Frame();
void mouse(int button, int state, int x, int y) {
	Controller::getInstance().mouse(button, state, x, y);
}
void motion(int x, int y) {
	Controller::getInstance().motion(x, y);
}
void keyboard(unsigned char key, int x, int y) {
	Controller::getInstance().keyboard(key, x, y);
}



int main(int argc, char *argv[])
{

	//1，  init glut;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(512, 512);
	glutCreateWindow("4DZB");
	//glutDisplayFunc(&Frame);
	glutIdleFunc(&Frame);
	glutKeyboardFunc(&keyboard);
	glutMouseFunc(&mouse);
	glutMotionFunc(&motion);

	glutReportErrors();


	//2，init glew， load 模型
	init();


	FD::FDregisterGLres();

	glutMainLoop();

	FD::FDunregisterGLres();
	return 0;
}


void init() {
	
	glewInit();

	if (!glewIsSupported("GL_EXT_framebuffer_object"))
	{
		printf("Unable to load the necessary extensions\n");
		printf("This sample requires:\n"
			"OpenGL 4.0\n"
			"GL_EXT_framebuffer_object\n"
			"Exiting...\n");
		exit(-1);
	}


	MeshManager& meshM = MeshManager::getInstance();
	const string ResDir = "./Resources/";
	meshM.addMesh(ResDir + "plane.obj");
	meshM.addMesh(ResDir + "dragon_1.obj");
	//meshM.addMesh(ResDir + "box.obj");
	//meshM.addMesh(ResDir + "triangle.obj");

	meshM.createGPUbufferForCUDA();

	TextureManager::getInstance().setBaseDirPath("./Resources/");

	// 固定投影矩阵
	Render::getInstance().m_projectMatrix.setupPerspective(50.0f, 1.0f, 0.1f, 100.0f);
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(&(static_cast<ColumnsMajorMatrix4x4>(Render::getInstance().m_projectMatrix)).m[0][0]);

}

void Display() {
	// Draw model.

	Render &render = Render::getInstance();
	Controller &controller = Controller::getInstance();

	// copy controller parm -=> render
	render.m_cameraPos = controller.m_camera.pos;
	render.m_viewMatrix = controller.m_camera.getMatrix();

	render.m_lightPos = controller.m_light.m_pos;
	render.m_lightWidth = controller.m_light.m_width;
	render.m_lightHeight = controller.m_light.m_height;
	render.m_lightSampleWidth = controller.m_light.m_sampleWidth;
	render.m_lightSampleHeight = controller.m_light.m_sampleHeight;
	render.m_lightLa = controller.m_light.m_La;
	render.m_lightLd = controller.m_light.m_Ld;
	render.m_lightLs = controller.m_light.m_Ls;
	render.m_lightDir = controller.m_light.m_Dir;



	render.pass0();
	// cuda ;
	render.pass1();
	render.pass2();



	glutSwapBuffers();
	glutPostRedisplay();
}





// 不限制帧率，每帧开始时，先更新场景，然后绘制。
// 
void Frame() {

	// update light pos, radius

	// update model matrix.

	if (Controller::getInstance().moveModel)
	{
		static float movePos = 0.0f;
		static float moveDelt = 0.1f;

		movePos += moveDelt;

		if (movePos > 3.0f || movePos < -3.0f)
			moveDelt *= -1;
			
		Matrix3x4 m;
		m.setupTanslation(vector3(moveDelt, 0.0f, 0.0f));

		MeshManager& meshM = MeshManager::getInstance();
		Mesh & modle = meshM.m_meshes[1];
		modle.m_ObjectToWorldMatrix *= m;

		float _m[16];

		_m[0] = m.m11; _m[4] = m.m12; _m[8] =  m.m13; _m[12] = m.tx;
		_m[1] = m.m21; _m[5] = m.m22; _m[9] =  m.m23; _m[13] = m.ty;
		_m[2] = m.m31; _m[6] = m.m32; _m[10] = m.m33; _m[14] = m.tz;
		_m[3] = 0.0f ; _m[7] = 0.0f ; _m[11] =  0.0f; _m[15] = 1.0f;


		FD::moveModel(_m);

	}
	


	// update cuda model;
	
	Display();
}