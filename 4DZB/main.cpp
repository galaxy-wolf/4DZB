
#include <GL/glew.h>
#include <GL/glut.h>

#include "Render\Render.h"
#include "ResourceManager\MeshManager.h"
#include "CG_MATH\FPScamera.h"
#include "Controller\Controller.h"
#include "util\ColumnsMajorMatrix4x4.h"
#include "util\Color.h"

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

	glutMainLoop();
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
	meshM.addMesh(ResDir +  "tree4.obj");


	TextureManager::getInstance().setBaseDirPath("./Resources/");

	// 固定投影矩阵
	Render::getInstance().m_projectMatrix.setupPerspective(50.0f, 1.0f, 0.1f, 100.0f);
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(&(static_cast<ColumnsMajorMatrix4x4>(Render::getInstance().m_projectMatrix)).m[0][0]);

	Render::getInstance().resize(512, 512);
	Render::getInstance().m_lightPos = vector3(5.0f, 5.0f, 5.0f);
	Render::getInstance().m_lightLa = Color3f(0.2f);
	Render::getInstance().m_lightLd = Color3f(0.4f);
	Render::getInstance().m_lightLs = Color3f(0.3f);
}

void Display() {
	// Draw model.

	Render &render = Render::getInstance();

	render.m_cameraPos = Controller::getInstance().m_camera.pos;
	render.m_viewMatrix = Controller::getInstance().m_camera.getMatrix();

	render.pass0();

	// cuda ;

	//render.pass1();

	//render.pass2();

	glutSwapBuffers();
	glutPostRedisplay();
}


// 不限制帧率，每帧开始时，先更新场景，然后绘制。
// 
void Frame() {

	// update model matrix.


	
	Display();
}