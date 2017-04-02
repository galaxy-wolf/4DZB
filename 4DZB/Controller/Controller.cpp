#include "Controller.h"

#include <GL/glew.h>
#include <GL/glut.h>
#include <string.h>
#include <stdio.h>

#include "..\Util\Color.h"
#include "..\Util\SaveBMP.h"

void Controller::mouse(int button, int state, int x, int y) {

	if (state == GLUT_DOWN) {
		m_lastX = x;
		m_lastY = y;
		m_enableMouseMove = true;
	}
	else {
		m_enableMouseMove = false;
	}
}

void Controller::motion(int x, int y) {

	if (m_enableMouseMove) {
		int deltY = y - m_lastY;
		int deltX = x - m_lastX;

		m_lastX = x;
		m_lastY = y;

		m_camera.rotate2D(deltX * -.2f, deltY * -0.2f);
	}
}


void Controller::keyboard(unsigned char key, int x, int y) {

	const float step = 0.05f;

	// ��дתСд
	if ('A' <= key && key <= 'Z')
		key += 'a' - 'A';

	switch (key)
	{
	case 'w':
		m_camera.move(step, 0, 0);
		break;

	case 's':
		m_camera.move(-step, 0, 0);
		break;

	case 'a':
		m_camera.move(0, step, 0);
		break;

	case 'd':
		m_camera.move(0, -step, 0);
		break;

	case 'q':
		m_camera.move(0, 0, step);
		break;

	case 'e':
		m_camera.move(0, 0, -step);
		break;

	// controller params
	case 'p':
		SaveParam();
		break;
	case 'l':
		LoadParam();
		break;
	case 't':
		SaveScreenShot();
		break;

	case 'm':
		moveModel = !moveModel;

		if (moveModel)
			printf("move model\n");
		break;


	default:
		break;
	}
}

inline void print(FILE* f, const vector3& v3)
{
	fprintf(f, "%f %f %f\n", v3.x, v3.y, v3.z);
}

inline void print(FILE* f, const CG_MATH::EulerAngles& e)
{
	fprintf(f, "%f %f %f\n", e.heading, e.pitch, e.bank);
}

inline void print(FILE * f, const Color3f& c3)
{
	fprintf(f, "%f %f %f\n", c3.r, c3.g, c3.b);
}


inline void scan(FILE* f, vector3& v3)
{
	fscanf(f, "%f %f %f\n", &v3.x, &v3.y, &v3.z);
}

inline void scan(FILE* f, CG_MATH::EulerAngles& e)
{
	fscanf(f, "%f %f %f\n", &e.heading, &e.pitch, &e.bank);
}

inline void scan(FILE * f, Color3f& c3)
{
	fscanf(f, "%f %f %f\n", &c3.r, &c3.g, &c3.b);
}

void Controller::SaveParam()
{
	const char fileName[] = ".\\ControllerParams\\controller.txt";
	FILE * f = fopen(fileName, "w+");

	// save camera;
	print(f, m_camera.pos);
	print(f, m_camera.dir);
	// save RectLight;
	print(f, m_light.m_pos);
	print(f, m_light.m_Dir);
	print(f, m_light.m_La);
	print(f, m_light.m_Ld);
	print(f, m_light.m_Ls);
	fprintf(f, "%f %f\n", m_light.m_width, m_light.m_height);
	fprintf(f, "%d %d\n", m_light.m_sampleWidth, m_light.m_sampleHeight);

	fclose(f);

	printf("Save controller params done!\n");
}

void Controller::LoadParam()
{
	const char fileName[] = ".\\ControllerParams\\controller.txt";
	FILE * f = fopen(fileName, "r");

	if (f == NULL)
	{
		fprintf(stderr, "load controller param failed:\n\t file %s not exist!", fileName);
		return;
	}

	// load camera;
	scan(f, m_camera.pos);
	scan(f, m_camera.dir);
	// load RectLight;
	scan(f, m_light.m_pos);
	scan(f, m_light.m_Dir);
	scan(f, m_light.m_La);
	scan(f, m_light.m_Ld);
	scan(f, m_light.m_Ls);
	fscanf(f, "%f %f\n", &m_light.m_width, &m_light.m_height);
	fscanf(f, "%d %d\n", &m_light.m_sampleWidth, &m_light.m_sampleHeight);

	fclose(f);
}

void Controller::SaveScreenShot()
{
	static unsigned int cnt = 0;
	char fileName[128];
	sprintf(fileName, "E:\\Results\\%d", ++cnt);

	saveScreenToBMP(fileName, 512, 512, false);
}



