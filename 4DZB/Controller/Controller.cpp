#include "Controller.h"

#include <GL/glew.h>
#include <GL/glut.h>


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


	default:
		break;
	}
}

