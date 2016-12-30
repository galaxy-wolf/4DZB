#pragma once

#include "..\CG_MATH\FPScamera.h"


// ����ģʽ
class Controller
{
public:
	// ���в���
	static Controller& getInstance() {

		static Controller Instance;
		return Instance;
	}

	Controller(const Controller& c) = delete;
	Controller& operator=(const Controller &c) = delete;

	void mouse(int button, int state, int x, int y);
	void motion(int x, int y);
	void keyboard(unsigned char key, int x, int y);


	// ��������

	FPScamera m_camera;

private:
	// ˽�в���
	Controller() :m_lastX(0), m_lastY(0), m_enableMouseMove(false) {}
	~Controller()=default;


	// ˽������
	int m_lastX;
	int m_lastY;
	bool m_enableMouseMove;
	
};
