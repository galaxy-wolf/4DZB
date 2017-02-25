#pragma once

#include "..\CG_MATH\FPScamera.h"
#include "..\RectAreaLight.h"

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
	RectAreaLight m_light;

private:
	// ˽�в���
	Controller() :m_lastX(0), m_lastY(0), m_enableMouseMove(false) 
	{
		m_camera.pos.y = 2.0f;
		m_camera.dir.pitch = -kPiOver2 / 2.0f;
	}
	~Controller()=default;


	void SaveParam();
	void LoadParam();
	void SaveScreenShot();


	// ˽������
	int m_lastX;
	int m_lastY;
	bool m_enableMouseMove;
	
};

