#pragma once

#include "..\CG_MATH\FPScamera.h"
#include "..\RectAreaLight.h"

// 单例模式
class Controller
{
public:
	// 公有操作
	static Controller& getInstance() {

		static Controller Instance;
		return Instance;
	}

	Controller(const Controller& c) = delete;
	Controller& operator=(const Controller &c) = delete;

	void mouse(int button, int state, int x, int y);
	void motion(int x, int y);
	void keyboard(unsigned char key, int x, int y);


	// 公有数据

	FPScamera m_camera;
	RectAreaLight m_light;

private:
	// 私有操作
	Controller() :m_lastX(0), m_lastY(0), m_enableMouseMove(false) 
	{
		m_camera.pos.y = 2.0f;
		m_camera.dir.pitch = -kPiOver2 / 2.0f;
	}
	~Controller()=default;


	void SaveParam();
	void LoadParam();
	void SaveScreenShot();


	// 私有数据
	int m_lastX;
	int m_lastY;
	bool m_enableMouseMove;
	
};

