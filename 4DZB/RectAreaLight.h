#pragma once
#include "CG_MATH\vector3.h"
#include "CG_MATH\EulerAngles.h"
#include "CG_MATH\MathUtil.h"
#include "CG_MATH\Quaternion.h"

class RectAreaLight
{
public:

	// 公共操作

	RectAreaLight(CG_MATH::vector3 pos) :m_pos(pos), m_rightRadius(1.0f), m_topRadius(1.0f), m_ControlEulerAngle(CG_MATH::kEulerAnglesIdentity){
		m_baseQuatDir.setToRotateAboutX(-CG_MATH::kPiOver2);
	}

	void move(float front, float right, float up);
	void rotate3D(float heading, float pitch, float bank);
	
	// 公共数据
	float  m_rightRadius;	//half width
	float  m_topRadius;	//half height
	CG_MATH::vector3 m_pos;	//light position


private:

	// 灯的坐标系与OpenGL相机初始坐标方向相同，然后绕X轴旋转-90度后，得到灯的初始方位m_baseQuatDir。
	// 之后控制灯的方位由m_ControlEulerAngle在初始方位的基础上进行旋转

	CG_MATH::EulerAngles m_ControlEulerAngle;
	CG_MATH::Quaternion m_baseQuatDir;
};

