#pragma once
#include "CG_MATH\vector3.h"
#include "CG_MATH\EulerAngles.h"
#include "CG_MATH\MathUtil.h"
#include "CG_MATH\Quaternion.h"

class RectAreaLight
{
public:

	// ��������

	RectAreaLight(CG_MATH::vector3 pos) :m_pos(pos), m_rightRadius(1.0f), m_topRadius(1.0f), m_ControlEulerAngle(CG_MATH::kEulerAnglesIdentity){
		m_baseQuatDir.setToRotateAboutX(-CG_MATH::kPiOver2);
	}

	void move(float front, float right, float up);
	void rotate3D(float heading, float pitch, float bank);
	
	// ��������
	float  m_rightRadius;	//half width
	float  m_topRadius;	//half height
	CG_MATH::vector3 m_pos;	//light position


private:

	// �Ƶ�����ϵ��OpenGL�����ʼ���귽����ͬ��Ȼ����X����ת-90�Ⱥ󣬵õ��Ƶĳ�ʼ��λm_baseQuatDir��
	// ֮����ƵƵķ�λ��m_ControlEulerAngle�ڳ�ʼ��λ�Ļ����Ͻ�����ת

	CG_MATH::EulerAngles m_ControlEulerAngle;
	CG_MATH::Quaternion m_baseQuatDir;
};

