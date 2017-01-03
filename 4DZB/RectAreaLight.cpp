#include "RectAreaLight.h"
#include "CG_MATH\RotationMatrix.h"

using namespace CG_MATH;

void RectAreaLight::move(float front, float right, float up)
{
	Quaternion q;
	q.setToRotateObjectToInertial(m_ControlEulerAngle);
	q = m_baseQuatDir * q;
	RotationMatrix r;
	r.fromObjectToInertialQuaternion(q);

	m_pos.x += -front*r.m31 + right*r.m11 + up*r.m21;
	m_pos.y += -front*r.m32 + right*r.m12 + up*r.m22;
	m_pos.z += -front*r.m33 + right*r.m13 + up*r.m23;

}

void RectAreaLight::rotate3D(float heading, float pitch, float bank)
{

	//½Ç¶È×ª»¡¶È
	heading = heading / 180.0f*kPi;
	pitch = pitch / 180.0f*kPi;
	bank = bank / 180.0f*kPi;

	m_ControlEulerAngle.heading += heading;
	m_ControlEulerAngle.heading = wrapPi(m_ControlEulerAngle.heading);

	m_ControlEulerAngle.pitch += pitch;
	m_ControlEulerAngle.pitch = wrapPi(m_ControlEulerAngle.pitch);

	m_ControlEulerAngle.bank += bank;
	m_ControlEulerAngle.bank = wrapPi(m_ControlEulerAngle.bank);
}


