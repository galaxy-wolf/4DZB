#pragma once
#include "CG_MATH\vector3.h"
#include "CG_MATH\EulerAngles.h"
#include "CG_MATH\MathUtil.h"
#include "CG_MATH\Quaternion.h"
#include "Util\Color.h"

class RectAreaLight
{
public:

	// 公共操作

	RectAreaLight() :
		m_pos(CG_MATH::vector3(0.0f, 5.0f, -5.0f)),
		m_width(0.1f), m_height(0.1f),
		m_sampleWidth(2), m_sampleHeight(2),
		m_Dir(0, -CG_MATH::kPiOver2, 0), 
		m_La(0.02f, 0.2f, 0.0f), 
		m_Ld(0.04f, 0.4f, 0.0f),
		m_Ls(0.03f, 0.3f, 0.0f)
	{}
	
	// 公共数据
	float  m_width;	//half width
	float  m_height;	//half height
	 
	unsigned int m_sampleWidth;
	unsigned int m_sampleHeight;

	CG_MATH::vector3 m_pos;	//light position
	CG_MATH::EulerAngles m_Dir; // light Dir

	Color3f m_La;
	Color3f m_Ld;
	Color3f m_Ls;
	

};

