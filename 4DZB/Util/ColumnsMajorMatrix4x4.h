#pragma once

#include "..\CG_MATH/Matrix4x4.h"
#include "..\CG_MATH/Matrix3x4.h"

using namespace CG_MATH;

class ColumnsMajorMatrix4x4
{
public:
	ColumnsMajorMatrix4x4(const Matrix3x4& a) {
		m[0][0] = a.m11; m[0][1] = a.m21; m[0][2] = a.m31; m[0][3] = 0.0f;
		m[1][0] = a.m12; m[1][1] = a.m22; m[1][2] = a.m32; m[1][3] = 0.0f;
		m[2][0] = a.m13; m[2][1] = a.m23; m[2][2] = a.m33; m[2][3] = 0.0f;
		m[3][0] = a.tx; m[3][1] = a.ty; m[3][2] = a.tz; m[3][3] = 1.0f;
	}

	ColumnsMajorMatrix4x4(const Matrix4x4& a) {
		m[0][0] = a.m11; m[0][1] = a.m21; m[0][2] = a.m31; m[0][3] = a.m41;
		m[1][0] = a.m12; m[1][1] = a.m22; m[1][2] = a.m32; m[1][3] = a.m42;
		m[2][0] = a.m13; m[2][1] = a.m23; m[2][2] = a.m33; m[2][3] = a.m43;
		m[3][0] = a.m14; m[3][1] = a.m24; m[3][2] = a.m34; m[3][3] = a.m44;
	}


	float m[4][4];
};