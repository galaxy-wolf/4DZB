#pragma once

#include "Matrix3x4.h"

namespace CG_MATH
{
	class Matrix3x4;

	//----------------------------------------------------------------------------------------
	//Matrix4x4类
	//实现4x4转换矩阵

	class Matrix4x4
	{
	public:

		// 公共数据

		//矩阵的值
		
		float m11, m12, m13, m14;
		float m21, m22, m23, m24;
		float m31, m32, m33, m34;
		float m41, m42, m43, m44;

		// 公共操作

		//置为单位矩阵

		void identity();

		// 从3x4矩阵构造
		
		Matrix4x4(Matrix3x4 &a) {

			// 直接赋值

			m11 = a.m11; m12 = a.m12; m13 = a.m13; m14 = a.tx;
			m21 = a.m21; m22 = a.m22; m23 = a.m23; m24 = a.ty;
			m31 = a.m31; m32 = a.m32; m33 = a.m33; m34 = a.tz;
			m41 = 0.0f;  m42 = 0.0f;  m43 = 0.0f;  m44 = 1.0f;
		}

		// 默认构造函数
		
		Matrix4x4() = default;

		// 构造投影变换矩阵

		void setupPerspective(float fov, float aspect, float near, float far);
		void setupFrustum(float left, float right, float bottom, float top, float near, float far);

	};

	// 运算符* 用来连接矩阵，乘法的顺序从左向右，与变换顺序相反。

	Matrix4x4 operator*(const Matrix4x4 &a, const Matrix4x4 &m);

	// 运算符*= 保持和c++标准语法一致。

	Matrix4x4 &operator*= (Matrix4x4 &a, const Matrix4x4 &m);

	// 计算矩阵得逆

	Matrix4x4 inverse(const Matrix4x4 &m);

}