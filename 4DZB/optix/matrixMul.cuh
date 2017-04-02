#pragma once


/*
 *	column-major
 */
namespace OPTIX{
	__host__ __device__ float4 mul(float matrix4x4[16], float4 tVector)
	{
		float4 result;
		result.x = matrix4x4[0] * tVector.x + matrix4x4[4] * tVector.y + matrix4x4[8] * tVector.z + matrix4x4[12] * tVector.w;
		result.y = matrix4x4[1] * tVector.x + matrix4x4[5] * tVector.y + matrix4x4[9] * tVector.z + matrix4x4[13] * tVector.w;
		result.z = matrix4x4[2] * tVector.x + matrix4x4[6] * tVector.y + matrix4x4[10] * tVector.z + matrix4x4[14] * tVector.w;
		result.w = matrix4x4[3] * tVector.x + matrix4x4[7] * tVector.y + matrix4x4[11] * tVector.z + matrix4x4[15] * tVector.w;
		return result;
	}
};