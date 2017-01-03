#ifndef __FOURD_PROJECTION_CUH__
#define __FOURD_PROJECTION_CUH__
#include <cuda_runtime.h>
namespace FD{

	__host__ __device__ __inline__ void makeRectangle(float3 & rect, float upRightMat[16], float3 tVector)
	{
		rect.z = upRightMat[3] * tVector.x + upRightMat[7] * tVector.y + upRightMat[11] * tVector.z + upRightMat[15] * 1.0f;

		rect.x = upRightMat[0] * tVector.x + upRightMat[4] * tVector.y + upRightMat[8] * tVector.z + upRightMat[12] * 1.0f;
		rect.x = rect.x / rect.z * 0.5f + 0.5f;
		rect.y = upRightMat[1] * tVector.x + upRightMat[5] * tVector.y + upRightMat[9] * tVector.z + upRightMat[13] * 1.0f;
		rect.y = rect.y / rect.z * 0.5f + 0.5f;
	}
}
#endif // __FOURD_PROJECTION_CUH__