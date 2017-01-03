#ifndef __MY_MATH_CUH__
#define __MY_MATH_CUH__

#include <helper_math.h>
__inline__ __host__ __device__ float2 make_float2(float4 a)
{
	return make_float2(a.x, a.y);
}

extern int __inline iDiviUp(int a, int b);

#endif// __MY_MATH_CUH__