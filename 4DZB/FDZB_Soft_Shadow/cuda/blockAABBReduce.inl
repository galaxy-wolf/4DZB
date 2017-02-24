#ifndef __BLOCK_AABB_REDUCE_CUH__
#define __BLOCK_AABB_REDUCE_CUH__

#define REDUCE_BLOCK_SIZE 512
//
//#define fadd(a, b) (a+b)

//// max block size is 512!
//#define BLOCK_REDUCE(opos, sdata, mySum, op, blockSize)\
//sdata[threadIdx.x] = mySum;\
//__syncthreads();\
//if (blockSize >= 512) { if (threadIdx.x < 256) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 256]); } __syncthreads(); }\
//if (blockSize >= 256) { if (threadIdx.x < 128) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 128]); } __syncthreads(); }\
//if (blockSize >= 128) { if (threadIdx.x <  64) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 64]); } __syncthreads(); }\
//	\
//if (threadIdx.x < 32)\
//{\
//	if (blockSize >= 64) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 32]); }\
//	if (blockSize >= 32) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 16]);}\
//	if (blockSize >= 16) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 8]); }\
//	if (blockSize >= 8) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 4]); }\
//	if (blockSize >= 4) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 2]); }\
//	if (blockSize >= 2) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 1]); }\
//}\
//\
//if (threadIdx.x == 0)\
//*opos = sdata[0];\
//	__syncthreads();\

//// 定义min 的float 版
//inline __device__ float min(float a, float b)
//{
//	return fminf(a, b);
//}
//inline __device__ float max(float a, float b)
//{
//	return fmaxf(a, b);
//}

template<typename T>
class OperatorMin
{
public:
	inline __device__ T operator() (const T a, const T b) const { return min(a, b); }
};

template<typename T>
class OperatorMax
{
public:
	inline __device__ T operator() (const T a, const T b) const { return max(a, b); }
};

template<typename T>
class OperatorAdd
{
public:
	inline __device__ T operator() (const T a, const T b) const { return a + b; }
};

template<typename T, typename Oper, size_t blockSize>
__device__ __forceinline void block_reduce(T* const odata, T mySum)
{
	Oper op;
	__shared__  volatile T sdata[blockSize]; // must have "volatile"

	sdata[threadIdx.x] = mySum;
	__syncthreads();
	if (blockSize >= 1024) { if (threadIdx.x < 512) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 512]); } __syncthreads(); }
	if (blockSize >= 512) { if (threadIdx.x < 256) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (threadIdx.x < 128) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (threadIdx.x <  64) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 64]); } __syncthreads(); }

	if (threadIdx.x < 32)
	{
		if (blockSize >= 64) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 32]); }
		if (blockSize >= 32) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 16]); }
		if (blockSize >= 16) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 8]); }
		if (blockSize >= 8) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 4]); }
		if (blockSize >= 4) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 2]); }
		if (blockSize >= 2) { sdata[threadIdx.x] = mySum = op(mySum, sdata[threadIdx.x + 1]); }
	}

	if (threadIdx.x == 0)
		*odata = sdata[0];
	__syncthreads();
}

/**********************************************

n 表示aabb的个数
**********************************************/
enum REDUCE_TYPE { reduceMaxMinOnly, reduceXYsumOnly, reduceBoth, INVALID };
template<REDUCE_TYPE reduceType>
__global__ void AABBReduce_kernel(int n)
{
	float * aabbTemp = (float *)c_fdStaticParams.aabbTempBuffer;

	float myXmin = FD_F32_MAX, myXmax = -FD_F32_MAX;
	float myYmin = FD_F32_MAX, myYmax = -FD_F32_MAX;
	float myZmin = FD_F32_MAX, myZmax = -FD_F32_MAX;
	float myXsum = 0.0f, myYsum = 0.0f;

	uint pos = (blockIdx.x * blockDim.x + threadIdx.x);

	if (pos < n) {

		if (reduceType == reduceMaxMinOnly || reduceType == reduceBoth)
		{
			myXmin = aabbTemp[(pos << 3) | 0x00];
			myYmin = aabbTemp[(pos << 3) | 0x01];
			myZmin = aabbTemp[(pos << 3) | 0x02];

			myXmax = aabbTemp[(pos << 3) | 0x04];
			myYmax = aabbTemp[(pos << 3) | 0x05];
			myZmax = aabbTemp[(pos << 3) | 0x06];
		}

		if (reduceType == reduceBoth)
		{
			myXsum = aabbTemp[(pos << 3) | 0x03];
			myYsum = aabbTemp[(pos << 3) | 0x07];
		}

		if (reduceType == reduceXYsumOnly)
		{
			myXsum = aabbTemp[(pos << 1) | 0x00];
			myYsum = aabbTemp[(pos << 1) | 0x01];
		}
	}
	int opos = blockIdx.x;
	if (reduceType == reduceMaxMinOnly || reduceType == reduceBoth)
	{
		block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x00], myXmin);
		block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x01], myYmin);
		block_reduce<float, OperatorMin<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x02], myZmin);

		block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x04], myXmax);
		block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x05], myYmax);
		block_reduce<float, OperatorMax<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x06], myZmax);
	}
	if (reduceType == reduceBoth)
	{
		block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x03], myXsum);
		block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 3) | 0x07], myYsum);
	}

	if (reduceType == reduceXYsumOnly)
	{
		block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 1) | 0x00], myXsum);
		block_reduce<float, OperatorAdd<float>, REDUCE_BLOCK_SIZE>(&aabbTemp[(opos << 1) | 0x01], myYsum);
	}
}

#endif // __BLOCK_AABB_REDUCE_CUH__