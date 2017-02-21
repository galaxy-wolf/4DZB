#include <helper_math.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <vector>
#include <string>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>


#include "FourD.h"
#include "../constants.h"
#include "../../Util/SaveBMP.h"

__inline__ __host__ __device__ float2 make_float2(float4 a)
{
	return make_float2(a.x, a.y);
}


//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------
__constant__ FDStaticParams    c_fdStaticParams;
//__constant__ FDDynamicParams    c_fdDynamicParams;
__constant__ FDLightPlaneParams    c_fdLightPlaneParams;
__constant__ float2					c_fdRectangleSubConstant;
__device__   FDAtomics   g_fdAtomics;
__constant__ float3 c_REF_CAL_lightPos[32 * 32];
float3 h_REF_CAL_lightPos[32 * 32];


// from GL  全局常量
texture<float4, 2, cudaReadModeElementType> samplePositionTex;
texture<float4, 2, cudaReadModeElementType> sampleNormalTex;
//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------



//------------------------------------------------------------------------
// host Globals.
//------------------------------------------------------------------------
FDStaticParams    h_fdStaticParams;
//FDDynamicParams    h_fdDynamicParams;
FDLightPlaneParams    h_fdLightPlaneParams;
float2				h_fdRectangleSubConstant;

Layered2DSurfaceManager sampleRectangleSurfManager;

// gl interoperate
std::vector<FD::FDModel> FDscene;
//FD::FDSample			 FDsample;
FD::FDResult			 FDresult;


// config
bool m_enable_backfacecull = true;
float m_scenes_esp = 1e-6;

int m_viewWidth = 0, m_viewHeight = 0, m_viewWidth_LOG2UP;
int m_maxTriangleNum = 0;
int m_maxVerticesNum = 0;
int m_triangleNum = 0;
int m_verteicesNum = 0;
// 1D， 真正个数为 m_binNum * m_binNum;
//int m_binNum;


int m_validTriangleNum;// without backFace cull;
int m_validSampleNum;
int m_validBT_PairNum;

int m_validBinNum;

// device info
GPU_INFO my_device;
int m_maxShadowCalWarpNumPerBlock;


//int m_CTA_num = 1;
//int m_deviceID = 0;
//int m_maxGridDimX = 0;
//int m_maxBlockDimX = 0;

// buffers
//FD::Buffer<float4> verticesRectangleBuffer;
//FD::Buffer<float4> triVertexBuffer;

FD::Buffer<float4> triDataBuffer;
//FD::Buffer<float4> sampleDatabuffer;

FD::Buffer<float4> triAABBBuffer;
FD::Buffer<struct BinRange> triBinRangeBuffer;

FD::Buffer<int> binTriStartBuffer;
FD::Buffer<int> binTriEndBuffer;
FD::Buffer<int> binTriPairBinBuffer; //动态开辟
FD::Buffer<int> binTriPairTriBuffer; //动态开辟
FD::Buffer<int> triPairNumBuffer;
FD::Buffer<int> triPairNumPrefixSumBuffer;

FD::Buffer<int> binSampleStartBuffer;
FD::Buffer<int> binSampleEndBuffer;
FD::Buffer<float> binSampleMaxZBuffer;
FD::Buffer<float4> binSampleMinRangeBuffer;
FD::Buffer<int> binSamplePairBinBuffer;
FD::Buffer<int> binSamplePairSampleBuffer;

FD::Buffer<int> isBinValidBuffer;
FD::Buffer<int> isBinValidPrefixSumBuffer;
FD::Buffer<int> validBinBuffer;

FD::Buffer<float> AABB3DReduceBuffer;

//valid buffer
FD::Buffer<int> validBuffer;



//// linked- list;
//
//FD::LinkedList<unsigned int> sampleTileLinkedList;

// aabb
float4 modelRectangleAABB[2];
float4 sampleRectangleAABB[2];



// global atomic adrress;
void * g_fdAtomics_addr;
void * g_fdCounters_addr;



/// sample position;
namespace FDSample{
	GLuint GLsamplePositionTex;
	GLuint GLsampleNormalTex;

	cudaGraphicsResource * positionRes;
	cudaArray_t positionArray;
	cudaGraphicsResource * normalRes;
	cudaArray_t normalArray;

	void setGLtexture(GLuint positionTex, GLuint normalTex)
	{
		GLsamplePositionTex = positionTex;
		GLsampleNormalTex = normalTex;
	}
	void registerGLtexture()
	{
		checkCudaErrors(cudaGraphicsGLRegisterImage(&positionRes, GLsamplePositionTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
		checkCudaErrors(cudaGraphicsGLRegisterImage(&normalRes, GLsampleNormalTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
	}
	void unregisterGLtexture()
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(positionRes));
		checkCudaErrors(cudaGraphicsUnregisterResource(normalRes));
	}
	void mapGLtexture()
	{
		checkCudaErrors(cudaGraphicsMapResources(1, &positionRes, NULL));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&positionArray, positionRes, 0, 0));
		// 2D 纹理
		samplePositionTex.addressMode[0] = cudaAddressModeWrap;
		samplePositionTex.addressMode[1] = cudaAddressModeWrap;
		samplePositionTex.filterMode = cudaFilterModePoint;
		samplePositionTex.normalized = false;  // access with normalized texture coordinates

		// 绑定到CUDA纹理
		checkCudaErrors(cudaBindTextureToArray(samplePositionTex, positionArray, FD::float4_channelDesc));

		checkCudaErrors(cudaGraphicsMapResources(1, &normalRes, NULL));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&normalArray, normalRes, 0, 0));
		// 2D 纹理
		sampleNormalTex.addressMode[0] = cudaAddressModeWrap;
		sampleNormalTex.addressMode[1] = cudaAddressModeWrap;
		sampleNormalTex.filterMode = cudaFilterModePoint;
		sampleNormalTex.normalized = false;  // access with normalized texture coordinates
		// 绑定到CUDA纹理
		checkCudaErrors(cudaBindTextureToArray(sampleNormalTex, normalArray, FD::float4_channelDesc));
	}
	void unmapGLtexture()
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &positionRes, NULL));
		checkCudaErrors(cudaUnbindTexture(samplePositionTex));

		checkCudaErrors(cudaGraphicsUnmapResources(1, &normalRes, NULL));
		checkCudaErrors(cudaUnbindTexture(sampleNormalTex));
	}
}

extern "C" void inline getG_Atomic_addr()
{
	checkCudaErrors(cudaGetSymbolAddress((void **)&g_fdAtomics_addr, g_fdAtomics));
}

extern "C" void inline setStaticParams()
{
	checkCudaErrors(cudaMemcpyToSymbol(c_fdStaticParams, (void *)&h_fdStaticParams, sizeof(FDStaticParams)));
}

//extern "C" void inline setDynamicParams()
//{
//	checkCudaErrors(cudaMemcpyToSymbol(c_fdDynamicParams, (void *)&h_fdDynamicParams, sizeof(FDDynamicParams)));
//}

extern "C" void inline setLightPlaneParams()
{
	checkCudaErrors(cudaMemcpyToSymbol(c_fdLightPlaneParams, &(h_fdLightPlaneParams), sizeof(FDLightPlaneParams), 0, cudaMemcpyHostToDevice));
}

extern "C" void inline setRectangleSubConstant()
{
	checkCudaErrors(cudaMemcpyToSymbol(c_fdRectangleSubConstant, &(h_fdRectangleSubConstant), sizeof(float2), 0, cudaMemcpyHostToDevice));
}

extern "C" void inline setRefLightPos()
{
	checkCudaErrors(cudaMemcpyToSymbol(c_REF_CAL_lightPos, &h_REF_CAL_lightPos, sizeof(c_REF_CAL_lightPos), 0, cudaMemcpyHostToDevice));
}

 //for release 
// 公共函数
#include "FDprojection.inl"
#include "blockAABBReduce.inl"
#include "binRaster.inl"



// step 0
#include "Step0_triangleSetup.inl"

// step1 defer rendering scene

// step 2 define Frame buffer

#include "Step2_3_defineFDbuffer.inl"

// step 3 bind sample
#include "Step3_bindSampleToBin.inl"

// step 4 raster triangle
#include "Step4_RasterizeTriangleToBin.inl"

// step 5 cal shadow
#include "Step5_1_prepareBin.inl"
#include "Step5_2_shadowCal.inl"
#include "Step5_2_calRef.inl"


