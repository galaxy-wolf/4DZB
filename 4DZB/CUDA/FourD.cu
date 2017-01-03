#include <helper_math.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <vector>
#include <string>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "myMath.cuh"

#include "PrivateDefs.hpp"
#include "channel.h"
#include "CudaBuffer.h"
#include "layeredSurface.h"
#include "OpenGLinteroperate.h" // 定义数据struct

#include "Constants.hpp" // 常量值；

#include "../../debug/debug.h"
#include "../../image/SaveBMP.h"

#include "FourD.h"
//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------
__constant__ FDStaticParams    c_fdStaticParams;
__constant__ FDDynamicParams    c_fdDynamicParams;
__constant__ FDLightPlaneParams    c_fdLightPlaneParams;
__device__   FDAtomics   g_fdAtomics;



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
FDDynamicParams    h_fdDynamicParams;
FDLightPlaneParams    h_fdLightPlaneParams;

Layered2DSurfaceManager sampleRectangleSurfManager;

// gl interoperate
std::vector<FD::FDModel> FDscene;
//FD::FDSample			 FDsample;
FD::FDResult			 FDresult;


// config
bool m_enable_backfacecull = true;

int m_viewWidth = 0, m_viewHeight = 0, m_viewWidth_LOG2UP;
int m_maxTriangleNum = 0;
int m_maxVerticesNum = 0;
int m_triangleNum = 0;
int m_verteicesNum = 0;
// 1D， 真正个数为 m_binNum * m_binNum;
int m_binNum;
int m_lightResWidth;
int m_lightResHeight;
int m_lightResNum;
int m_validBinNum;

// device info
int m_CTA_num = 1;
int m_deviceID = 0;
int m_maxGridDimX = 0;
int m_maxBlockDimX = 0;

// buffers
FD::Buffer<float4> verticesRectangleBuffer;
FD::Buffer<float4> triVertexBuffer;
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
FD::Buffer<int> binSamplePairBinBuffer;
FD::Buffer<int> binSamplePairSampleBuffer;

FD::Buffer<int> isBinValidBuffer;
FD::Buffer<int> isBinValidPrefixSumBuffer;
FD::Buffer<int> validBinBuffer;

FD::Buffer<float> AABB3DReduceBuffer;

//valid buffer
FD::Buffer<int> validBuffer;

// aabb
float4 modelRectangleAABB[2];
float4 sampleRectangleAABB[2];



// global atomic adrress;
void * g_fdAtomics_addr;




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

extern "C" void inline setDynamicParams()
{
	checkCudaErrors(cudaMemcpyToSymbol(c_fdDynamicParams, (void *)&h_fdDynamicParams, sizeof(FDDynamicParams)));
}

extern "C" void inline setLightPlaneParams()
{
	checkCudaErrors(cudaMemcpyToSymbol(c_fdLightPlaneParams, &(h_fdLightPlaneParams), sizeof(FDLightPlaneParams), 0, cudaMemcpyHostToDevice));
}

 //for release 
// 投影函数
#include "FourDprojection.inl"

// setup
#include "blockAABBReduce.inl"
#include "modelRectangleAABB.inl"
#include "samplerRectangleAABB.inl"
#include "triangleSetup.inl"

// bin raster;
#include "binRaster.inl"
#include "bindSampleToBin.inl"
#include "RasterizeTriangleToBin.inl"

// cal;
#include "prepareBin.inl"
#include "calRef.inl"
#include "shadowCal.inl"

