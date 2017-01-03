#pragma once

#ifndef __FOURD_CUH__
#define __FOURD_CUH__

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include <vector>
#include <string>

#include "myMath.cuh"

#include "PrivateDefs.hpp"
#include "CudaBuffer.h"
#include "layeredSurface.h"
#include "OpenGLinteroperate.h" // 定义数据struct


//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------



//------------------------------------------------------------------------
// host Globals.
//------------------------------------------------------------------------
extern FDStaticParams    h_fdStaticParams;
extern FDDynamicParams    h_fdDynamicParams;
extern FDLightPlaneParams    h_fdLightPlaneParams;

extern Layered2DSurfaceManager sampleRectangleSurfManager;

// gl interoperate
extern std::vector<FD::FDModel> FDscene;
//extern "C" //FD::FDSample			 FDsample;
extern FD::FDResult			 FDresult;


// config
extern bool m_enable_backfacecull;

extern int m_viewWidth, m_viewHeight, m_viewWidth_LOG2UP;
extern int m_maxTriangleNum;
extern int m_maxVerticesNum;
extern int m_triangleNum;
extern int m_verteicesNum;
extern int m_binNum;
extern int m_lightResWidth;
extern int m_lightResHeight;
extern int m_lightResNum;

// device info
extern int m_CTA_num;
extern int m_deviceID;
extern int m_maxGridDimX;
extern int m_maxBlockDimX;

// buffers
extern FD::Buffer<float4> verticesRectangleBuffer;
extern FD::Buffer<float4> triVertexBuffer;
extern FD::Buffer<float4> triAABBBuffer;
extern FD::Buffer<struct BinRange> triBinRangeBuffer;

extern FD::Buffer<int> binTriStartBuffer;
extern FD::Buffer<int> binTriEndBuffer;
extern FD::Buffer<int> binTriPairBinBuffer; //动态开辟
extern FD::Buffer<int> binTriPairTriBuffer; //动态开辟
extern FD::Buffer<int> triPairNumBuffer;
extern FD::Buffer<int> triPairNumPrefixSumBuffer;

extern FD::Buffer<int> binSampleStartBuffer;
extern FD::Buffer<int> binSampleEndBuffer;
extern FD::Buffer<float> binSampleMaxZBuffer;
extern FD::Buffer<int> binSamplePairBinBuffer; 
extern FD::Buffer<int> binSamplePairSampleBuffer;

extern FD::Buffer<int> isBinValidBuffer;
extern FD::Buffer<int> isBinValidPrefixSumBuffer;
extern FD::Buffer<int> validBinBuffer;

extern FD::Buffer<float> AABB3DReduceBuffer;

// aabb
extern float4 modelRectangleAABB[2];
extern float4 sampleRectangleAABB[2];

//valid buffer
extern FD::Buffer<int> validBuffer;

// get cuda addr;
extern "C" void inline getG_Atomic_addr();

/// sample position;
namespace FDSample{
	extern GLuint GLsamplePositionTex;
	extern GLuint GLsampleNormalTex;

	extern cudaGraphicsResource * positionRes;
	extern cudaArray_t positionArray;
	extern cudaGraphicsResource * normalRes;
	extern cudaArray_t normalArray;

	void setGLtexture(GLuint positionTex, GLuint normalTex);
	
	void registerGLtexture();
	
	void unregisterGLtexture();
	void mapGLtexture();
	void unmapGLtexture();
}

//\\\\\\\\\\\\\\\\\\\\\\\\\\\
// cuda 提供的接口函数；
//\\\\\\\\\\\\\\\\\\\\\\\\\\\

extern "C" void inline setStaticParams();
extern "C" void inline setDynamicParams();
extern "C" void inline setLightPlaneParams();

bool lightPlaneParamCal();
void setupTriangleVertex();
void bindSampleToBin();
void countTriBinPairNum();
void bindTriToBin();
void shadowCal();
void refCal();
#endif// __FOURD_CUH__
