#pragma once
#ifndef __PRIVATE_DEFS_HPP__
#define __PRIVATE_DEFS_HPP__
#include "../../light/AreaLightDesc.h"
#include "Util.hpp"
#include <vector_types.h>
//------------------------------------------------------------------------
// Device-side globals.
//------------------------------------------------------------------------

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

struct BinRange{ // 16Byte
	ushort2 start;
	ushort2 end;
	float minZ;
	float maxZ; // 
};

struct AreaLight
{
	// basic variables
	float  rightRadius;	//half width
	float  topRadius;	//half height
	float3 position;	//light position
	float3 viewDir;		//light view direction(normalized)
	float3 upDir;		//up direction(normalized)



	// 从基本变量生成
	float3 rightDir;		//left direction(normalized)
	float3 topDir;		//top direction(normalized)
	float3 x_delt;
	float3 y_delt;
	float3 upRightCornerPosition; // 可以从上面几项计算得到。
	float3 downLeftCornerPosition;
	float3 upLeftCornerPosition;
	float3 downRightCornerPosition;
	float2 cb, ce;

	// light plane 
	float  upRightMat[16];
	float2 RectangleFactor;

	// cal ref;
	float3 lightPos[16*16];

};

struct FDStaticParams
{
	//////////////////////////////////////////////////////////////////////////
	// config part
	//////////////////////////////////////////////////////////////////////////
	bool enable_backfaceCull;

	//////////////////////////////////////////////////////////////////////////
	// light part
	//////////////////////////////////////////////////////////////////////////
	AreaLight light;


	////////////////////////////////////////////////////////////////////////
	/// surfaces, textures;
	////////////////////////////////////////////////////////////////////////
	cudaTextureObject_t verticesRectangleTex;		/// linear memory can only use tex1dfetch()
	cudaTextureObject_t triangleVertexTex;		/// linear memory can only use tex1dfetch()
	cudaTextureObject_t triangleAABBTex;			/// linear memory can only use tex1dfetch()

	cudaTextureObject_t binSampleMaxZTex;
	cudaSurfaceObject_t sampleRectangleSurf;



	//-------------------------------------//
	//		triangle part
	//-------------------------------------//
	// Common.

	S32         numTris;
	S32			numVertices;
	CUdeviceptr vertexBuffer;       // numVerts * ShadedVertexSubclass
	CUdeviceptr indexBuffer;        // numTris * int3
	S32			vertexSizeFloat;

	S32         viewportWidth;      // 
	S32         viewportHeight;
	S32			viewportWidth_LOG2_UP;

	S32         widthBins;          // 16*16 must be 2^n
	S32			widthBins_LOG2;
	S32         heightBins;         // 
	S32         numBins;            // widthBins * heightBins

	CUdeviceptr aabbTempBuffer;
	CUdeviceptr validTempBuffer;

	// Setup output / bin input.
	CUdeviceptr verticesRectangleBuffer;
	CUdeviceptr triPositionBuffer;
	CUdeviceptr triAABBBuffer;
	CUdeviceptr triBinRangeBuffer;


	// Bin output / tile raster input
	CUdeviceptr triBinNum;
	CUdeviceptr triBinNumPrefixSum;
	CUdeviceptr binTriStart;			// triangle from binStart to binEnd;
	CUdeviceptr binTriEnd;


	//-------------------------------------//
	//		pixel part
	//-------------------------------------//
	// bind to tile output
	CUdeviceptr pb_pair_pixel;			// pixelID
	CUdeviceptr pb_pair_bin;			// tielID
	CUdeviceptr binPixelStart;
	CUdeviceptr binPixelEnd;			// 
	CUdeviceptr binPixelMaxZperBin;


	CUdeviceptr isBinValidBuffer;
	CUdeviceptr isBinValidPrefixSumBuffer;
	CUdeviceptr validBinBuffer;

	// shadow output ;
	S32			lightResWidth;
	S32			lightResHeight;
	S32			lightResNum;
	CUdeviceptr shadowValue;
};

struct FDDynamicParams{
	CUdeviceptr bt_pair_bin;			// triangle and bin pair;
	CUdeviceptr bt_pair_tri;
};

struct FDLightPlaneParams{

	float3 begin;
	float3 end;
	float2 factor; // 

};
//------------------------------------------------------------------------

struct FDAtomics
{

	// Bin.

	//S32         binCounter;         // = 0
	//S32         numBinSegs;         // = 0

	// Coarse.

	//S32         coarseCounter;      // = 0
	//S32         numTileSegs;        // = 0
	//S32         numActiveTiles;     // = 0

	// Fine.

	//S32         fineCounter;        // = 0



	// shadow computer
	S32			shadowComputerCounter;
};

#endif // __PRIVATE_DEFS_HPP__