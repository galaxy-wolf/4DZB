#pragma once
#ifndef __PRIVATE_DEFS_HPP__
#define __PRIVATE_DEFS_HPP__
#include "../util/Util.h"
#include <vector_types.h>
//------------------------------------------------------------------------
// Device-side globals.
//------------------------------------------------------------------------

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

struct GPU_INFO{
	int deviceID;

	int maxRegisterPerSM;
	int maxRegisterPerBlock;

	int maxBlockDimX;
	int maxBlockDimY;
	int maxGridDimX;

	int warpSize;

	int maxSharedMemPerSM;
	int maxSharedMemPerBlock;

	int maxThreadsPerBlock;

	int SM_num;
	int bigBlockPerSM;
};

struct BinRange{ // 16Byte
	ushort2 start;
	ushort2 end;
	float minZ;
	float maxZ; // 
};

struct __align__(64) TriangleData
{
	float3 v0;
	float3 v1;
	float3 v2;

	float maxz;
	float minz;
	float2 aabb2Dmin;
	float2 aabb2Dmax;
};

struct __align__(32) SampleData
{
	float3 pos;
	float projectionZ;
	float2 aabb2Dmin;
	float2 aabb2Dmax;
};



struct AreaLight
{
	// basic variables
	float  rightRadius;	//half width
	float  topRadius;	//half height
	float3 position;	//light position
	float3 viewDir;		//light view direction(normalized)
	float3 upDir;		//up direction(normalized)
	int lightResWidth;		// area light res;
	int lightResHeight;
	//int lightResWidthSubOne;
	int lightResNum;
	int lightBitMaskSizeT;

	U32 U32lightWidthBit;
//	U32 U32lightHeightBit;
	U64 U64lightWidthBit;
//	U64 U64lightHeightBit;


	// 从基本变量生成
	float3 rightDir;		//left direction(normalized)
	float3 topDir;		//top direction(normalized)
	float3 x_delt;
	float3 y_delt;
	float3 upRightCornerPosition; // 可以从上面几项计算得到。
	float3 downLeftCornerPosition;
	float3 upLeftCornerPosition;
	float3 downRightCornerPosition;
	//float2 cb, ce;

	// light plane 
	float  upRightMat[16];
	float2 RectangleFactor;

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

	cudaTextureObject_t verticesTex; 

	//cudaTextureObject_t verticesRectangleTex;		/// linear memory can only use tex1dfetch()
	//cudaTextureObject_t triangleVertexTex;		/// linear memory can only use tex1dfetch()
	cudaTextureObject_t triangleAABBTex;			/// linear memory can only use tex1dfetch()
	cudaTextureObject_t triangleDataTex;

	cudaTextureObject_t sampleDataTex;

	cudaTextureObject_t binSampleMaxZTex;
	cudaTextureObject_t binSampleMinRangeTex;
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

	// move to light plane params.
	//S32         widthBins;          // 16*16 must be 2^n
	//S32			widthBins_LOG2;
	//S32         heightBins;         // 
	//S32         numBins;            // widthBins * heightBins

	CUdeviceptr aabbTempBuffer;
	CUdeviceptr validTempBuffer;

	// Setup output / bin input.
	//CUdeviceptr verticesRectangleBuffer;
	//CUdeviceptr triPositionBuffer;
	CUdeviceptr triDataBuffer;
	CUdeviceptr triAABBBuffer;
	CUdeviceptr triBinRangeBuffer;


	// Bin output / tile raster input
	CUdeviceptr triBinNum;
	CUdeviceptr triBinNumPrefixSum;
	CUdeviceptr bt_pair_bin;			// triangle and bin pair;
	CUdeviceptr bt_pair_tri;
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
	CUdeviceptr binPixelMinRangeperBin;


	CUdeviceptr isBinValidBuffer;
	CUdeviceptr isBinValidPrefixSumBuffer;
	CUdeviceptr validBinBuffer;

	CUdeviceptr shadowValue;
	float scenes_esp;
};

//struct FDDynamicParams{
//	
//};

struct FDLightPlaneParams{

	// bin params
	S32         widthBins;          // 16*16 must be 2^n
	S32			widthBins_LOG2;
	S32         heightBins;         // 
	S32         numBins;            // widthBins * heightBins

	float3 begin;
	float3 end;
	float2 factor; // 

};
//------------------------------------------------------------------------

struct FDAtomics
{
	// prepare to shadow compute
	S32			invalidSampleCounter;
	S32			invalidBT_PairCounter;

	// shadow computer
	S32			shadowComputerCounter;


	U64		allTriangleSampleCounter;
	U64		aabbCoverRectangleCounter;
	U64		shadowVolumeCoverCounter;
	U64		triangleBlockSampleCounter;
	U64		validBTpairCounter;
	U64     BTpairCounter;
};


#endif // __PRIVATE_DEFS_HPP__