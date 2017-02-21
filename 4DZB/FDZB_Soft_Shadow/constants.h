#pragma once
#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__


// 不要使用define； 使用const 变量。
//------------------------------------------------------------------------
// user set;
//------------------------------------------------------------------------
//#define FD_LIGHT_RES_HEIGHT  16// 灯的分辨率 取值范围1~64
//#define FD_LIGHT_RES_WIDTH  16// 灯的分辨率 取值范围1~64

const int FD_MAX_LP_GRID_SIZE_LOG2 = 7;    // bin尺寸的上限， 目前为512*512， 根据三角形和场景光源的情况自动调整；

const int FD_MAX_BT_PIAR_Num = 50 * 1024 * 1024; // 最多有maxBTpairNum 个三角形pair, 过大会影响排序时间，


const int FD_MAX_REGISTER_PER_THREAD = 32;// 在求阴影时每个线程使用51个寄存器， 在gtx960下 每个sm 可以开 40 个warps；

const int FD_MAX_AABB_REDUCE_TEMP_SIZE = 100 * 1024 * 1024 / 512;// 最多可以使用1亿个元素； 这里512 是第一次reduce 的block size；

#define FD_MAXBINS_LOG2			4 // 4D plane / binsize.
#define FD_BIN_LOG2				3 // binsize / tileSize.

#define FD_BIN_TRI_SEG_LOG2			9		// 32-bit entries.   512 个线程
//#define FD_TILE_SEG_LOG2		5		// 32-bit entries.
#define FD_SAMPLE_TILE_SEG_LOG2 4  // 16个 sample 为1组


#define FD_BIN_WARPS			16	
#define FD_COARSE_WARPS			32


// GPU 相关：
const int FD_MAX_WARPS_PER_SM = 64;
const int FD_MAX_BLOCK_PER_SM = 32;
const int FD_MIN_BLOCK_PER_SM = 2;
const int FD_SHARED_MEM_PER_SM = 96 * 1024; //bytes;



//------------------------------------------------------------------------
//generated;
//------------------------------------------------------------------------
const int FD_MAX_LP_GRID_SIZE = (1 << FD_MAX_LP_GRID_SIZE_LOG2);



typedef unsigned int U32;
typedef unsigned long long U64;
//#if (FD_LIGHT_RES_WIDTH <=32 && FD_LIGHT_RES_HEIGHT <=32)
//const int FD_SC_BIT_MASK_PER_LINE_SIZE_U32 = 1; // 每行使用一个U32 存储
//#define FD_SC_USE_U32_LINE  1
//const U32 FD_SC_LIGHT_WIDTH_BIT = ((U32)1 << FD_LIGHT_RES_WIDTH) - 1;
//const U32 FD_SC_LIGHT_HEIGHT_BIT = ((U32)1 << FD_LIGHT_RES_HEIGHT) - 1;
//#else
//const int FD_SC_BIT_MASK_PER_LINE_SIZE_U32=	2; // 每行使用一个U64 存储；
//#define FD_SC_USE_U64_LINE	1
//const U64 FD_SC_LIGHT_WIDTH_BIT  =((U64)1 << FD_LIGHT_RES_WIDTH) -1;
//const U64 FD_SC_LIGHT_HEIGHT_BIT=	((U64)1<<FD_LIGHT_RES_HEIGHT) -1;
//#endif
//
//const int  FD_SC_SHARE_MEMORY_PER_SAMPLE = FD_SC_SAMPLE_DATA_SIZE_U32 + (FD_LIGHT_RES_HEIGHT + 1)* FD_SC_BIT_MASK_PER_LINE_SIZE_U32;//每个sample 占用的内存
//const int  FD_SC_SHARE_MEMORY_MAX_SAMPLE_NUM = FD_SC_SHARE_MEMORY_LEFT_SIZE_U32 / FD_SC_SHARE_MEMORY_PER_SAMPLE;// 每个block 可以存放的sample 数目；


#define FD_MAXBINS_SIZE				(1<<FD_MAXBINS_LOG2)
#define FD_MAXBINS_SQR				(1<<(FD_MAXBINS_LOG2*2))
#define FD_BIN_SIZE					(1<<FD_BIN_LOG2)
#define FD_BIN_SQR					(1<<(FD_BIN_LOG2*2))

#define FD_MAXTILES_LOG2			(FD_MAXBINS_LOG2 + FD_BIN_LOG2)
#define FD_MAXTILES_SIZE			(1<<FD_MAXTILES_LOG2)
#define FD_MAXTILES_SQR				(1<<(FD_MAXTILES_LOG2*2))

#define FD_BIN_SEG_SIZE				(1<<FD_BIN_SEG_LOG2)
//#define FD_TILE_SEG_SIZE			(1<<FD_TILE_SEG_LOG2)
#define FD_SAMPLE_TILE_SEG_SIZE     (1<<FD_SAMPLE_TILE_SEG_LOG2)


//------------------------------------------------------------------------




//------------------------------------------------------------------------

#endif //__CONSTANTS_HPP__