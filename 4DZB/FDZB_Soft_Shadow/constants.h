#pragma once
#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__


// ��Ҫʹ��define�� ʹ��const ������
//------------------------------------------------------------------------
// user set;
//------------------------------------------------------------------------
//#define FD_LIGHT_RES_HEIGHT  16// �Ƶķֱ��� ȡֵ��Χ1~64
//#define FD_LIGHT_RES_WIDTH  16// �Ƶķֱ��� ȡֵ��Χ1~64

const int FD_MAX_LP_GRID_SIZE_LOG2 = 7;    // bin�ߴ�����ޣ� ĿǰΪ512*512�� ���������κͳ�����Դ������Զ�������

const int FD_MAX_BT_PIAR_Num = 50 * 1024 * 1024; // �����maxBTpairNum ��������pair, �����Ӱ������ʱ�䣬


const int FD_MAX_REGISTER_PER_THREAD = 32;// ������Ӱʱÿ���߳�ʹ��51���Ĵ����� ��gtx960�� ÿ��sm ���Կ� 40 ��warps��

const int FD_MAX_AABB_REDUCE_TEMP_SIZE = 100 * 1024 * 1024 / 512;// ������ʹ��1�ڸ�Ԫ�أ� ����512 �ǵ�һ��reduce ��block size��

#define FD_MAXBINS_LOG2			4 // 4D plane / binsize.
#define FD_BIN_LOG2				3 // binsize / tileSize.

#define FD_BIN_TRI_SEG_LOG2			9		// 32-bit entries.   512 ���߳�
//#define FD_TILE_SEG_LOG2		5		// 32-bit entries.
#define FD_SAMPLE_TILE_SEG_LOG2 4  // 16�� sample Ϊ1��


#define FD_BIN_WARPS			16	
#define FD_COARSE_WARPS			32


// GPU ��أ�
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
//const int FD_SC_BIT_MASK_PER_LINE_SIZE_U32 = 1; // ÿ��ʹ��һ��U32 �洢
//#define FD_SC_USE_U32_LINE  1
//const U32 FD_SC_LIGHT_WIDTH_BIT = ((U32)1 << FD_LIGHT_RES_WIDTH) - 1;
//const U32 FD_SC_LIGHT_HEIGHT_BIT = ((U32)1 << FD_LIGHT_RES_HEIGHT) - 1;
//#else
//const int FD_SC_BIT_MASK_PER_LINE_SIZE_U32=	2; // ÿ��ʹ��һ��U64 �洢��
//#define FD_SC_USE_U64_LINE	1
//const U64 FD_SC_LIGHT_WIDTH_BIT  =((U64)1 << FD_LIGHT_RES_WIDTH) -1;
//const U64 FD_SC_LIGHT_HEIGHT_BIT=	((U64)1<<FD_LIGHT_RES_HEIGHT) -1;
//#endif
//
//const int  FD_SC_SHARE_MEMORY_PER_SAMPLE = FD_SC_SAMPLE_DATA_SIZE_U32 + (FD_LIGHT_RES_HEIGHT + 1)* FD_SC_BIT_MASK_PER_LINE_SIZE_U32;//ÿ��sample ռ�õ��ڴ�
//const int  FD_SC_SHARE_MEMORY_MAX_SAMPLE_NUM = FD_SC_SHARE_MEMORY_LEFT_SIZE_U32 / FD_SC_SHARE_MEMORY_PER_SAMPLE;// ÿ��block ���Դ�ŵ�sample ��Ŀ��


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