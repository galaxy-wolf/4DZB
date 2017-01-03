#pragma once
#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__
//------------------------------------------------------------------------

#define FD_MAXBINS_LOG2         5       // bin 的个数
#define FD_BIN_LOG2             4       // 每个bin中有多少tile

#define FD_TILE_SEG_LOG2        9       // 32-bit entries. tile 链表的segment大小

#define FD_TRIANGLE_QUEUE_LOG2    9      // Triangles. 在share memory 中存放triangle id 的个数

#define FD_RASTERIZE_WARPS         16      // Must be a power of two.
#define FD_COMPUTE_SHADOW_WARPS     16 // 

#define FD_MAX_PIXEL_IN_BATCH		32


#define FD_MAX_LIGHT_RES_WIDTH_LOG2		5
#define FD_MAX_LIGHT_RES_HEIGHT_LOG2	6

//------------------------------------------------------------------------

#define FD_MAXBINS_SIZE         (1 << FD_MAXBINS_LOG2)
#define FD_MAXBINS_SQR          (1 << (FD_MAXBINS_LOG2 * 2))
#define FD_BIN_SIZE             (1 << FD_BIN_LOG2)
#define FD_BIN_SQR              (1 << (FD_BIN_LOG2 * 2))

#define FD_TILE_SEG_SIZE        (1 << FD_TILE_SEG_LOG2)

#define FD_TRIANGLE_QUEUE_SIZE    (1 << FD_TRIANGLE_QUEUE_LOG2)

#define FD_MAX_LIGHT_RES_WIDTH	(1<<FD_MAX_LIGHT_RES_WIDTH_LOG2)
#define FD_MAX_LIGHT_RES_HEIGHT (1<< FD_MAX_LIGHT_RES_HEIGHT_LOG2)

#define FD_LIGHT_BIT_MASK_TYPE_U32	FD_MAX_LIGHT_RES_WIDTH <= 32
#define FD_LIGHT_BIT_MASK_TYPE_U64  FD_MAX_LIGHT_RES_WIDTH > 32

//------------------------------------------------------------------------




//------------------------------------------------------------------------

#endif //__CONSTANTS_HPP__