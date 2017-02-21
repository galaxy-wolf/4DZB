#ifndef MY_DEBUG_H
#define MY_DEBUG_H

#include <stdio.h>

// degbug 模块号
enum MY_DEBUG_SECTION_ID{

	MY_DEBUG_SECTION_MEM,
	MY_DEBUG_SECTION_CUDA_INFO,
	MY_DEBUG_SECTION_AABB,
	MY_DEBUG_SECTION_SETUP,
	MY_DEBUG_SECTION_RASTER,
	MY_DEBUG_SECTION_SHADOW_CAL,
	MY_DEBUG_SECTION_GL,
	MY_DEBUG_SECTION_NUM
};

#if _DEBUG
extern int __my_allow_debug_levels[MY_DEBUG_SECTION_NUM];


// (内部使用) 判断"SECTION"模块功能号是否允许"DEBUG_LEVEL"等级的调试信息输出
#define __my_unallow_debug(SECTION, DEBUG_LEVEL) (DEBUG_LEVEL > __my_allow_debug_levels[SECTION])

// (内部使用) 调试信息输出函数
#define __my_debug(FORMAT, ...) printf("#%s(%d): " FORMAT, __FUNCTION__, __LINE__, __VA_ARGS__)

// 初始化"SECTION"模块功能号的调试等级
#define my_init_debug_levels(SECTION, ALLOW_DEBUG_LEVEL) (__my_allow_debug_levels[SECTION] = ALLOW_DEBUG_LEVEL)

// 调试信息输出函数，该信息为"SECTION"模块功能号"DEBUG_LEVEL"等级的调试信息
#define my_debug(SECTION, DEBUG_LEVEL) (__my_unallow_debug(SECTION, DEBUG_LEVEL)) ? (void)0 : __my_debug
#else
#define __my_unallow_debug(SECTION, DEBUG_LEVEL)  

// (内部使用) 调试信息输出函数
#define __my_debug(FORMAT, ...) 
// 初始化"SECTION"模块功能号的调试等级
#define my_init_debug_levels(SECTION, ALLOW_DEBUG_LEVEL) 
// 调试信息输出函数，该信息为"SECTION"模块功能号"DEBUG_LEVEL"等级的调试信息
#define my_debug(SECTION, DEBUG_LEVEL) 
#endif
#endif // MY_DEBUG_H
