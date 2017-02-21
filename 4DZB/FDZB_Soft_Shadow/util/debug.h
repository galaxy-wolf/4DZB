#ifndef MY_DEBUG_H
#define MY_DEBUG_H

#include <stdio.h>

// degbug ģ���
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


// (�ڲ�ʹ��) �ж�"SECTION"ģ�鹦�ܺ��Ƿ�����"DEBUG_LEVEL"�ȼ��ĵ�����Ϣ���
#define __my_unallow_debug(SECTION, DEBUG_LEVEL) (DEBUG_LEVEL > __my_allow_debug_levels[SECTION])

// (�ڲ�ʹ��) ������Ϣ�������
#define __my_debug(FORMAT, ...) printf("#%s(%d): " FORMAT, __FUNCTION__, __LINE__, __VA_ARGS__)

// ��ʼ��"SECTION"ģ�鹦�ܺŵĵ��Եȼ�
#define my_init_debug_levels(SECTION, ALLOW_DEBUG_LEVEL) (__my_allow_debug_levels[SECTION] = ALLOW_DEBUG_LEVEL)

// ������Ϣ�������������ϢΪ"SECTION"ģ�鹦�ܺ�"DEBUG_LEVEL"�ȼ��ĵ�����Ϣ
#define my_debug(SECTION, DEBUG_LEVEL) (__my_unallow_debug(SECTION, DEBUG_LEVEL)) ? (void)0 : __my_debug
#else
#define __my_unallow_debug(SECTION, DEBUG_LEVEL)  

// (�ڲ�ʹ��) ������Ϣ�������
#define __my_debug(FORMAT, ...) 
// ��ʼ��"SECTION"ģ�鹦�ܺŵĵ��Եȼ�
#define my_init_debug_levels(SECTION, ALLOW_DEBUG_LEVEL) 
// ������Ϣ�������������ϢΪ"SECTION"ģ�鹦�ܺ�"DEBUG_LEVEL"�ȼ��ĵ�����Ϣ
#define my_debug(SECTION, DEBUG_LEVEL) 
#endif
#endif // MY_DEBUG_H
