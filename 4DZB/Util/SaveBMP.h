//将数据保存为bmp图像的函数
#include <cstdio>
#include <windows.h>
#include <GL/glut.h>
#include <string>

#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
void SaveBMP1(const char* fileName, BYTE * buf, UINT width, UINT height);
void saveScreenToBMP(const std::string& filename, int screenW, int screenH, bool halfCompress);
