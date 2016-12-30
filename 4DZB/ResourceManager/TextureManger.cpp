#include "TextureManger.h"
#include <cstdio>

#include <opencv2/opencv.hpp>

// 释放纹理信息
TextureManager::~TextureManager()
{
	reset();
}

void TextureManager::reset()
{
	// 删除OpengGL中的纹理
	for (auto i : m_nameToTexID)
		glDeleteTextures(1, &i.second);

	m_nameToTexID.clear();

	// 纹理存储路径置为空
	m_baseDir = string();
}

GLuint TextureManager::getTexID(const char * texPath)
{

	if (m_nameToTexID.count(texPath))
		return m_nameToTexID[texPath];

	// load image;

	if (texPath[0] == '\0')
		return -1;
	string fullPath = m_baseDir + "//" + texPath;
	IplImage *Iface = cvLoadImage(fullPath.c_str());
	if (!Iface)
	{
		fprintf(stderr, "%s %d: opencv load file error: %s\n", __FILE__, __LINE__, fullPath.c_str());
		return -1;
	}

	// 创建OpengGL 纹理

	GLuint texid;
	glGenTextures(1, &texid);

	glBindTexture(GL_TEXTURE_2D, texid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Iface->width, Iface->height, 0,
		GL_BGR, GL_UNSIGNED_BYTE, Iface->imageData);
	glBindTexture(GL_TEXTURE_2D, 0);

	m_nameToTexID[texPath] = texid;
	return texid;
}
