#pragma once
#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <map>
#include <string>

using namespace std;

// TextureManger �࣬�����������е�������Դ
// ��������ֻ����һ��TextureManger������ʹ�õ���ģʽ
class TextureManager
{

private:
	TextureManager()=default;

	~TextureManager();

public:

	// ��ֹ��������͸�ֵ

	TextureManager(const TextureManager& t) = delete;
	TextureManager& operator=(const TextureManager& t) = delete;

	// ���ʵ��

	static TextureManager& getInstance() {
		
		static TextureManager Instance;
		
		return Instance;
	}

	// ���������ļ���·��
	void setBaseDirPath(const string& path) { m_baseDir = path; }

	// ����
	void reset();

	// ���TexID
	GLuint getTexID(const char* texPath);

	
private:
	// ˽�г�Ա����
	map<string, GLuint> m_nameToTexID;
	string m_baseDir;

};



