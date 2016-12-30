#pragma once
#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <map>
#include <string>

using namespace std;

// TextureManger 类，用来管理场景中的纹理资源
// 整个程序只能有一个TextureManger，所以使用单例模式
class TextureManager
{

private:
	TextureManager()=default;

	~TextureManager();

public:

	// 禁止拷贝构造和赋值

	TextureManager(const TextureManager& t) = delete;
	TextureManager& operator=(const TextureManager& t) = delete;

	// 获得实例

	static TextureManager& getInstance() {
		
		static TextureManager Instance;
		
		return Instance;
	}

	// 设置纹理文件夹路径
	void setBaseDirPath(const string& path) { m_baseDir = path; }

	// 重置
	void reset();

	// 获得TexID
	GLuint getTexID(const char* texPath);

	
private:
	// 私有成员函数
	map<string, GLuint> m_nameToTexID;
	string m_baseDir;

};



