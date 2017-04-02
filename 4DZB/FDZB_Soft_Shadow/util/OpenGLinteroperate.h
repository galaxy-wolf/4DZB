#ifndef __OPENGL_INTEROPERATE_CUH__
#define __OPENGL_INTEROPERATE_CUH__
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include "channel.h"
namespace FD{

	class FDModel{
	public:
		GLuint VBO;
		GLuint IBO;
		size_t vertexSizeBytes;

		cudaGraphicsResource *vboRes;
		cudaGraphicsResource *iboRes;
		float* vboPtr;
		int * iboPtr;
		size_t vboNumBytes;
		size_t iboNumBytes;
		size_t tNum;
		size_t vNum;
		size_t vertexSizeFloat;
		std::string name;
	
	private:
		// for bind texture;
		struct cudaResourceDesc resDesc;
		struct cudaTextureDesc texDesc;
		cudaTextureObject_t m_tex;
	public:
		void registerRes()
		{
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&(vboRes), VBO, cudaGraphicsMapFlagsReadOnly));
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&(iboRes), IBO, cudaGraphicsMapFlagsReadOnly));
		}
		void unregisterRes()
		{
			checkCudaErrors(cudaGraphicsUnregisterResource(vboRes));
			checkCudaErrors(cudaGraphicsUnregisterResource(iboRes));
		}
		void map()
		{
			checkCudaErrors(cudaGraphicsMapResources(1, &vboRes, 0));
			checkCudaErrors(cudaGraphicsMapResources(1, &iboRes, 0));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vboPtr, &vboNumBytes,
				vboRes));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&iboPtr, &iboNumBytes,
				iboRes));
			tNum = iboNumBytes / sizeof(int) / 3;
			vNum = vboNumBytes / vertexSizeBytes;
			vertexSizeFloat = vertexSizeBytes / sizeof(float);

			bindToTexture();
		}

		void mapVBO()
		{
			checkCudaErrors(cudaGraphicsMapResources(1, &vboRes, 0));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vboPtr, &vboNumBytes,
				vboRes));
			vNum = vboNumBytes / vertexSizeBytes;
			vertexSizeFloat = vertexSizeBytes / sizeof(float);
		}

		void unmap()
		{
			unbindTexture();

			checkCudaErrors(cudaGraphicsUnmapResources(1, &vboRes, NULL));
			checkCudaErrors(cudaGraphicsUnmapResources(1, &iboRes, NULL));
		}

		void unmapVBO()
		{
			checkCudaErrors(cudaGraphicsUnmapResources(1, &vboRes, NULL));
		}

		cudaTextureObject_t getTexture(){ return m_tex; }

		void bindToTexture()
		{
			if (vertexSizeFloat % 4)
			{
				printf("bind vbo to texture error: element size can not divid by 4\n");
				return;
			}

			// set resource desc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeLinear; // 线性内存
			resDesc.res.linear.devPtr = vboPtr;
			resDesc.res.linear.sizeInBytes = vboNumBytes;

			// set texture desc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;// 只有1维；
			texDesc.filterMode = cudaFilterModePoint; // 不要插值；
			texDesc.readMode = cudaReadModeElementType; // 读原始值；
			texDesc.normalizedCoords = false; // 使用数组下标取数据。

			resDesc.res.linear.desc = float4_channelDesc;

			checkCudaErrors(cudaCreateTextureObject(&m_tex, &resDesc, &texDesc, NULL));
		}

		void unbindTexture()
		{
			checkCudaErrors(cudaDestroyTextureObject(m_tex));
		}

	};

	

	class FDResult{
	public:
		GLuint resultPBO;
		cudaGraphicsResource_t resultRes;
		float * dev_result;
		size_t result_num_bytes;
	public:
		void registerRes()
		{
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&resultRes, resultPBO, cudaGraphicsMapFlagsWriteDiscard));
		}
		void unregisterRes()
		{
			checkCudaErrors(cudaGraphicsUnregisterResource(resultRes));
		}
		void map()
		{
			checkCudaErrors(cudaGraphicsMapResources(1, &resultRes, 0));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dev_result, &result_num_bytes,
				resultRes));
		}
		void unmap()
		{
			checkCudaErrors(cudaGraphicsUnmapResources(1, &resultRes, NULL));
		}
		void clear()
		{
			checkCudaErrors(cudaMemset(dev_result, 0, result_num_bytes));
		}
	};
}

#endif //__OPENGL_INTEROPERATE_CUH__