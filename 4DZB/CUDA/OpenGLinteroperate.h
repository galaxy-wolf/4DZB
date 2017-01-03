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
		}
		void unmap()
		{
			checkCudaErrors(cudaGraphicsUnmapResources(1, &vboRes, NULL));
			checkCudaErrors(cudaGraphicsUnmapResources(1, &iboRes, NULL));
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