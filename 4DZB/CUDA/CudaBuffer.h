#ifndef __CUDA_BUFFER_CUH__
#define __CUDA_BUFFER_CUH__
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "channel.h"
#include "../../debug/debug.h"
namespace FD{
	template<typename T>
	class Buffer{
	public:
		void * devPtr;
		size_t size_in_element;
		
		Buffer() :devPtr(NULL), elementSize(0), size_in_byte(0), size_in_element(0), m_tex(0), m_binded(false){ elementSize = sizeof(T); };
		~Buffer(){ freeBuffer(); };
		
		void allocBuffer()
		{
			applySize();
			my_debug(MY_DEBUG_SECTION_MEM, 2)("alloc %d element %d MByte\n", size_in_element, size_in_byte/1024/1024);
			if (size_in_byte > 0) checkCudaErrors(cudaMalloc((void**)&devPtr, size_in_byte)); 
		};
		void freeBuffer() { if (m_binded) { unbindTexture(); m_binded = false; }if (devPtr != NULL) checkCudaErrors(cudaFree(devPtr));  devPtr = NULL; };
		void clear(int val) { checkCudaErrors(cudaMemset(devPtr, val, size_in_byte)); };

		cudaTextureObject_t getTexture() { return m_tex; };
		// must unbind texture 
		void bindTexture()
		{
			
			prepareBind();
			setDataChannel();
			checkCudaErrors(cudaCreateTextureObject(&m_tex, &resDesc, &texDesc, NULL));
			my_debug(MY_DEBUG_SECTION_MEM, 2)("bind to texture done!\n");
			m_binded = true;
		}
		void unbindTexture()
		{
			checkCudaErrors(cudaDestroyTextureObject(m_tex));
			m_binded = false;
		}
	private:
		void applySize()
		{
			size_in_byte = size_in_element * elementSize;
		};

		void prepareBind()
		{
			// set resource desc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeLinear; // 线性内存
			resDesc.res.linear.devPtr = devPtr;
			resDesc.res.linear.sizeInBytes = size_in_byte;

			// set texture desc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;// 只有1维；
			texDesc.filterMode = cudaFilterModePoint; // 不要插值；
			texDesc.readMode = cudaReadModeElementType; // 读原始值；
			texDesc.normalizedCoords = false; // 使用数组下标取数据。
		}

		void setDataChannel();

	private:
		size_t size_in_byte;
		size_t elementSize;

		cudaTextureObject_t m_tex;
		bool m_binded;


		struct cudaResourceDesc resDesc;
		struct cudaTextureDesc texDesc;
	};

	
}

#endif// __CUDA_BUFFER_CUH__