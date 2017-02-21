#ifndef __LAYERD_SURFACE_CUH__
#define __LAYERD_SURFACE_CUH__
#include <cuda_runtime.h>



#include "channel.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <helper_functions.h>
#include <helper_cuda.h> 
#include <cuda_gl_interop.h>


/* 元素必须是float4类型的*/
class Layered1DSurfaceManager{
public:
	Layered1DSurfaceManager();
	~Layered1DSurfaceManager();
	int createSurface(int width, int layers);
	cudaSurfaceObject_t getSurface(){ return m_surf; };
	int deleteSurface();
private:
	int m_width, m_layers;
	cudaArray_t m_2Darray;
	cudaSurfaceObject_t m_surf;
	bool m_binded;
};

class Layered2DSurfaceManager{
public:
	Layered2DSurfaceManager();
	~Layered2DSurfaceManager();
	int createSurface(int width, int height, int layers);
	cudaSurfaceObject_t getSurface(){ return m_surf; };
	int deleteSurface();
private:
	int m_width, m_height, m_layers;
	cudaArray_t m_3Darray;
	cudaSurfaceObject_t m_surf;
	bool m_binded;

};

#endif // __LAYERD_SURFACE_CUH__