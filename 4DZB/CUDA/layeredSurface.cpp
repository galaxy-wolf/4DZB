#include "layeredSurface.h"

/***********************************************************************

					1D layered texture;
				
************************************************************************/
Layered1DSurfaceManager::Layered1DSurfaceManager() :
		m_width(0), 
		m_layers(0),
		m_binded(false),
		m_surf(0)
{
}

Layered1DSurfaceManager::~Layered1DSurfaceManager()
{
	if (!m_binded)
		return;
	deleteSurface();
}

int Layered1DSurfaceManager::createSurface(int width, int layers)
{
	if (m_binded || width <=0 || layers <=0)
		return -1;
	m_width = width;
	m_layers = layers;
	//create 2D array
	checkCudaErrors(cudaMallocArray(&m_2Darray,
		&FD::float4_channelDesc,
		m_width,
		m_layers));

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	resDesc.res.array.array = m_2Darray;
	cudaCreateSurfaceObject(&m_surf, &resDesc);

	m_binded = true;
	return 0;
}

int Layered1DSurfaceManager::deleteSurface()
{
	if (!m_binded)
		return 0;
	checkCudaErrors(cudaDestroySurfaceObject(m_surf));
	checkCudaErrors(cudaFreeArray(m_2Darray));
	m_binded = false;
}



/**********************************************************************

		2D layered texture
***********************************************************************/
Layered2DSurfaceManager::Layered2DSurfaceManager():
m_width(0),
m_height(0),
m_layers(0),
m_binded(false),
m_surf(0)
{
}

Layered2DSurfaceManager::~Layered2DSurfaceManager()
{
	deleteSurface();
}

int Layered2DSurfaceManager::createSurface(int width, int height, int layers)
{
	if (width <= 0 || height <= 0 || layers <= 0 || m_binded)
		return -1;

	m_width = width;
	m_height = height;
	m_layers = layers;

	checkCudaErrors(cudaMalloc3DArray(&m_3Darray, &FD::float4_channelDesc, make_cudaExtent(m_width, m_height, m_layers), cudaArrayLayered));

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	resDesc.res.array.array = m_3Darray;
	cudaCreateSurfaceObject(&m_surf, &resDesc);

	m_binded = true;
	return 0;
}

int Layered2DSurfaceManager::deleteSurface()
{
	if (!m_binded)
		return 0;
	checkCudaErrors(cudaDestroySurfaceObject(m_surf));
	checkCudaErrors(cudaFreeArray(m_3Darray));
	m_binded = false;
	return 0;
}
