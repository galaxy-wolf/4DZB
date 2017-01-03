#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "FourD.h"

namespace FD{
	// init
	void FDinit()
	{
		
		cudaGetDevice(&m_deviceID);
		cudaDeviceGetAttribute(&m_maxGridDimX, cudaDevAttrMaxGridDimX, m_deviceID);
		cudaDeviceGetAttribute(&m_maxBlockDimX, cudaDevAttrMaxBlockDimX, m_deviceID);
		int SM_num, shareMemPerBlock, shareMemPerSM;
		cudaDeviceGetAttribute(&SM_num, cudaDevAttrMultiProcessorCount, m_deviceID);
		cudaDeviceGetAttribute(&shareMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, m_deviceID);
		cudaDeviceGetAttribute(&shareMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, m_deviceID);

		// for maxwell we have 96KB shared memory for per SM but 48KB for per block;
		m_CTA_num = SM_num * (shareMemPerSM / shareMemPerBlock);

		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 2)("max grid dim x num is %d\n", m_maxGridDimX);
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 2)("max block dim x num is %d\n", m_maxBlockDimX);
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 2)("SM num is %d\n", SM_num);
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 2)("share memory per block is %dKB\n", shareMemPerBlock / 1024);
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 2)("share memory per SM is %dKB\n", shareMemPerSM / 1024);
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 1)("CTA num is %d\n", m_CTA_num);

		
		// aabb temp buffer 
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc aabb temp buffer start\n");
		AABB3DReduceBuffer.freeBuffer();
		AABB3DReduceBuffer.size_in_element = (m_maxGridDimX / m_maxBlockDimX )* 8;// 3 for min x, y, z; 3 for max x, y, z; 2 for avg x, y;
		AABB3DReduceBuffer.allocBuffer();
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc aabb temp buffer done\n");


		// bin buffer;
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binTriEndBuffer start\n");
		binTriStartBuffer.freeBuffer();
		binTriStartBuffer.size_in_element = FD_MAXBINS_SIZE * FD_MAXBINS_SIZE;
		binTriStartBuffer.allocBuffer();
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binTriEndBuffer done\n");

		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binTriEndBuffer start\n");
		binTriEndBuffer.freeBuffer();
		binTriEndBuffer.size_in_element = FD_MAXBINS_SIZE * FD_MAXBINS_SIZE;
		binTriEndBuffer.allocBuffer();
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binTriEndBuffer done\n");

		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binSampleStartBuffer start\n");
		binSampleStartBuffer.freeBuffer();
		binSampleStartBuffer.size_in_element = FD_MAXBINS_SIZE * FD_MAXBINS_SIZE +1; // +1 for INvalid Bin;
		binSampleStartBuffer.allocBuffer();
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binSampleStartBuffer done\n");

		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binSampleEndBuffer start\n");
		binSampleEndBuffer.freeBuffer();
		binSampleEndBuffer.size_in_element = FD_MAXBINS_SIZE * FD_MAXBINS_SIZE + 1; // +1 for INVALID bin;
		binSampleEndBuffer.allocBuffer();
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binSampleEndBuffer done\n");

		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binSampleMaxZBuffer start\n");
		binSampleMaxZBuffer.freeBuffer();
		binSampleMaxZBuffer.size_in_element = FD_MAXBINS_SIZE * FD_MAXBINS_SIZE;
		binSampleMaxZBuffer.allocBuffer();
		binSampleMaxZBuffer.bindTexture();
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("alloc binSampleMaxZBuffer done\n");

		isBinValidBuffer.freeBuffer();
		isBinValidBuffer.size_in_element = FD_MAXBINS_SIZE * FD_MAXBINS_SIZE;
		isBinValidBuffer.allocBuffer();

		isBinValidPrefixSumBuffer.freeBuffer();
		isBinValidPrefixSumBuffer.size_in_element = FD_MAXBINS_SIZE * FD_MAXBINS_SIZE;
		isBinValidPrefixSumBuffer.allocBuffer();

		validBinBuffer.freeBuffer();
		validBinBuffer.size_in_element = FD_MAXBINS_SIZE * FD_MAXBINS_SIZE;
		validBinBuffer.allocBuffer();

		getG_Atomic_addr();

	}
	// 开启反面剔除;
	void FDsetBackfaceCull(bool enable)
	{
		m_enable_backfacecull = enable;
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 1)("back face cull is %s\n", m_enable_backfacecull ? "enable" : "disable");
	}
	// 添加模型，
	void FDaddModel(GLuint modelVBO, GLuint modelIBO, size_t vertexSizeBytes, std::string modelName)
	{
		FDModel newModel;
		newModel.VBO = modelVBO;
		newModel.IBO = modelIBO;
		newModel.name = modelName;
		newModel.vertexSizeBytes = vertexSizeBytes;
		FDscene.push_back(newModel);
	}
	// 设置采样点纹理
	void FDsetSampleGLTex(GLuint samplePositionTex, GLuint sampleNormalTex)
	{
		FDSample::GLsamplePositionTex = samplePositionTex;
		FDSample::GLsampleNormalTex = sampleNormalTex;
	}
	// 设置输出buffer
	void FDsetOutputGLBuffer(GLuint resultPBO)
	{
		FDresult.resultPBO = resultPBO;
	}
	// 当纹理大小发生改变时，需要重新注册
	void FDregisterGLres()
	{
		FDSample::registerGLtexture();
		FDresult.registerRes();
		for (int i = 0; i < FDscene.size(); i++)
			FDscene[i].registerRes();

		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 1)("register gl resource done!\n");
	}
	//程序结束解除注册
	void FDunregisterGLres()
	{
		FDSample::unregisterGLtexture();
		FDresult.unregisterRes();
		for (int i = 0; i < FDscene.size(); i++)
			FDscene[i].unregisterRes();
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 1)("unregister gl resource done!\n");
	}
	// 设置bin参数
	int FDsetBinNum(size_t binNum)
	{
		if (binNum > FD_MAXBINS_SIZE)
			return -1;
		m_binNum = binNum;

		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 1) ("bin num is %d\n", m_binNum);
		return 0;
	}
	
	// 设置视口大小
	void FDsetView(int viewWidth, int viewHeight)
	{
		m_viewWidth = viewWidth;
		m_viewHeight = viewHeight;
		m_viewWidth_LOG2UP = ((int)ceilf(log2f(m_viewWidth)));
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 1)("view width %d ,viewWidth_LOG2UP %d, height %d\n", m_viewWidth, m_viewWidth_LOG2UP, m_viewHeight);

		binSamplePairBinBuffer.freeBuffer();
		binSamplePairBinBuffer.size_in_element = viewWidth *viewHeight;
		binSamplePairBinBuffer.allocBuffer();

		binSamplePairSampleBuffer.freeBuffer();
		binSamplePairSampleBuffer.size_in_element = viewWidth * viewHeight;
		binSamplePairSampleBuffer.allocBuffer();

		sampleRectangleSurfManager.deleteSurface();
		sampleRectangleSurfManager.createSurface(viewWidth, viewHeight, 2);
		my_debug(MY_DEBUG_SECTION_CUDA_INFO, 3)("create sample rectangle surface done!\n");

		//valid buffer
		if (m_viewWidth *m_viewHeight > validBuffer.size_in_element)
		{
			validBuffer.freeBuffer();
			validBuffer.size_in_element = m_viewWidth *m_viewHeight;
			validBuffer.allocBuffer();
		}
	}
	// 设置场景中三角形数量信息
	void FDsetSceneParams(size_t maxTriangleNum, size_t maxVerticesNum)
	{
		m_maxTriangleNum = maxTriangleNum;
		m_maxVerticesNum = maxVerticesNum;

		// verticesRectangleBuffer
		verticesRectangleBuffer.freeBuffer();
		verticesRectangleBuffer.size_in_element = maxVerticesNum *2;
		verticesRectangleBuffer.allocBuffer();
		verticesRectangleBuffer.bindTexture();

		// triangleVertexBuffer
		triVertexBuffer.freeBuffer();
		triVertexBuffer.size_in_element = maxTriangleNum * 3;
		triVertexBuffer.allocBuffer();
		triVertexBuffer.bindTexture();
		

		//triAABBBuffer
		triAABBBuffer.freeBuffer();
		triAABBBuffer.size_in_element = maxTriangleNum * 2;
		triAABBBuffer.allocBuffer();
		triAABBBuffer.bindTexture();

		// triBinRangeBuffer
		triBinRangeBuffer.freeBuffer();
		triBinRangeBuffer.size_in_element = maxTriangleNum ;
		triBinRangeBuffer.allocBuffer();

		// triPairNumBuffer
		triPairNumBuffer.freeBuffer();
		triPairNumBuffer.size_in_element = maxTriangleNum;
		triPairNumBuffer.allocBuffer();

		// triPairNumPrefixSumBuffer
		triPairNumPrefixSumBuffer.freeBuffer();
		triPairNumPrefixSumBuffer.size_in_element = maxTriangleNum;
		triPairNumPrefixSumBuffer.allocBuffer();

		// valid buffer
		if (m_maxTriangleNum > validBuffer.size_in_element)
		{
			validBuffer.freeBuffer();
			validBuffer.size_in_element = m_maxTriangleNum;
			validBuffer.allocBuffer();
		}
	}
	void FDsetLightRes(int lightResWidth, int lightResHeight)
	{
		m_lightResWidth = lightResWidth;
		m_lightResHeight = lightResHeight;
		m_lightResNum = lightResHeight * lightResWidth;
	}
	// 设置灯参数
	void FDsetAreaLight(AreaLightDes lightDes)
	{
		AreaLight &curLight = h_fdStaticParams.light;
		curLight.position = make_float3(lightDes.position.x, lightDes.position.y, lightDes.position.z);
		curLight.viewDir = make_float3(lightDes.viewDir.x, lightDes.viewDir.y, lightDes.viewDir.z);
		curLight.upDir = make_float3(lightDes.upDir.x, lightDes.upDir.y, lightDes.upDir.z);
		curLight.rightRadius = lightDes.rightRadius;
		curLight.topRadius = lightDes.topRadius;

		curLight.upDir = normalize(curLight.upDir);
		curLight.viewDir = normalize(curLight.viewDir);
		curLight.rightDir = cross(curLight.viewDir, curLight.upDir);
		curLight.rightDir = normalize(curLight.rightDir);
		curLight.topDir = cross(curLight.rightDir, curLight.viewDir);
		curLight.topDir = normalize(curLight.topDir);
		
		curLight.x_delt = curLight.rightDir * (curLight.rightRadius * -2.0f / m_lightResWidth);
		curLight.y_delt = curLight.topDir * (curLight.topRadius * -2.0f / m_lightResHeight);
		

		float xfactor, yfactor;
		xfactor = (m_lightResWidth - 1.0f) / m_lightResWidth;
		yfactor = (m_lightResHeight - 1.0f) / m_lightResHeight;

		curLight.upRightCornerPosition = curLight.position + curLight.rightDir * xfactor * curLight.rightRadius + curLight.topDir * yfactor * curLight.topRadius;
		curLight.upLeftCornerPosition = curLight.position + (-1.0f*curLight.rightDir)* xfactor * curLight.rightRadius + (curLight.topDir) * yfactor * curLight.topRadius;
		curLight.downLeftCornerPosition = curLight.position + (-1.0f*curLight.rightDir)* xfactor * curLight.rightRadius + (-1.0f* curLight.topDir) * yfactor * curLight.topRadius;
		curLight.downRightCornerPosition = curLight.position + (curLight.rightDir)* xfactor * curLight.rightRadius + (-1.0f* curLight.topDir) * yfactor * curLight.topRadius;


		for (int i = 0; i < m_lightResHeight; i++)
		{
			for (int j = 0; j < m_lightResWidth; j++)
			{
				curLight.lightPos[i * m_lightResWidth + j] = curLight.upRightCornerPosition + curLight.x_delt *j + curLight.y_delt *i;
			}
		}


		glm::vec3 position = glm::vec3(curLight.position.x, curLight.position.y, curLight.position.z);
		glm::vec3 viewDir = glm::vec3(curLight.viewDir.x, curLight.viewDir.y, curLight.viewDir.z);
		glm::vec3 center = position + 10.0f * viewDir;
		glm::vec3 upDir = glm::vec3(curLight.upDir.x, curLight.upDir.y, curLight.upDir.z);
		glm::mat4 centerMV = glm::lookAt(position, center, upDir);
		glm::vec4 cb, ce;
		glm::vec4 cwe = glm::vec4(curLight.upRightCornerPosition.x, curLight.upRightCornerPosition.y, curLight.upRightCornerPosition.z, 1.0);
		glm::vec4 cwb = glm::vec4(curLight.downLeftCornerPosition.x, curLight.downLeftCornerPosition.y, curLight.downLeftCornerPosition.z, 1.0);
		ce = centerMV * cwe;
		cb = centerMV * cwb;
		curLight.cb = make_float2(cb.x, cb.y);
		curLight.ce = make_float2(ce.x, ce.y);


		glm::vec3 upRightPos = glm::vec3(curLight.upRightCornerPosition.x, curLight.upRightCornerPosition.y, curLight.upRightCornerPosition.z);
		center = upRightPos + 10.0f * viewDir;
		glm::mat4 lightMV = glm::lookAt(upRightPos, center, upDir);
		glm::mat4 P = glm::infinitePerspective(120.0f, 1.0f, 0.1f);
		glm::mat4 upRightMVP = P * lightMV;
		float * upRightMVPptr = &upRightMVP[0][0];
		for (int i = 0; i < 16; i++)
			curLight.upRightMat[i] = upRightMVPptr[i];

		// rectangle factor
		curLight.RectangleFactor.x = curLight.rightRadius * 2 * xfactor * P[0][0] * 0.5f;
		curLight.RectangleFactor.y = curLight.topRadius * 2 * yfactor * P[1][1] * 0.5f;
	}
	// 开始阴影计算
	void FDLaunch()
	{
		/////// map gl res;
		{
			FDSample::mapGLtexture();
			FDresult.map();
			for (int i = 0; i < FDscene.size(); i++)
				FDscene[i].map();
		}
			/////// set static param;
		{
			h_fdStaticParams.enable_backfaceCull = m_enable_backfacecull;

			// light is set before;

			// textureObj surfaceObj;
			h_fdStaticParams.verticesRectangleTex	= verticesRectangleBuffer.getTexture();
			h_fdStaticParams.triangleVertexTex = triVertexBuffer.getTexture();
			h_fdStaticParams.triangleAABBTex = triAABBBuffer.getTexture();
			h_fdStaticParams.binSampleMaxZTex = binSampleMaxZBuffer.getTexture();
			h_fdStaticParams.sampleRectangleSurf = sampleRectangleSurfManager.getSurface();

			

			h_fdStaticParams.numTris = m_triangleNum = FDscene[0].tNum;
			h_fdStaticParams.numVertices = m_verteicesNum = FDscene[0].vNum;
			h_fdStaticParams.vertexBuffer = (CUdeviceptr)FDscene[0].vboPtr;
			h_fdStaticParams.indexBuffer = (CUdeviceptr)FDscene[0].iboPtr;
			h_fdStaticParams.vertexSizeFloat = FDscene[0].vertexSizeFloat;

			h_fdStaticParams.viewportWidth = m_viewWidth;
			h_fdStaticParams.viewportHeight = m_viewHeight;
			h_fdStaticParams.viewportWidth_LOG2_UP = m_viewWidth_LOG2UP;

			h_fdStaticParams.widthBins = m_binNum;
			h_fdStaticParams.widthBins_LOG2 = (log2(m_binNum)); // bin Num is 2^i
			h_fdStaticParams.heightBins = m_binNum;
			h_fdStaticParams.numBins = m_binNum * m_binNum;

			h_fdStaticParams.aabbTempBuffer = (CUdeviceptr)AABB3DReduceBuffer.devPtr;
			h_fdStaticParams.validTempBuffer = (CUdeviceptr)validBuffer.devPtr;

			h_fdStaticParams.verticesRectangleBuffer = (CUdeviceptr)verticesRectangleBuffer.devPtr;
			h_fdStaticParams.triPositionBuffer = (CUdeviceptr)triVertexBuffer.devPtr;
			h_fdStaticParams.triAABBBuffer = (CUdeviceptr)triAABBBuffer.devPtr;
			h_fdStaticParams.triBinRangeBuffer = (CUdeviceptr)triBinRangeBuffer.devPtr;

			h_fdStaticParams.triBinNum = (CUdeviceptr)triPairNumBuffer.devPtr;
			h_fdStaticParams.triBinNumPrefixSum = (CUdeviceptr)triPairNumPrefixSumBuffer.devPtr;
			h_fdStaticParams.binTriStart = (CUdeviceptr)binTriStartBuffer.devPtr;
			h_fdStaticParams.binTriEnd = (CUdeviceptr)binTriEndBuffer.devPtr;

			h_fdStaticParams.pb_pair_pixel = (CUdeviceptr)binSamplePairSampleBuffer.devPtr;
			h_fdStaticParams.pb_pair_bin = (CUdeviceptr)binSamplePairBinBuffer.devPtr;
			h_fdStaticParams.binPixelStart = (CUdeviceptr)binSampleStartBuffer.devPtr;
			h_fdStaticParams.binPixelEnd = (CUdeviceptr)binSampleEndBuffer.devPtr;
			h_fdStaticParams.binPixelMaxZperBin = ((CUdeviceptr)binSampleMaxZBuffer.devPtr);

			h_fdStaticParams.isBinValidBuffer = (CUdeviceptr)isBinValidBuffer.devPtr;
			h_fdStaticParams.isBinValidPrefixSumBuffer = (CUdeviceptr)isBinValidPrefixSumBuffer.devPtr;
			h_fdStaticParams.validBinBuffer = (CUdeviceptr)validBinBuffer.devPtr;


			h_fdStaticParams.lightResWidth = m_lightResWidth;
			h_fdStaticParams.lightResHeight = m_lightResHeight;
			h_fdStaticParams.lightResNum = m_lightResNum;
			h_fdStaticParams.shadowValue = (CUdeviceptr)FDresult.dev_result;

			
			setStaticParams();
		}
		// init result
		FDresult.clear();
		//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
		//1, 求出模型AABB 的包围盒 和 sample rectangle 的包围盒，从而得到light plane 的参数。
		//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

		if (!lightPlaneParamCal()) // 如果为true 表示没有影子。
		{
			//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
			//2,  setup  triangle 
			//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
			
			setupTriangleVertex();

			//************************************************************
			// 

			//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
			//3,  将sample 分配到bin 中，并求出per bin 中sample 的maxZ
			//    将triangle 分配到对应的bin 中去
			//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
			
			bindSampleToBin();
			
			countTriBinPairNum();
			///// set dynamic param
			{
				h_fdDynamicParams.bt_pair_bin = (CUdeviceptr)(binTriPairBinBuffer.devPtr);
				h_fdDynamicParams.bt_pair_tri = (CUdeviceptr)(binTriPairTriBuffer.devPtr);

				setDynamicParams();
			}
			bindTriToBin();

			//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
			//4, 对于每个bin 开辟一个block的threads， 求解bin 中sample 的阴影值。
			//\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
			
			shadowCal();
			//refCal();


			// unmap texture;
		}



		/////// unmap gl res;
		{
			FDSample::unmapGLtexture();
			FDresult.unmap();
			for (int i = 0; i < FDscene.size(); i++)
				FDscene[i].unmap();
			
		}
		
		//exit(111);
	}
}