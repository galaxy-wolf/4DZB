#include "CudaBuffer.h"

namespace FD{
	template<>
	void Buffer<float>::setDataChannel()
	{
		resDesc.res.linear.desc = float_channelDesc;
	}

	template<>
	void Buffer<float2>::setDataChannel()
	{
		resDesc.res.linear.desc = float2_channelDesc;
	}

	template<>
	void Buffer<float3>::setDataChannel()
	{
		resDesc.res.linear.desc = float3_channelDesc;
	}

	template<>
	void Buffer<float4>::setDataChannel()
	{
		resDesc.res.linear.desc = float4_channelDesc;
	}

	template<>
	void Buffer<int>::setDataChannel()
	{
		resDesc.res.linear.desc = int_channelDesc;
	}

	template<>
	void Buffer<unsigned int>::setDataChannel()
	{
		resDesc.res.linear.desc = unsigned_int_channelDesc;
	}
}