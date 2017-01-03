#ifndef __CHANNEL_CUH__
#define __CHANNEL_CUH__
#include <cuda_runtime.h>
namespace FD{
	extern cudaChannelFormatDesc float_channelDesc;
	extern cudaChannelFormatDesc float2_channelDesc;
	extern cudaChannelFormatDesc float3_channelDesc;
	extern cudaChannelFormatDesc float4_channelDesc;

	extern cudaChannelFormatDesc int_channelDesc;
	extern cudaChannelFormatDesc unsigned_int_channelDesc;
}
#endif // __BIN_RASTER_CUH__