#pragma once



#include <cuda_runtime.h>
#include <host_defines.h>
#include <cuda_gl_interop.h>

#include <vector>

#include <driver_types.h>

#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions

#include <helper_cuda.h>         // helper functions for CUDA error check



#include <float.h>

// cudpp
#include <stdio.h>
#include <limits.h>



__constant__ float devConst_modelMat[16];
