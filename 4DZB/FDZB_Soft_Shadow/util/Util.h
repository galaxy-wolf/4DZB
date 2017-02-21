#pragma once
#ifndef __UTIL_HPP__
#define __UTIL_HPP__
#include "channel.h"
#include "cudaBuffer.h"
#include "layeredSurface.h"
#include "OpenGLinteroperate.h"
#include "debug.h"

//------------------------------------------------------------------------
#ifdef _M_X64
#   define FW_64    1
#else
#   define FW_64    0
#endif

#ifdef __CUDACC__
#   define FW_CUDA 1
#else
#   define FW_CUDA 0
#endif
//---------------------------------------------------------------------------

//------------------------------------------------------------------------
typedef unsigned char       U8;
typedef unsigned short      U16;
typedef unsigned int        U32;
typedef signed char         S8;
typedef signed short        S16;
typedef signed int          S32;
typedef float               F32;
typedef double              F64;
typedef void                (*FuncPtr)(void);

#if FW_CUDA
typedef unsigned long long  U64;
typedef signed long long    S64;
#else
typedef unsigned __int64    U64;
typedef signed __int64      S64;
#endif

#if FW_64
typedef S64                 SPTR;
typedef U64                 UPTR;
#else
typedef __w64 S32           SPTR;
typedef __w64 U32           UPTR;
#endif
//------------------------------------------------------------------------

#define FD_U32_MAX          (0xFFFFFFFFu)
#define FD_S32_MIN          (~0x7FFFFFFF)
#define FD_S32_MAX          (0x7FFFFFFF)
#define FD_U64_MAX          ((U64)(S64)-1)
#define FD_S64_MIN          ((S64)-1 << 63)
#define FD_S64_MAX          (~((S64)-1 << 63))
#define FD_F32_MIN          (1.175494351e-38f)
#define FD_F32_MAX          (3.402823466e+38f)
#define FD_F64_MIN          (2.2250738585072014e-308)
#define FD_F64_MAX          (1.7976931348623158e+308)
#define FD_PI               (3.14159265358979323846f)

//------------------------------------------------------------------------
unsigned int __inline__ iDiviUp(unsigned int a, unsigned int b)	{ return a %b ? a / b + 1 : a / b; }

//------------------------------------------------------------------------

#if FW_CUDA

#if FW_64
#   define PTX_PTR(P) "l"(P)
#else
#   define PTX_PTR(P) "r"(P)
#endif

__device__ __inline__ U32   getLo(U64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ S32   getLo(S64 a)                 { return __double2loint(__longlong_as_double(a)); }
__device__ __inline__ U32   getHi(U64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ S32   getHi(S64 a)                 { return __double2hiint(__longlong_as_double(a)); }
__device__ __inline__ U64   combineLoHi(U32 lo, U32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ S64   combineLoHi(S32 lo, S32 hi)        { return __double_as_longlong(__hiloint2double(hi, lo)); }
__device__ __inline__ U32   getLaneMaskLt(void)                  { U32 r; asm("mov.u32 %0, %lanemask_lt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskLe(void)                  { U32 r; asm("mov.u32 %0, %lanemask_le;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGt(void)                  { U32 r; asm("mov.u32 %0, %lanemask_gt;" : "=r"(r)); return r; }
__device__ __inline__ U32   getLaneMaskGe(void)                  { U32 r; asm("mov.u32 %0, %lanemask_ge;" : "=r"(r)); return r; }
__device__ __inline__ int   findLeadingOne(U32 v)                 { U32 r; asm("bfind.u32 %0, %1;" : "=r"(r) : "r"(v)); return r; }
__device__ __inline__ bool  singleLane(void)                  { return ((__ballot(true) & getLaneMaskLt()) == 0); }

__device__ __inline__ void  add_add_carry(U32& rlo, U32 alo, U32 blo, U32& rhi, U32 ahi, U32 bhi) { U64 r = combineLoHi(alo, ahi) + combineLoHi(blo, bhi); rlo = getLo(r); rhi = getHi(r); }
__device__ __inline__ S32   f32_to_s32_sat(F32 a)                 { S32 v; asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u32_sat(F32 a)                 { U32 v; asm("cvt.rni.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u32_sat_rmi(F32 a)                 { U32 v; asm("cvt.rmi.sat.u32.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ U32   f32_to_u8_sat(F32 a)                 { U32 v; asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(v) : "f"(a)); return v; }
__device__ __inline__ S64   f32_to_s64(F32 a)                 { S64 v; asm("cvt.rni.s64.f32 %0, %1;" : "=l"(v) : "f"(a)); return v; }
__device__ __inline__ S32   add_s16lo_s16lo(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16hi_s16lo(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16lo_s16hi(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   add_s16hi_s16hi(S32 a, S32 b)			{ S32 v; asm("vadd.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16lo_s16lo(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16hi_s16lo(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16lo_s16hi(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_s16hi_s16hi(S32 a, S32 b)			{ S32 v; asm("vsub.s32.s32.s32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16lo_u16lo(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16hi_u16lo(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h1, %2.h0;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16lo_u16hi(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h0, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ S32   sub_u16hi_u16hi(U32 a, U32 b)			{ S32 v; asm("vsub.s32.u32.u32 %0, %1.h1, %2.h1;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b0(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b0, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b1(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b2(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b2, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   add_b3(U32 a, U32 b)			{ U32 v; asm("vadd.u32.u32.u32 %0, %1.b3, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ U32   vmad_b0(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b0, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b1(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b2(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b2, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b3(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b3, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b0_b3(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b0, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b1_b3(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b1, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b2_b3(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b2, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   vmad_b3_b3(U32 a, U32 b, U32 c)	{ U32 v; asm("vmad.u32.u32.u32 %0, %1.b3, %2.b3, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_mask8(U32 a, U32 b)			{ U32 v; U32 z = 0; asm("vadd.u32.u32.u32 %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(z)); return v; }
__device__ __inline__ U32   sub_mask8(U32 a, U32 b)			{ U32 v; U32 z = 0; asm("vsub.u32.u32.u32 %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(z)); return v; }
__device__ __inline__ S32   max_max(S32 a, S32 b, S32 c)	{ S32 v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   min_min(S32 a, S32 b, S32 c)	{ S32 v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   max_add(S32 a, S32 b, S32 c)	{ S32 v; asm("vmax.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   min_add(S32 a, S32 b, S32 c)	{ S32 v; asm("vmin.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_add(U32 a, U32 b, U32 c)	{ U32 v; asm("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   sub_add(U32 a, U32 b, U32 c)	{ U32 v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   add_sub(U32 a, U32 b, U32 c)	{ U32 v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(c), "r"(b)); return v; }
__device__ __inline__ S32   add_clamp_0_x(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   add_clamp_b0(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat %0.b0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   add_clamp_b2(S32 a, S32 b, S32 c)	{ S32 v; asm("vadd.u32.s32.s32.sat %0.b2, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ U32   prmt(U32 a, U32 b, U32 c)   { U32 v; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   u32lo_sext(U32 a)                 { U32 v; asm("cvt.s16.u32 %0, %1;" : "=r"(v) : "r"(a)); return v; }
__device__ __inline__ U32   slct(U32 a, U32 b, S32 c)   { U32 v; asm("slct.u32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ S32   slct(S32 a, S32 b, S32 c)   { S32 v; asm("slct.s32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ F32   slct(F32 a, F32 b, S32 c)   { F32 v; asm("slct.f32.s32 %0, %1, %2, %3;" : "=f"(v) : "f"(a), "f"(b), "r"(c)); return v; }
__device__ __inline__ U32   isetge(S32 a, S32 b)          { U32 v; asm("set.ge.u32.s32 %0, %1, %2;" : "=r"(v) : "r"(a), "r"(b)); return v; }
__device__ __inline__ F64   rcp_approx(F64 a)                 { F64 v; asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(v) : "d"(a)); return v; }
__device__ __inline__ F32   fma_rm(F32 a, F32 b, F32 c)   { F32 v; asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(v) : "f"(a), "f"(b), "f"(c)); return v; }
__device__ __inline__ U32   idiv_fast(U32 a, U32 b);

__device__ __inline__ U32   cachedLoad(const U32* p)          { U32 v; asm("ld.global.ca.u32 %0, [%1];" : "=r"(v) : PTX_PTR(p)); return v; }
__device__ __inline__ uint2 cachedLoad(const uint2* p)        { uint2 v; asm("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : PTX_PTR(p)); return v; }
__device__ __inline__ uint4 cachedLoad(const uint4* p)        { uint4 v; asm("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : PTX_PTR(p)); return v; }
__device__ __inline__ void  cachedStore(U32* p, U32 v)         { asm("st.global.wb.u32 [%0], %1;" ::PTX_PTR(p), "r"(v)); }
__device__ __inline__ void  cachedStore(uint2* p, uint2 v)     { asm("st.global.wb.v2.u32 [%0], {%1, %2};" ::PTX_PTR(p), "r"(v.x), "r"(v.y)); }
__device__ __inline__ void  cachedStore(uint4* p, uint4 v)     { asm("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" ::PTX_PTR(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)); }

__device__ __inline__ U32   uncachedLoad(const U32* p)          { U32 v; asm("ld.global.cg.u32 %0, [%1];" : "=r"(v) : PTX_PTR(p)); return v; }
__device__ __inline__ uint2 uncachedLoad(const uint2* p)        { uint2 v; asm("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : PTX_PTR(p)); return v; }
__device__ __inline__ uint4 uncachedLoad(const uint4* p)        { uint4 v; asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : PTX_PTR(p)); return v; }
__device__ __inline__ void  uncachedStore(U32* p, U32 v)         { asm("st.global.cg.u32 [%0], %1;" ::PTX_PTR(p), "r"(v)); }
__device__ __inline__ void  uncachedStore(uint2* p, uint2 v)     { asm("st.global.cg.v2.u32 [%0], {%1, %2};" ::PTX_PTR(p), "r"(v.x), "r"(v.y)); }
__device__ __inline__ void  uncachedStore(uint4* p, uint4 v)     { asm("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" ::PTX_PTR(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)); }

__device__ __inline__ U32   toABGR(float4 color);
__device__ __inline__ U32   blendABGR(U32 src, U32 dst, U32 srcColorFactor, U32 dstColorFactor, U32 srcAlphaFactor, U32 dstAlphaFactor); // Uses 8 highest bits of xxxFactor.
__device__ __inline__ U32   blendABGRClamp(U32 src, U32 dst, U32 srcColorFactor, U32 dstColorFactor, U32 srcAlphaFactor, U32 dstAlphaFactor); // Clamps the result to 255.

__device__ __inline__ uint3 setupPleq(float3 values, int2 v0, int2 d1, int2 d2, F32 areaRcp, int samplesLog2);

__device__ __inline__ U64   cover8x8_exact_ref(S32 ox, S32 oy, S32 dx, S32 dy); // reference implementation
__device__ __inline__ U64   cover8x8_conservative_ref(S32 ox, S32 oy, S32 dx, S32 dy);
__device__ __inline__ U64   cover8x8_generateMask_ref(S64 curr, S64 stepX, S64 stepY);
__device__ __inline__ bool  cover8x8_missesTile(S32 ox, S32 oy, S32 dx, S32 dy);

__device__ __inline__ void  cover8x8_setupLUT(volatile U64* lut);
__device__ __inline__ U64   cover8x8_exact_fast(S32 ox, S32 oy, S32 dx, S32 dy, U32 flips, volatile const U64* lut); // Assumes viewport <= 2^11, subpixels <= 2^4, no guardband.
__device__ __inline__ U64   cover8x8_conservative_fast(S32 ox, S32 oy, S32 dx, S32 dy, U32 flips, volatile const U64* lut);
__device__ __inline__ U64   cover8x8_lookupMask(S64 yinit, U32 yinc, U32 flips, volatile const U64* lut);

__device__ __inline__ U64   cover8x8_exact_noLUT(S32 ox, S32 oy, S32 dx, S32 dy); // optimized reference implementation, does not require look-up table
__device__ __inline__ U64   cover8x8_conservative_noLUT(S32 ox, S32 oy, S32 dx, S32 dy);
__device__ __inline__ U64   cover8x8_generateMask_noLUT(S32 curr, S32 dx, S32 dy);

__device__ __inline__ U32   coverMSAA_ref(int samplesLog2, S32 ox, S32 oy, S32 dx, S32 dy);
__device__ __inline__ U32   coverMSAA_fast(int samplesLog2, S32 ox, S32 oy, S32 dx, S32 dy);


#endif

//------------------------------------------------------------------------
#endif // __UTIL_HPP__