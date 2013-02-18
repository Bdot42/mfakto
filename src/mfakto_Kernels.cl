/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2013  Oliver Weihe (o.weihe@t-online.de)
                           Bertram Franz (bertramf@gmx.net)

mfaktc (mfakto) is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc (mfakto) is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
                                
You should have received a copy of the GNU General Public License
along with mfaktc (mfakto).  If not, see <http://www.gnu.org/licenses/>.

Version 0.13

*/
/*
 All OpenCL kernels for mfakto Trial-Factoring, version 0.10

   is 2^p-1 divisible by q (q=2kp+1)? 
	                        Remove   Optional   
            Square        top bit  mul by 2       mod 47
            ------------  -------  -------------  ------
            1*1 = 1       1  0111  1*2 = 2           2
            2*2 = 4       0   111     no             4
            4*4 = 16      1    11  16*2 = 32        32
            32*32 = 1024  1     1  1024*2 = 2048    27
            27*27 = 729   1        729*2 = 1458      1

    Thus, 2^23 = 1 mod 47. Subtract 1 from both sides. 2^23-1 = 0 mod 47.
    Since we've shown that 47 is a factor, 2^23-1 is not prime.
 */

// Starting with Catalyst 11.10, not all parameters were passed to the kernel
// -> replace user-defined struct with uint8
#define WA_FOR_CATALYST11_10_BUG

// TRACE_KERNEL: higher is more trace, 0-5 currently used
#define TRACE_KERNEL 0

// If above tracing is on, only the thread with the ID below will trace
#define TRACE_TID 0

/***********************************
 * DONT CHANGE ANYTHING BELOW THIS *
 ***********************************/

#pragma  OPENCL EXTENSION cl_amd_printf : enable
//#pragma  OPENCL EXTENSION cl_khr_fp64 : enable

// HD4xxx does not have atomics, but mfakto will work on these GPUs as well.
// Without atomics, the factors found may be scrambled when more than one
// factor is found per grid => if the reported factor(s) are not accepted
// by primenet, then run the bitlevel again with the smallest possible grid size,
// or run it on at least HD5...

#ifdef cl_khr_global_int32_base_atomics 
#pragma  OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMIC_INC(x) atomic_inc(&x)
#else
// No atomic operations available - using simple ++
#define ATOMIC_INC(x) ((x)++)
#endif

#ifdef cl_amd_media_ops
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#else
// we should define something for bitalign() ...
//     Build-in Function
//      uintn  amd_bitalign (uintn src0, uintn src1, uintn src2)
//    Description
//      dst.s0 =  (uint) (((((long)src0.s0) << 32) | (long)src1.s0) >> (src2.s0 & 31))
//      similar operation applied to other components of the vectors.
#define amd_bitalign(src0, src1, src2) (src0  << (32-src2)) | (src1 >> src2)
#endif

#ifdef cl_amd_media_ops2
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#else
// we need to define what we need:
//     Build-in Function
//      uintn amd_max3 (uintn src0, uintn src1, uintn src2)
#define amd_max3(src0, src1, src2)  max(src0, max(src1, src2))
#endif

#ifdef CHECKS_MODBASECASE
// this check only works for single vector (i.e. no vector)
#if (VECTOR_SIZE != 1)
# error "CHECKS_MODBASECASE only works with VECTOR_SIZE = 1"
#endif
// to make tf_debug.h happy:
#define USE_DEVICE_PRINTF
#define __CUDA_ARCH__ 200

#include "tf_debug.h"

#else

#define MODBASECASE_QI_ERROR(A, B, C, D)
#define MODBASECASE_NONZERO_ERROR(A, B, C, D)
#define MODBASECASE_NN_BIG_ERROR(A, B, C, D)

#endif

#include "datatypes.h"

#ifdef CL_GPU_SIEVE
#include "gpusieve.cl"
#endif

#define EVAL_RES_b(comp) \
  if((a.d2.comp|a.d1.comp)==0 && a.d0.comp==1) \
  { \
      tid=ATOMIC_INC(RES[0]); \
      if(tid<10) \
      { \
        RES[tid*3 + 1]=f.d2.comp; \
        RES[tid*3 + 2]=f.d1.comp; \
        RES[tid*3 + 3]=f.d0.comp; \
      } \
  }

#define EVAL_RES_a(comp) \
  if((a.d2.comp|a.d1.comp)==0 && a.d0.comp==1) \
  { \
    if ((f.d2.comp|f.d1.comp)!=0 || f.d0.comp != 1) \
    { \
      tid=ATOMIC_INC(RES[0]); \
      if(tid<10) \
      { \
        RES[tid*3 + 1]=f.d2.comp; \
        RES[tid*3 + 2]=f.d1.comp; \
        RES[tid*3 + 3]=f.d0.comp; \
      } \
    } \
  }

#define EVAL_RES_l(comp) \
  if(a.comp == 1) \
  { \
      tid=ATOMIC_INC(RES[0]); \
      if(tid<10) \
      { \
        RES[tid*3 + 1]=0; \
        RES[tid*3 + 2]=convert_uint(f.comp >> 32); \
        RES[tid*3 + 3]=convert_uint(f.comp); \
      } \
  }

#include "barrett15.cl"  // mul24-based barrett 73-bit kernel using a word size of 15 bit
#include "barrett.cl"   // one kernel file for 32-bit-barrett of different vector sizes (1, 2, 4, 8, 16)
#define EVAL_RES(x) EVAL_RES_b(x)  // no check for f==1 if running the "big" version
#include "mul24.cl" // one kernel file for 24-bit-kernels of different vector sizes (1, 2, 4, 8, 16)
#include "barrett24.cl"  // mul24-based barrett 72-bit kernel (all vector sizes)
#include "montgomery.cl"  // montgomery kernels

#define _63BIT_MUL24_K
#undef EVAL_RES
#define EVAL_RES(x) EVAL_RES_a(x)
#include "mul24.cl" // include again, now for small factors < 64 bit


// this kernel is only used for a quick test at startup - no need to be correct ;-)
// currently this kernel is used for testing what happens without atomics when multiple factors are found
__kernel void test_k(const ulong hi, const ulong lo, const ulong q,
                           const float qr, __global uint *res
)
{
  __private uint i,f, tid;
  int180_v resv = {{0,0,0,0}};
  int90_v a, as, b, r, m;
  tid = get_global_id(0);
  float_v ff = 1.0f;


//  barrier(CLK_GLOBAL_MEM_FENCE);
#if (TRACE_KERNEL > 0)
    printf("kernel tracing level %d enabled\n", TRACE_KERNEL);
#endif
  a.d0=1;
  a.d1=0;
  a.d2=0;
  a.d3=0x0003;
  a.d4=0x7000;
  a.d5=0x0010;
  /*
  b=invmod2pow90(a);
  r=neginvmod2pow90(a);

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("test: a=%x:%x:%x:%x:%x:%x, invmod2pow90(a)=%x:%x:%x:%x:%x:%x(b)  neg= %x:%x:%x:%x:%x:%x(r)\n",
        a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        r.d5.s0, r.d4.s0, r.d3.s0, r.d2.s0, r.d1.s0, r.d0.s0);
#endif

  mul_90(&b, a, b);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("test: a*b= %x:%x:%x:%x:%x:%x\n",
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0);
#endif
  ff= CONVERT_FLOAT_RTP_V(mad24(a.d5, 32768u, a.d4));
  ff= ff * 1073741824.0f+ CONVERT_FLOAT_RTP_V(mad24(a.d3, 32768u, a.d2));

  ff = as_float(0x3f7ffffd) / ff;
  b = neg_90(a);
#ifndef CHECKS_MODBASECASE
  mod_simple_90(&as, b, a, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  mod_simple_90(&as, b, a, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max, 1 << (90-bit_max), modbasecase_debug);
#endif

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("test: as=%x:%x:%x:%x:%x:%x, -a=%x:%x:%x:%x:%x:%x\n",
        as.d5.s0, as.d4.s0, as.d3.s0, as.d2.s0, as.d1.s0, as.d0.s0,
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0);
#endif

  m = mod_REDC90(as, a, r);

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("test: mod_REDC90(as, a, r)=%x:%x:%x:%x:%x:%x\n",
        m.d5.s0, m.d4.s0, m.d3.s0, m.d2.s0, m.d1.s0, m.d0.s0);
#endif

  m = squaremod_REDC90(as, a, r);

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("test: mulmod_REDC90(as, as, a, r)=%x:%x:%x:%x:%x:%x\n",
        m.d5.s0, m.d4.s0, m.d3.s0, m.d2.s0, m.d1.s0, m.d0.s0);
#endif

  b.d0=11;
  b.d1=0;
  b.d2=0x0;
  b.d3=0x0;
  b.d4=0x0;
  b.d5=0x3FFF;

  mul_90_180_no_low3(&resv, a, b);
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("test: a=%x:%x:%x:%x:%x:%x * b=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x:%x:%x:%x:...\n",
        a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        resv.db.s0, resv.da.s0, resv.d9.s0, resv.d8.s0, resv.d7.s0, resv.d6.s0, resv.d5.s0, resv.d4.s0, resv.d3.s0);
#endif
  inc_if_ge_90(&a, a, b);
  mul_90_180_no_low3(&resv, a, b);
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("test: a=%x:%x:%x:%x:%x:%x * b=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x:%x:%x:%x:...\n",
        a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        resv.db.s0, resv.da.s0, resv.d9.s0, resv.d8.s0, resv.d7.s0, resv.d6.s0, resv.d5.s0, resv.d4.s0, resv.d3.s0);
#endif

  ff= CONVERT_FLOAT_RTP_V(mad24(a.d5, 32768u, a.d4));
  ff= ff * 32768.0f * 32768.0f+ CONVERT_FLOAT_RTP_V(mad24(a.d3, 32768u, a.d2));   // f.d1 needed?

  //ff= as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we always underestimate the quotient
  ff= as_float(0x3f7ffffd) / ff;		// just a little bit below 1.0f so we always underestimate the quotient
  */
#if (BARRETT_VECTOR_SIZE > 1)
  i=mad24(resv.db.s0, 32768u, resv.da.s0)>>2;
  f=(mad24(resv.db.s0, 32768u, resv.da.s0)<<30) + mad24(resv.d9.s0, 32768u, resv.d8.s0);
  div_180_90(&r, i, a, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  );
  // enforce evaluation ... otherwise some calculations are optimized away ;-)
  res[30] = r.d5.s0 + r.d4.s0 + r.d3.s0 + r.d2.s0 + r.d1.s0 + r.d0.s0;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("test: %x:%x:120x0 / a=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x\n",
        i, f, 
        a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        r.d5.s0, r.d4.s0, r.d3.s0, r.d2.s0, r.d1.s0, r.d0.s0);
#endif

  ff= CONVERT_FLOAT_RTP_V(mad24(b.d5, 32768u, b.d4));
  ff= ff * 32768.0f * 32768.0f+ CONVERT_FLOAT_RTP_V(mad24(b.d3, 32768u, b.d2));   // f.d1 needed?

  ff= as_float(0x3f7ffffd) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.
        
  i=mad24(resv.db.s0, 32768u, resv.da.s0)>>2;
  f=(mad24(resv.db.s0, 32768u, resv.da.s0)<<30) + mad24(resv.d9.s0, 32768u, resv.d8.s0);

  div_180_90(&r, i, b, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  );
  // enforce evaluation ... otherwise some calculations are optimized away ;-)
  res[31] = r.d5.s0 + r.d4.s0 + r.d3.s0 + r.d2.s0 + r.d1.s0 + r.d0.s0;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("test: %x:%x:120x0 / b=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x\n",
        i, f, 
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        r.d5.s0, r.d4.s0, r.d3.s0, r.d2.s0, r.d1.s0, r.d0.s0);
#endif


  f=tid+1; // let the reported results start with 1
  if (1 == 1)
  {
    i=ATOMIC_INC(res[0]);
    printf("thread %d: i=%d, res[0]=%d\n", get_global_id(0), i, res[0]);

    if(i<10)				/* limit to 10 results */
    {
      res[i*3 + 1]=f;
      res[i*3 + 2]=f;
      res[i*3 + 3]=f;
    }
  }
}
