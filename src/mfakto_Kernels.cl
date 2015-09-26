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

Version 0.15

*/
/*
 All OpenCL kernels for mfakto Trial-Factoring

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

// Starting with Catalyst 11.10, not all parameters are passed to the kernel
// -> replace user-defined struct with uint8
// still needed as of cat 13.1
#define WA_FOR_CATALYST11_10_BUG

// TRACE_KERNEL: higher is more trace, 0-5 currently used
#define TRACE_KERNEL 0

// If above tracing is on, only the thread with the ID below will trace
#define TRACE_TID 0

/***********************************
 * DONT CHANGE ANYTHING BELOW THIS *
 ***********************************/
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
#define MODBASECASE_PAR_DEF
#define MODBASECASE_PAR
#endif

//#if VECTOR_SIZE == 1 && TRACE_KERNEL > 2
//# error "Kernel tracing > 2 works only for VectorSize > 1"
//#endif

#include "datatypes.h"
#include "common.cl"

// for the GPU sieve, we don't implement some kernels
#ifdef CL_GPU_SIEVE
  #include "gpusieve.cl"
  #include "barrett15.cl"  // mul24-based barrett kernels using a word size of 15 bit
  #include "barrett.cl"   // one kernel file for 32-bit-barrett of different vector sizes (1, 2, 4, 8, 16)
#else
  #define EVAL_RES(x) EVAL_RES_b(x)  // no check for f==1 if running the "big" version

  #include "barrett15.cl"  // mul24-based barrett kernels using a word size of 15 bit
  #include "barrett.cl"   // one kernel file for 32-bit-barrett of different vector sizes (1, 2, 4, 8, 16)

  #include "mul24.cl" // one kernel file for 24-bit-kernels of different vector sizes (1, 2, 4, 8, 16)
  #include "montgomery.cl"  // montgomery kernels

  #define _63BIT_MUL24_K
  #include "mul24.cl" // include again, now for small factors < 64 bit
#endif

// this kernel is only used for a quick test at startup - no need to be correct ;-)
// currently this kernel is used for testing what happens without atomics when multiple factors are found
__kernel void test_k(const ulong hi, const ulong lo, const ulong q,
                           const float qr, __global uint *res
)
{
  __private uint i,f, tid;
  int180_v resv;
  int90_v a, b, r;
  tid = get_global_id(0);
#if defined USE_DP
  double_v ff = 1.0;
#else
  float_v ff = 1.0f;
#endif
  uint_v carry0, carry1;


//  barrier(CLK_GLOBAL_MEM_FENCE);
#if (TRACE_KERNEL > 0)
    printf((__constant char *)"kernel tracing level %d enabled\n", TRACE_KERNEL);
#endif
  a.d0=1;
  a.d1=hi >> 11;
  a.d2=0;
  a.d3=0x0003;
  a.d4=0x7000;
  a.d5=0x0010;

  square_90_180(&resv, a);

  b.d0=1;
  b.d1=lo >> 11;

  carry0 = (hi < lo);
  carry1 = AS_UINT_V((a.d1 > b.d1) || (carry0 && AS_UINT_V(a.d1 == b.d1)));
#if (TRACE_KERNEL > 0)
  printf((__constant char *)"a=%d b=%d carry=%x => carry=%x\n", a.d1, b.d1, carry0, carry1);
#endif

  carry0 = (hi > lo);
  carry1 = AS_UINT_V((a.d1 > b.d1) || (carry0 && AS_UINT_V(a.d1 == b.d1)));
#if (TRACE_KERNEL > 0)
  printf((__constant char *)"a=%d b=%d carry=%x => carry=%x\n", a.d1, b.d1, carry0, carry1);
#endif

  b.d1=hi;
  a.d1=lo;
  carry1 = AS_UINT_V((a.d1 > b.d1) || (carry0 && AS_UINT_V(a.d1 == b.d1)));
#if (TRACE_KERNEL > 0)
  printf((__constant char *)"a=%d b=%d carry=%x => carry=%x\n", a.d1, b.d1, carry0, carry1);
#endif

  a.d1=hi;
  b.d1=lo;
  carry0 = (hi > q);
  carry1 = AS_UINT_V((a.d1 > b.d1) || (carry0 && AS_UINT_V(a.d1 == b.d1)));
#if (TRACE_KERNEL > 0)
  printf((__constant char *)"a=%d b=%d carry=%x => carry=%x\n", a.d1, b.d1, carry0, carry1);
#endif

  carry0 = (hi < q);
  carry1 = AS_UINT_V((a.d1 > b.d1) || (carry0 && AS_UINT_V(a.d1 == b.d1)));
#if (TRACE_KERNEL > 0)
  printf((__constant char *)"a=%d b=%d carry=%x => carry=%x\n", a.d1, b.d1, carry0, carry1);
#endif


  /*
  b=invmod2pow90(a);
  r=neginvmod2pow90(a);

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"test: a=%x:%x:%x:%x:%x:%x, invmod2pow90(a)=%x:%x:%x:%x:%x:%x(b)  neg= %x:%x:%x:%x:%x:%x(r)\n",
        a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        r.d5.s0, r.d4.s0, r.d3.s0, r.d2.s0, r.d1.s0, r.d0.s0);
#endif

  mul_90(&b, a, b);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"test: a*b= %x:%x:%x:%x:%x:%x\n",
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
  if (tid==TRACE_TID) printf((__constant char *)"test: as=%x:%x:%x:%x:%x:%x, -a=%x:%x:%x:%x:%x:%x\n",
        as.d5.s0, as.d4.s0, as.d3.s0, as.d2.s0, as.d1.s0, as.d0.s0,
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0);
#endif

  m = mod_REDC90(as, a, r);

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"test: mod_REDC90(as, a, r)=%x:%x:%x:%x:%x:%x\n",
        m.d5.s0, m.d4.s0, m.d3.s0, m.d2.s0, m.d1.s0, m.d0.s0);
#endif

  m = squaremod_REDC90(as, a, r);

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"test: mulmod_REDC90(as, as, a, r)=%x:%x:%x:%x:%x:%x\n",
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
  if (tid==TRACE_TID) printf((__constant char *)"test: a=%x:%x:%x:%x:%x:%x * b=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x:%x:%x:%x:...\n",
        a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        resv.db.s0, resv.da.s0, resv.d9.s0, resv.d8.s0, resv.d7.s0, resv.d6.s0, resv.d5.s0, resv.d4.s0, resv.d3.s0);
#endif
  inc_if_ge_90(&a, a, b);
  mul_90_180_no_low3(&resv, a, b);
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"test: a=%x:%x:%x:%x:%x:%x * b=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x:%x:%x:%x:...\n",
        a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        resv.db.s0, resv.da.s0, resv.d9.s0, resv.d8.s0, resv.d7.s0, resv.d6.s0, resv.d5.s0, resv.d4.s0, resv.d3.s0);
#endif

  ff= CONVERT_FLOAT_RTP_V(mad24(a.d5, 32768u, a.d4));
  ff= ff * 32768.0f * 32768.0f+ CONVERT_FLOAT_RTP_V(mad24(a.d3, 32768u, a.d2));   // f.d1 needed?

  //ff= as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we always underestimate the quotient
  ff= as_float(0x3f7ffffd) / ff;		// just a little bit below 1.0f so we always underestimate the quotient
  */
  b.d0=11;
  b.d1=0;
  b.d2=0x0;
  b.d3=0x0;
  b.d4=0x0;
  b.d5=0x7FFF;

  i=mad24(V(resv.db), 32768u, V(resv.da))>>2;
  f=(mad24(V(resv.db), 32768u, V(resv.da))<<30) + mad24(V(resv.d9), 32768u, V(resv.d8));
#if defined USE_DP
  div_180_90_d(&r, i, a, ff
#else
  div_180_90(&r, i, a, ff
#endif
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  );
  // enforce evaluation ... otherwise some calculations are optimized away ;-)
  res[30] = V(r.d5) + V(r.d4) + V(r.d3) + V(r.d2) + V(r.d1) + V(r.d0);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"test: %x:%x:120x0 / a=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x\n",
        i, f, 
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(r.d5), V(r.d4), V(r.d3), V(r.d2), V(r.d1), V(r.d0));
#endif

#if defined USE_DP
  ff= CONVERT_DOUBLE_RTP_V(mad24(b.d5, 32768u, b.d4));
  ff= ff * 32768.0 * 32768.0+ CONVERT_DOUBLE_RTP_V(mad24(b.d3, 32768u, b.d2));   // f.d1 needed?
  ff= as_double(0x3feffffffffffffdL) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.
#else
  ff= CONVERT_FLOAT_RTP_V(mad24(b.d5, 32768u, b.d4));
  ff= ff * 32768.0f * 32768.0f+ CONVERT_FLOAT_RTP_V(mad24(b.d3, 32768u, b.d2));   // f.d1 needed?
  ff= as_float(0x3f7ffffd) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.
#endif
        
  i=mad24(V(resv.db), 32768u, V(resv.da))>>2;
  f=(mad24(V(resv.db), 32768u, V(resv.da))<<30) + mad24(V(resv.d9), 32768u, V(resv.d8));

#if defined USE_DP
  div_180_90_d(&r, i, b, ff
#else
  div_180_90(&r, i, b, ff
#endif
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  );
  // enforce evaluation ... otherwise some calculations are optimized away ;-)
  //res[31] = r.d5.s0 + r.d4.s0 + r.d3.s0 + r.d2.s0 + r.d1.s0 + r.d0.s0;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"test: %x:%x:120x0 / b=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x\n",
        i, f, 
        V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0),
        V(r.d5), V(r.d4), V(r.d3), V(r.d2), V(r.d1), V(r.d0));
#endif

  int75_v a75, b75;
  int150_v r75;

  a75.d0=hi & 0x7fff;
  a75.d1=(hi >> 15) & 0x7fff;
  a75.d2=0x7fff;
  a75.d3=0x7fff;
  a75.d4=0xffff;
  b75.d0=lo & 0x7fff;
  b75.d1=(lo >> 15) & 0x7fff;
  b75.d2=0x7fff;
  b75.d3=0x7fff;
  b75.d4=0x7fff;

  mul_75_150_no_low5_big(&r75, a75, b75);
#if (TRACE_KERNEL > 0)
  printf((__constant char *) "no_low5_big: a=%x:%x:%x:%x:%x * b=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x:%x:0:0:0:0\n",
       V(a75.d4), V(a75.d3), V(a75.d2), V(a75.d1), V(a75.d0), V(b75.d4), V(b75.d3), V(b75.d2), V(b75.d1), V(b75.d0),
       V(r75.d9), V(r75.d8), V(r75.d7), V(r75.d6), V(r75.d5), V(r75.d4));
#endif
  mul_75_150_no_low5(&r75, a75, b75);
#if (TRACE_KERNEL > 0)
  printf((__constant char *) "no_low5: a=%x:%x:%x:%x:%x * b=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x:%x:0:0:0:0\n",
       V(a75.d4), V(a75.d3), V(a75.d2), V(a75.d1), V(a75.d0), V(b75.d4), V(b75.d3), V(b75.d2), V(b75.d1), V(b75.d0),
       V(r75.d9), V(r75.d8), V(r75.d7), V(r75.d6), V(r75.d5), V(r75.d4));
#endif
  mul_75_150_no_low3(&r75, a75, b75);
#if (TRACE_KERNEL > 0)
  printf((__constant char *) "no_low3: a=%x:%x:%x:%x:%x * b=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x:%x:%x:%x:0:0\n",
       V(a75.d4), V(a75.d3), V(a75.d2), V(a75.d1), V(a75.d0), V(b75.d4), V(b75.d3), V(b75.d2), V(b75.d1), V(b75.d0),
       V(r75.d9), V(r75.d8), V(r75.d7), V(r75.d6), V(r75.d5), V(r75.d4), V(r75.d3), V(r75.d2));
#endif
  mul_75_150(&r75, a75, b75);
#if (TRACE_KERNEL > 0)
  printf((__constant char *) "exact: a=%x:%x:%x:%x:%x * b=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x\n",
       V(a75.d4), V(a75.d3), V(a75.d2), V(a75.d1), V(a75.d0), V(b75.d4), V(b75.d3), V(b75.d2), V(b75.d1), V(b75.d0),
       V(r75.d9), V(r75.d8), V(r75.d7), V(r75.d6), V(r75.d5), V(r75.d4), V(r75.d3), V(r75.d2), V(r75.d1), V(r75.d0));
#endif


  f=tid+1; // let the reported results start with 1
  if (1 == 1)
  {
    i=ATOMIC_INC(res[0]);
#if (TRACE_KERNEL > 0)
    printf((__constant char *)"thread %d: i=%d, res[0]=%d\n", get_global_id(0), i, res[0]);
#endif

    if(i<10)				/* limit to 10 results */
    {
      res[i*3 + 1]=f;
      res[i*3 + 2]=f;
      res[i*3 + 3]=f;
    }
  }
}
