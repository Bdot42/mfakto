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

#define EVAL_RES_b(comp) \
  if((a.d2.comp|a.d1.comp)==0 && a.d0.comp==1) \
  { \
      int index = ATOMIC_INC(RES[0]); \
      if (index < 10) \
      { \
        RES[index *3 + 1]=f.d2.comp; \
        RES[index *3 + 2]=f.d1.comp; \
        RES[index *3 + 3]=f.d0.comp; \
      } \
  }

#define EVAL_RES_a(comp) \
  if((a.d2.comp|a.d1.comp)==0 && a.d0.comp==1) \
  { \
    if ((f.d2.comp|f.d1.comp)!=0 || f.d0.comp != 1) \
    { \
      int index = ATOMIC_INC(RES[0]); \
      if (index < 10) \
      { \
        RES[index *3 + 1]=f.d2.comp; \
        RES[index *3 + 2]=f.d1.comp; \
        RES[index *3 + 3]=f.d0.comp; \
      } \
    } \
  }

#define EVAL_RES_l(comp) \
  if(a.comp == 1) \
  { \
      int index =ATOMIC_INC(RES[0]); \
      if(index <10) \
      { \
        RES[index *3 + 1]=0; \
        RES[index *3 + 2]=convert_uint(f.comp >> 32); \
        RES[index *3 + 3]=convert_uint(f.comp); \
      } \
  }

#define EVAL_RES_tmp(comp) \
  if((tmp.comp)==0) \
  { \
      int index = ATOMIC_INC(RES[0]); \
      if (index < 10) \
      { \
        RES[index *3 + 1]=n.d2.comp; \
        RES[index *3 + 2]=n.d1.comp; \
        RES[index *3 + 3]=n.d0.comp; \
      } \
  }


void check_factor96(const int96_v f, const int96_v a, __global uint * const RES)
/* Check whether f is a factor or not. If f != 1 and a == 1 then f is a factor,
in this case f is written into the RES array. */
{
#if (VECTOR_SIZE == 1)
  if(((a.d2|a.d1) == 0) && (a.d0 == 1))
  {
    if(f.d2 != 0 || f.d1 != 0 || f.d0 != 1)	/* 1 isn't really a factor ;) */
    {
#if (TRACE_KERNEL > 0)  // trace this for any thread
      printf((__constant char *)"tid=%ld found factor: q=%x:%x:%x\n", get_local_id(0), f.d2.s0, f.d1.s0, f.d0.s0);
#endif
      int index=ATOMIC_INC(RES[0]);
      if(index < 10)				/* limit to 10 factors per class */
      {
        RES[index * 3 + 1] = f.d2;
        RES[index * 3 + 2] = f.d1;
        RES[index * 3 + 3] = f.d0;
      }
    }
  }
#elif (VECTOR_SIZE == 2)
  EVAL_RES_a(x)
  EVAL_RES_a(y)
#elif (VECTOR_SIZE == 3)
  EVAL_RES_a(x)
  EVAL_RES_a(y)
  EVAL_RES_a(z)
#elif (VECTOR_SIZE == 4)
  EVAL_RES_a(x)
  EVAL_RES_a(y)
  EVAL_RES_a(z)
  EVAL_RES_a(w)
#elif (VECTOR_SIZE == 8)
  EVAL_RES_a(s0)
  EVAL_RES_a(s1)
  EVAL_RES_a(s2)
  EVAL_RES_a(s3)
  EVAL_RES_a(s4)
  EVAL_RES_a(s5)
  EVAL_RES_a(s6)
  EVAL_RES_a(s7)
#elif (VECTOR_SIZE == 16)
  EVAL_RES_a(s0)
  EVAL_RES_a(s1)
  EVAL_RES_a(s2)
  EVAL_RES_a(s3)
  EVAL_RES_a(s4)
  EVAL_RES_a(s5)
  EVAL_RES_a(s6)
  EVAL_RES_a(s7)
  EVAL_RES_a(s8)
  EVAL_RES_a(s9)
  EVAL_RES_a(sa)
  EVAL_RES_a(sb)
  EVAL_RES_a(sc)
  EVAL_RES_a(sd)
  EVAL_RES_a(se)
  EVAL_RES_a(sf)
#endif
}


void check_big_factor96(const int96_v f, const int96_v a, __global uint * const RES)
/* Similar to check_factor96() but without checking f != 1. This is a little
bit faster but only safe for kernel which have a lower limit well above 1. The
barrett based kernels have a lower limit of 2^64 so this function is used
there. */
{
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf((__constant char *)"tid=%ld found factor: q=%x:%x:%x\n", get_local_id(0), f.d2.s0, f.d1.s0, f.d0.s0);
#endif
/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f != 1 */
    int index=ATOMIC_INC(RES[0]);
    if(index<10)				/* limit to 10 factors per class */
    {
      RES[index*3 + 1]=f.d2;
      RES[index*3 + 2]=f.d1;
      RES[index*3 + 3]=f.d0;
    }
  }
#elif (VECTOR_SIZE == 2)
  EVAL_RES_b(x)
  EVAL_RES_b(y)
#elif (VECTOR_SIZE == 3)
  EVAL_RES_b(x)
  EVAL_RES_b(y)
  EVAL_RES_b(z)
#elif (VECTOR_SIZE == 4)
  EVAL_RES_b(x)
  EVAL_RES_b(y)
  EVAL_RES_b(z)
  EVAL_RES_b(w)
#elif (VECTOR_SIZE == 8)
  EVAL_RES_b(s0)
  EVAL_RES_b(s1)
  EVAL_RES_b(s2)
  EVAL_RES_b(s3)
  EVAL_RES_b(s4)
  EVAL_RES_b(s5)
  EVAL_RES_b(s6)
  EVAL_RES_b(s7)
#elif (VECTOR_SIZE == 16)
  EVAL_RES_b(s0)
  EVAL_RES_b(s1)
  EVAL_RES_b(s2)
  EVAL_RES_b(s3)
  EVAL_RES_b(s4)
  EVAL_RES_b(s5)
  EVAL_RES_b(s6)
  EVAL_RES_b(s7)
  EVAL_RES_b(s8)
  EVAL_RES_b(s9)
  EVAL_RES_b(sa)
  EVAL_RES_b(sb)
  EVAL_RES_b(sc)
  EVAL_RES_b(sd)
  EVAL_RES_b(se)
  EVAL_RES_b(sf)
#endif
}


void create_FC96(int96_v * const f, const uint exp, const int96_t k, const uint_v k_offset)
/* calculates f = 2 * (k+k_offset) * exp + 1 */
{
  int96_t exp96;
  int96_v my_k;
  uint_v tmp;

  exp96.d1 = exp >> 31;
  exp96.d0 = exp << 1;			// exp96 = 2 * exp

  my_k.d0 = mad24(k_offset, NUM_CLASSES, k.d0);
  my_k.d1 = k.d1 + mul_hi(k_offset, NUM_CLASSES) + AS_UINT_V(k.d0 > my_k.d0 ? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */

  f->d0 = 1 + my_k.d0 * exp96.d0;
  tmp   = exp96.d1 ? my_k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f->d2  = exp96.d1 ? my_k.d1 : 0;

  f->d1  = mul_hi(my_k.d0, exp96.d0) + tmp;
  f->d2 += AS_UINT_V((tmp > f->d1)? 1 : 0);

  tmp   = my_k.d1 * exp96.d0;
  f->d1 += tmp;

  f->d2 += mul_hi(my_k.d1, exp96.d0) + AS_UINT_V((tmp > f->d1)? 1 : 0); 	// f = 2 * k * exp + 1
}


void create_FC96_mad(int96_v * const f, const uint exp, const int96_t k, const uint_v k_offset)
/* similar to create_FC96(), this versions uses multiply-add with carry which
is faster for _SOME_ kernels. */
{
  int96_t exp96;
  int96_v my_k;
  uint_v tmp;

  exp96.d1 = exp >> 31;
  exp96.d0 = exp << 1;			// exp96 = 2 * exp

  my_k.d0 = mad24(k_offset, NUM_CLASSES, k.d0);
  my_k.d1 = k.d1 + mul_hi(k_offset, NUM_CLASSES) + AS_UINT_V(k.d0 > my_k.d0 ? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */

  f->d0 = 1 + my_k.d0 * exp96.d0;
  tmp   = exp96.d1 ? my_k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f->d2  = exp96.d1 ? my_k.d1 : 0;

  f->d1  = mad_hi(my_k.d0, exp96.d0, tmp);
  f->d2 += AS_UINT_V((tmp > f->d1)? 1 : 0);

  tmp   = my_k.d1 * exp96.d0;
  f->d1 += tmp;

  f->d2 += mad_hi(my_k.d1, exp96.d0, AS_UINT_V((tmp > f->d1)? 1 : 0)); 	// f = 2 * k * exp + 1
}


void mod_simple_96(int96_v * const res, const int96_v q, const int96_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
)
/*
res = q mod n
used for refinement in barrett modular multiplication
assumes q < 6n (6n includes "optional mul 2")
*/
{
  __private float_v qf;
  __private uint_v qi, tmp, carry;
  __private int96_v nn;

  qf = CONVERT_FLOAT_V(q.d2);
  qf = qf * 4294967296.0f + CONVERT_FLOAT_V(q.d1);

  qi = CONVERT_UINT_V(qf*nf);

#ifdef CHECKS_MODBASECASE
/* all barrett based kernels are made for factor candidates above 2^64,
atleast the 79bit variant fails on factor candidates less than 2^64!
Lets ignore those errors...
Factor candidates below 2^64 can occur when TFing from 2^64 to 2^65, the
first candidate in each class can be smaller than 2^64.
This is NOT an issue because those exponents should be TFed to 2^64 with a
kernel which can handle those "small" candidates before starting TF from
2^64 to 2^65. So in worst case we have a false positive which is cought
easily by the primenetserver.
The same applies to factor candidates which are bigger than 2^bit_max for the
barrett92 kernel. If the factor candidate is bigger than 2^bit_max then
usually just the correction factor is bigger than expected. There are tons
of messages that qi is to high (better: higher than expected) e.g. when trial
factoring huge exponents from 2^64 to 2^65 with the barrett92 kernel (during
selftest). The factor candidates might be as high a 2^68 in some of these
cases! This is related to the _HUGE_ blocks that mfaktc processes at once.
To make it short: let's ignore warnings/errors from factor candidates which
are "out of range".
*/
  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 100, qi, 12);
  }
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mod_simple_96: q=%x:%x:%x, n=%x:%x:%x, nf=%G, qf=%G, qi=%x\n",
        q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, nf.s0, qf.s0, qi.s0);
#endif

  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V((tmp > nn.d1)? 1 : 0);
  nn.d2 += mul_hi(n.d1, qi) + n.d2* qi;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char*)"mod_simple_96: nn=%x:%x:%x\n",
        nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  carry= AS_UINT_V((nn.d0 > q.d0) ? 1 : 0);
  res->d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - carry;
  carry= AS_UINT_V(((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1))) ? 1 : 0);
  res->d1 = tmp;

  res->d2 = q.d2 - nn.d2 - carry;
}


void mod_simple_96_and_check_big_factor96(const int96_v q,const int96_v n, const float_v nf, __global uint * const RES)
/*
This function is a combination of mod_simple_96(), check_big_factor96() and an additional correction step.
If q mod n == 1 then n is a factor and written into the RES array.
q must be less than 100n!
*/
{
  __private float_v qf;
  __private uint_v qi, tmp;
  __private int96_v nn;

  qf = CONVERT_FLOAT_V(q.d2);
  qf = qf * 4294967296.0f + CONVERT_FLOAT_V(q.d1);

  qi = CONVERT_UINT_V(qf*nf);

/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi += ((~qi) ^ q.d0) & 1;
 
  nn.d0  = n.d0 * qi + 1;
  if(any(q.d0 == nn.d0)) /* the lowest word of the final result would be 1 for at least one of the vector components (only in this case n might be a factor) */
  { // it would be sufficient to calculate the one component that made the above "any" return true. But it would require a bigger EVAL macro ...
    nn.d1  = mul_hi(n.d0, qi);
    tmp    = n.d1* qi;
    nn.d1 += tmp;
    nn.d2  = AS_UINT_V((tmp > nn.d1)? 1 : 0);
    nn.d2 += mul_hi(n.d1, qi) + n.d2* qi;

    tmp  = q.d1 - nn.d1;
    tmp |= q.d2 - nn.d2 - AS_UINT_V(tmp > q.d1 ? 1 : 0);

#if (VECTOR_SIZE == 1)
    if(tmp == 0)
    {
      int index;
      index =ATOMIC_INC(RES[0]);
      if(index < 10)                              /* limit to 10 factors per class */
      {
        RES[index * 3 + 1] = n.d2;
        RES[index * 3 + 2] = n.d1;
        RES[index * 3 + 3] = n.d0;
      }
    }
#elif (VECTOR_SIZE == 2)
  EVAL_RES_tmp(x)
  EVAL_RES_tmp(y)
#elif (VECTOR_SIZE == 3)
  EVAL_RES_tmp(x)
  EVAL_RES_tmp(y)
  EVAL_RES_tmp(z)
#elif (VECTOR_SIZE == 4)
  EVAL_RES_tmp(x)
  EVAL_RES_tmp(y)
  EVAL_RES_tmp(z)
  EVAL_RES_tmp(w)
#elif (VECTOR_SIZE == 8)
  EVAL_RES_tmp(s0)
  EVAL_RES_tmp(s1)
  EVAL_RES_tmp(s2)
  EVAL_RES_tmp(s3)
  EVAL_RES_tmp(s4)
  EVAL_RES_tmp(s5)
  EVAL_RES_tmp(s6)
  EVAL_RES_tmp(s7)
#elif (VECTOR_SIZE == 16)
  EVAL_RES_tmp(s0)
  EVAL_RES_tmp(s1)
  EVAL_RES_tmp(s2)
  EVAL_RES_tmp(s3)
  EVAL_RES_tmp(s4)
  EVAL_RES_tmp(s5)
  EVAL_RES_tmp(s6)
  EVAL_RES_tmp(s7)
  EVAL_RES_tmp(s8)
  EVAL_RES_tmp(s9)
  EVAL_RES_tmp(sa)
  EVAL_RES_tmp(sb)
  EVAL_RES_tmp(sc)
  EVAL_RES_tmp(sd)
  EVAL_RES_tmp(se)
  EVAL_RES_tmp(sf)
#endif
  }
}

// for the GPU sieve, we don't implement some kernels
#ifdef CL_GPU_SIEVE
  #include "gpusieve.cl"
  #include "barrett15.cl"  // mul24-based barrett 73-bit kernel using a word size of 15 bit
  #include "barrett.cl"   // one kernel file for 32-bit-barrett of different vector sizes (1, 2, 4, 8, 16)
#else
  #define EVAL_RES(x) EVAL_RES_b(x)  // no check for f==1 if running the "big" version

  #include "barrett15.cl"  // mul24-based barrett 73-bit kernel using a word size of 15 bit
  #include "barrett.cl"   // one kernel file for 32-bit-barrett of different vector sizes (1, 2, 4, 8, 16)

  #include "mul24.cl" // one kernel file for 24-bit-kernels of different vector sizes (1, 2, 4, 8, 16)
  #include "barrett24.cl"  // mul24-based barrett 72-bit kernel (all vector sizes)
  #include "montgomery.cl"  // montgomery kernels

  #define _63BIT_MUL24_K
  #undef EVAL_RES
  #define EVAL_RES(x) EVAL_RES_a(x)
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
  float_v ff = 1.0f;


//  barrier(CLK_GLOBAL_MEM_FENCE);
#if (TRACE_KERNEL > 0)
    printf((__constant char *)"kernel tracing level %d enabled\n", TRACE_KERNEL);
#endif
  a.d0=1;
  a.d1=0;
  a.d2=0;
  a.d3=0x0003;
  a.d4=0x7000;
  a.d5=0x0010;

  square_90_180(&resv, a);

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
#if (VECTOR_SIZE > 1)
  b.d0=11;
  b.d1=0;
  b.d2=0x0;
  b.d3=0x0;
  b.d4=0x0;
  b.d5=0x3FFF;

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
  if (tid==TRACE_TID) printf((__constant char *)"test: %x:%x:120x0 / a=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x\n",
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
  if (tid==TRACE_TID) printf((__constant char *)"test: %x:%x:120x0 / b=%x:%x:%x:%x:%x:%x  = %x:%x:%x:%x:%x:%x\n",
        i, f, 
        b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        r.d5.s0, r.d4.s0, r.d3.s0, r.d2.s0, r.d1.s0, r.d0.s0);
#endif


  f=tid+1; // let the reported results start with 1
  if (1 == 1)
  {
    i=ATOMIC_INC(res[0]);
    printf((__constant char *)"thread %d: i=%d, res[0]=%d\n", get_global_id(0), i, res[0]);

    if(i<10)				/* limit to 10 results */
    {
      res[i*3 + 1]=f;
      res[i*3 + 2]=f;
      res[i*3 + 3]=f;
    }
  }
}
