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
 Common functions and defines for various kernels
*/

// prototypes

void check_big_factor96(const int96_v f, const int96_v a, __global uint * const RES);

void calculate_FC32(const uint exponent, const uint tid, const __global uint * restrict k_tab, const int96_t k_base, __private int96_v * restrict const f);

void calculate_FC32_mad(const uint exponent, const uint tid, const __global uint * restrict k_tab, const int96_t k_base, __private int96_v * restrict const f);

void mod_simple_96(int96_v * const res, const int96_v q, const int96_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_96_and_check_big_factor96(const int96_v q,const int96_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_even_96_and_check_big_factor96(const int96_v q,const int96_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_75(int75_v * const res, const int75_v q, const int75_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_75_big(int75_v * const res, const int75_v q, const int75_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_even_75_and_check_big_factor75(const int75_v q, const int75_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_even_75_and_check_big_factor75_big(const int75_v q, const int75_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_75_and_check_big_factor75(const int75_v q, const int75_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_90_and_check_big_factor90(const int90_v q, const int90_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void calculate_FC75(const uint exponent, const uint tid, const __global uint * restrict k_tab, const int75_t k_base, __private int75_v * restrict const f);

void mod_simple_90(int90_v * const res, const int90_v q, const int90_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void mod_simple_even_90_and_check_big_factor90(const int90_v q, const int90_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
);

void calculate_FC90(const uint exponent, const uint tid, const __global uint * restrict k_tab, const int75_t k_base, __private int90_v * restrict const f);

// portability

#if ( __OPENCL_VERSION__ < 120 )
// in OpenCL 1.1, we're missing popcount and printf
#ifdef cl_amd_printf
#pragma "Enabling printf"
#pragma  OPENCL EXTENSION cl_amd_printf : enable
#else
// hack to make printfs ignored when unknown (e.g. on NVIDIA)
#pragma "Defining printf"
#define printf(x)
#endif

#ifdef cl_amd_popcnt
#pragma  OPENCL EXTENSION cl_amd_popcnt : enable
#pragma "Replacing popcount"
#define popcount popcnt
#else
#pragma "Emulating popcount"
uint popcount(uint x)
{
  x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x & 0x07070707) + ((x >> 4) & 0x07070707);
  x = (x & 0x000f000f) + ((x >> 8) & 0x000f000f);
  x = (x & 0x0000000f) + ((x >> 16) & 0x0000000f);
  return x;
}
#endif
#endif // OCL < 1.2

#if defined GCN || GCN4 || GCN5 || RDNA || NVIDIA
#define SLOW_DP
#endif

#if defined USE_DP || cl_khr_fp64 && ! defined SLOW_DP
#pragma  OPENCL EXTENSION cl_khr_fp64 : enable
#define USE_DP
#else
#pragma "No double precision available"
#endif

// HD4xxx does not have atomics, but mfakto will work on these GPUs as well.
// Without atomics, the factors found may be scrambled when more than one
// factor is found per grid => if the reported factor(s) are not accepted
// by primenet, then run the bitlevel again with the smallest possible grid size,
// or run it on at least HD5...

#ifdef cl_khr_global_int32_base_atomics 
#pragma  OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMIC_INC(x) atomic_inc(&x)
#else
#pragma "Replacing atomic_inc by non-atomics"
// No atomic operations available - using simple ++
#define ATOMIC_INC(x) ((x)++)
#endif

#ifdef cl_amd_media_ops
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#else
#pragma "Emulating amd_bitalign"
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
#pragma "Emulating amd_max3"
// we need to define what we need:
//     Build-in Function
//      uintn amd_max3 (uintn src0, uintn src1, uintn src2)
#define amd_max3(src0, src1, src2)  max(src0, max(src1, src2))
#endif

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

// #define EVAL_RES_a(comp)  // this was checking for a == 1 and f != 1. The f != 1 is now done on the host

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

/****************************************
 ****************************************
 * 15-bit-stuff for the 90/75-bit-barrett-kernel
 * included by main kernel file
 ****************************************
 ****************************************/

// 75-bit
#define EVAL_RES_75(comp) \
  if(amd_max3(a.d4.comp, a.d3.comp, a.d2.comp|a.d1.comp)==0 && a.d0.comp==1) \
  { \
      tid=ATOMIC_INC(RES[0]); \
      if(tid<10) \
      { \
        RES[tid*3 + 1]=f.d4.comp;  \
        RES[tid*3 + 2]=mad24(f.d3.comp,0x8000u, f.d2.comp); \
        RES[tid*3 + 3]=mad24(f.d1.comp,0x8000u, f.d0.comp); \
      } \
  }

// 90-bit
#define EVAL_RES_90(comp) \
  if(amd_max3(amd_max3(a.d5.comp, a.d4.comp, a.d3.comp), a.d2.comp, a.d1.comp)==0 && a.d0.comp==1) \
  { \
      tid=ATOMIC_INC(RES[0]); \
      if(tid<10) \
      { \
        RES[tid*3 + 1]=mad24(f.d5.comp,0x8000u, f.d4.comp); \
        RES[tid*3 + 2]=mad24(f.d3.comp,0x8000u, f.d2.comp); \
        RES[tid*3 + 3]=mad24(f.d1.comp,0x8000u, f.d0.comp); \
      } \
  }

#define EVAL_RES_tmp75(comp) \
  if((tmp.comp)==0) \
  { \
      int index = ATOMIC_INC(RES[0]); \
      if (index < 10) \
      { \
        RES[index*3 + 1]=n.d4.comp;  \
        RES[index*3 + 2]=mad24(n.d3.comp,0x8000u, n.d2.comp); \
        RES[index*3 + 3]=mad24(n.d1.comp,0x8000u, n.d0.comp); \
      } \
  }

#define EVAL_RES_tmp90(comp) \
  if((tmp.comp)==0) \
  { \
      int index = ATOMIC_INC(RES[0]); \
      if (index < 10) \
      { \
        RES[index*3 + 1]=mad24(n.d5.comp,0x8000u, n.d4.comp); \
        RES[index*3 + 2]=mad24(n.d3.comp,0x8000u, n.d2.comp); \
        RES[index*3 + 3]=mad24(n.d1.comp,0x8000u, n.d0.comp); \
      } \
  }

void check_big_factor96(const int96_v f, const int96_v a, __global uint * const RES)
/* Similar to check_factor96() but without checking f != 1. This is a little
bit faster. The barrett based kernels have a lower limit of 2^64 so this function is used
there. The check for f != 1 is now done in the host program, therefore this function is
now used everywhere. */
{
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf((__constant char *)"tid=%ld found factor: q=%x:%x:%x\n", get_local_id(0), f.d2, f.d1, f.d0);
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


/*
 * read k from global memory and calculate the factor candidate
 * this is common for all CPU-sieve-based kernels
 */

void calculate_FC32(const uint exponent, const uint tid, const __global uint * restrict k_tab, const int96_t k_base, __private int96_v * restrict const f)
{
  __private int96_t exp96;
  __private uint_v t, tmp;
  __private int96_v k;

  exp96.d1=exponent>>31;exp96.d0=exponent+exponent;	// exp96 = 2 * exponent

#if (VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
#elif (VECTOR_SIZE == 3)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
#elif (VECTOR_SIZE == 4)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
  t.w  = k_tab[tid+3];
#elif (VECTOR_SIZE == 8)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
#elif (VECTOR_SIZE == 16)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
  t.s8 = k_tab[tid+8];
  t.s9 = k_tab[tid+9];
  t.sa = k_tab[tid+10];
  t.sb = k_tab[tid+11];
  t.sc = k_tab[tid+12];
  t.sd = k_tab[tid+13];
  t.se = k_tab[tid+14];
  t.sf = k_tab[tid+15];
#endif
//MAD only available for float
  k.d0 = mad24(t, 4620u, k_base.d0);
#ifdef INTEL
  // WA for intel optimizer bug. exponent has a min limit of 2^10, so the % will not change the value
  k.d1 = mul_hi(t, (4620 % (exponent + 1))) + k_base.d1 - AS_UINT_V(k_base.d0 > k.d0);
#else
  k.d1 = mul_hi(t, 4620u) + k_base.d1 - AS_UINT_V(k_base.d0 > k.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"calculate_FC32: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, V(t), V(k.d2), V(k.d1), V(k.d0));
#endif

  f->d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f->d2  = exp96.d1 ? k.d1 : 0;

  f->d1  = mul_hi(k.d0, exp96.d0) + tmp;
  f->d2 -= AS_UINT_V(tmp > f->d1);

  tmp   = k.d1 * exp96.d0;
  f->d1 += tmp;

  f->d2 += mul_hi(k.d1, exp96.d0) - AS_UINT_V(tmp > f->d1); 	// f = 2 * k * exp + 1
}

/*
 * read k from global memory and calculate the factor candidate
 * this is common for all CPU-sieve-based kernels
 * this version used mad_hi instead of mul_hi in 2 places - for some kernels this is faster
 */

void calculate_FC32_mad(const uint exponent, const uint tid, const __global uint * restrict k_tab, const int96_t k_base, __private int96_v * restrict const f)
{
  __private int96_t exp96;
  __private uint_v t, tmp;
  __private int96_v k;

  exp96.d1=exponent>>31;exp96.d0=exponent+exponent;	// exp96 = 2 * exponent

#if (VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
#elif (VECTOR_SIZE == 3)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
#elif (VECTOR_SIZE == 4)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
  t.w  = k_tab[tid+3];
#elif (VECTOR_SIZE == 8)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
#elif (VECTOR_SIZE == 16)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
  t.s8 = k_tab[tid+8];
  t.s9 = k_tab[tid+9];
  t.sa = k_tab[tid+10];
  t.sb = k_tab[tid+11];
  t.sc = k_tab[tid+12];
  t.sd = k_tab[tid+13];
  t.se = k_tab[tid+14];
  t.sf = k_tab[tid+15];
#endif
//MAD only available for float
  k.d0 = mad24(t, 4620u, k_base.d0);
#ifdef INTEL
  // WA for intel optimizer bug. exponent has a min limit of 2^10, so the % will not change the value
  k.d1 = mul_hi(t, (4620 % (exponent + 1))) + k_base.d1 - AS_UINT_V(k_base.d0 > k.d0);
#else
  k.d1 = mul_hi(t, 4620u) + k_base.d1 - AS_UINT_V(k_base.d0 > k.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"calculate_FC32_mad: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, V(t), V(k.d2), V(k.d1), V(k.d0));
#endif

  f->d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f->d2  = exp96.d1 ? k.d1 : 0;

  f->d1  = mad_hi(k.d0, exp96.d0, tmp);
  f->d2 -= AS_UINT_V(tmp > f->d1);

  tmp   = k.d1 * exp96.d0;
  f->d1 += tmp;

  f->d2  = mad_hi(k.d1, exp96.d0, f->d2) - AS_UINT_V(tmp > f->d1); 	// f = 2 * k * exp + 1
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
    MODBASECASE_QI_ERROR(limit, 102, qi, 14);
  }
#endif
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"mod_simple_96: q=%x:%x:%x, n=%x:%x:%x, qi=%x, tid=%u\n",
        V(q.d2), V(q.d1), V(q.d0), V(n.d2), V(n.d1), V(n.d0), V(qi), tid);
#endif

  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  nn.d2  = mul_hi(n.d1, qi) + n.d2* qi - nn.d2;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char*)"mod_simple_96: nn=%x:%x:%x\n",
        V(nn.d2), V(nn.d1), V(nn.d0));
#endif

  carry= AS_UINT_V(nn.d0 > q.d0);
  res->d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 + carry;
  carry= AS_UINT_V((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1)));
  res->d1 = tmp;

  res->d2 = q.d2 - nn.d2 + carry;
}


void mod_simple_96_and_check_big_factor96(const int96_v q,const int96_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
)
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

#ifdef CHECKS_MODBASECASE
  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(10, 103, qi, 15);
  }
#endif

/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi += ((~qi) ^ q.d0) & 1;
 
  nn.d0  = n.d0 * qi + 1;
#if (VECTOR_SIZE == 1)
  if(q.d0 == nn.d0)
#else
  if(any(q.d0 == nn.d0)) /* the lowest word of the final result would be 1 for at least one of the vector components (only in this case n might be a factor) */
#endif
  { // it would be sufficient to calculate the one component that made the above "any" return true. But it would require a bigger EVAL macro ...
    nn.d1  = mul_hi(n.d0, qi);
    tmp    = n.d1* qi;
    nn.d1 += tmp;
    nn.d2  = AS_UINT_V(tmp > nn.d1);
    nn.d2  = mul_hi(n.d1, qi) + n.d2* qi - nn.d2;

    tmp  = q.d1 - nn.d1;
//    tmp |= q.d2 - nn.d2 - AS_UINT_V(tmp > q.d1 ? 1 : 0);
    tmp |= q.d2 - nn.d2; // if we have a borrow here, then tmp will not be 0 anyway

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

void mod_simple_even_96_and_check_big_factor96(const int96_v q,const int96_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
)
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

#ifdef CHECKS_MODBASECASE
  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(100, 104, qi, 16);
  }
#endif

/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi |= 1;
 
  nn.d0  = n.d0 * qi + 1;
#if (VECTOR_SIZE == 1)
  if(q.d0 == nn.d0)
#else
  if(any(q.d0 == nn.d0)) /* the lowest word of the final result would be 1 for at least one of the vector components (only in this case n might be a factor) */
#endif
  { // it would be sufficient to calculate the one component that made the above "any" return true. But it would require a bigger EVAL macro ...
    nn.d1  = mul_hi(n.d0, qi);
    tmp    = n.d1* qi;
    nn.d1 += tmp;
    nn.d2  = AS_UINT_V(tmp > nn.d1);
    nn.d2  = mul_hi(n.d1, qi) + n.d2* qi - nn.d2;

    tmp  = q.d1 - nn.d1;
//    tmp |= q.d2 - nn.d2 - AS_UINT_V(tmp > q.d1 ? 1 : 0);
    tmp |= q.d2 - nn.d2; // if we have a borrow here, then tmp will not be 0 anyway

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


void mod_simple_75(int75_v * const res, const int75_v q, const int75_v n, const float_v nf
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
  __private uint_v qi;
  __private int75_v nn;

  qf = CONVERT_FLOAT_V(mad24(q.d4, 32768u, q.d3));
  qf = qf * 32768.0f;
  
  qi = CONVERT_UINT_V(qf*nf);

#ifdef CHECKS_MODBASECASE
/* both barrett based kernels are made for factor candidates above 2^64,
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
  if(n.d4 != 0 && n.d4 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 105, qi, 17);
  }
#endif
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"mod_simple_75#1: q=%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

// nn = n * qi
  nn.d0  = mul24(n.d0, qi);
  nn.d1  = mad24(n.d1, qi, nn.d0 >> 15);
  nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
  nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
  nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
  nn.d0 &= 0x7FFF;
  nn.d1 &= 0x7FFF;
  nn.d2 &= 0x7FFF;
  nn.d3 &= 0x7FFF;

  res->d0 = q.d0 - nn.d0;
  res->d1 = q.d1 - nn.d1 + AS_UINT_V(res->d0 > 0x7FFF);
  res->d2 = q.d2 - nn.d2 + AS_UINT_V(res->d1 > 0x7FFF);
  res->d3 = q.d3 - nn.d3 + AS_UINT_V(res->d2 > 0x7FFF);
  res->d4 = q.d4 - nn.d4 + AS_UINT_V(res->d3 > 0x7FFF);
  res->d0 &= 0x7FFF;
  res->d1 &= 0x7FFF;
  res->d2 &= 0x7FFF;
  res->d3 &= 0x7FFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_simple_75#2: nn=%x:%x:%x:%x:%x, res=%x:%x:%x:%x:%x\n",
        V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d4), V(res->d3), V(res->d2), V(res->d1), V(res->d0));
#endif

}

void mod_simple_75_big(int75_v * const res, const int75_v q, const int75_v n, const float_v nf
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
  __private uint_v qi;
  __private int75_v nn;

  qf = CONVERT_FLOAT_V(q.d4) * 1073741824.0f;
  
  qi = CONVERT_UINT_V(qf*nf);

#ifdef CHECKS_MODBASECASE
/* both barrett based kernels are made for factor candidates above 2^64,
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
  if(n.d4 != 0 && n.d4 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 105, qi, 17);
  }
#endif
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"mod_simple_75#1: q=%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

// nn = n * qi
  nn.d0  = mul24(n.d0, qi);
  nn.d1  = mad24(n.d1, qi, nn.d0 >> 15);
  nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
  nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
  nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
  nn.d0 &= 0x7FFF;
  nn.d1 &= 0x7FFF;
  nn.d2 &= 0x7FFF;
  nn.d3 &= 0x7FFF;

  res->d0 = q.d0 - nn.d0;
  res->d1 = q.d1 - nn.d1 + AS_UINT_V(res->d0 > 0x7FFF);
  res->d2 = q.d2 - nn.d2 + AS_UINT_V(res->d1 > 0x7FFF);
  res->d3 = q.d3 - nn.d3 + AS_UINT_V(res->d2 > 0x7FFF);
  res->d4 = q.d4 - nn.d4 + AS_UINT_V(res->d3 > 0x7FFF);
  res->d0 &= 0x7FFF;
  res->d1 &= 0x7FFF;
  res->d2 &= 0x7FFF;
  res->d3 &= 0x7FFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_simple_75#2: nn=%x:%x:%x:%x:%x, res=%x:%x:%x:%x:%x\n",
        V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d4), V(res->d3), V(res->d2), V(res->d1), V(res->d0));
#endif

}

void mod_simple_even_75_and_check_big_factor75(const int75_v q, const int75_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
)
/*
This function is a combination of mod_simple_96(), check_big_factor96() and an additional correction step.
If q mod n == 1 then n is a factor and written into the RES array.
q must be less than 100n!
*/
{
  __private float_v qf;
  __private uint_v qi, tmp;
  __private int75_v nn;

  qf = CONVERT_FLOAT_V(mad24(q.d4, 32768u, q.d3));
  qf = qf * 32768.0f;
  
  qi = CONVERT_UINT_V(qf*nf);

#ifdef CHECKS_MODBASECASE
  if(n.d4 != 0 && n.d4 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(10, 106, qi, 18);
  }
#endif
/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi |= 1;
 
  nn.d0  = mad24(n.d0, qi, 1u);
  nn.d1  = nn.d0 >> 15;
  nn.d0 &= 0x7FFF;

#if (TRACE_KERNEL > 1)
	  if (get_global_id(0)==TRACE_TID) printf((__constant char *)"mod_simple_e_75_a#1: q=%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

#if (VECTOR_SIZE == 1)
  if(q.d0 == nn.d0)
#else
  if(any(q.d0 == nn.d0)) /* the lowest word of the final result would be 1 for at least one of the vector components (only in this case n might be a factor) */
#endif
  { // it would be sufficient to calculate the one component that made the above "any" return true. But it would require a bigger EVAL macro ...

#if (TRACE_KERNEL > 1)
	  printf((__constant char *)"mod_simple_e_75_a#2: q=%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

// nn = n * qi
    nn.d1  = mad24(n.d1, qi, nn.d1);
    nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
    nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
    nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
    nn.d1 &= 0x7FFF;
    nn.d2 &= 0x7FFF;
    nn.d3 &= 0x7FFF;

// for the subtraction we don't need to evaluate any borrow: if any component is >0, then we won't have a factor anyway
    tmp  = q.d1 - nn.d1;
    tmp |= q.d2 - nn.d2;
    tmp |= q.d3 - nn.d3;
    tmp |= q.d4 - nn.d4;

#if (TRACE_KERNEL > 1)
     printf((__constant char *)"mod_simple_e_75_a#3: tid=%u, tmp=%u, q=%x:%x:%x:%x:%x, nn=%x:%x:%x:%x:%x\n",
        get_global_id(0), V(tmp), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif
#if (VECTOR_SIZE == 1)
    if(tmp == 0)
    {
      int index;
      index =ATOMIC_INC(RES[0]);
      if(index < 10)                              /* limit to 10 factors per class */
      {
        RES[index*3 + 1]=n.d4;
        RES[index*3 + 2]=mad24(n.d3,0x8000u, n.d2);
        RES[index*3 + 3]=mad24(n.d1,0x8000u, n.d0);
      }
    }
#elif (VECTOR_SIZE == 2)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
#elif (VECTOR_SIZE == 3)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
    EVAL_RES_tmp75(z)
#elif (VECTOR_SIZE == 4)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
    EVAL_RES_tmp75(z)
    EVAL_RES_tmp75(w)
#elif (VECTOR_SIZE == 8)
    EVAL_RES_tmp75(s0)
    EVAL_RES_tmp75(s1)
    EVAL_RES_tmp75(s2)
    EVAL_RES_tmp75(s3)
    EVAL_RES_tmp75(s4)
    EVAL_RES_tmp75(s5)
    EVAL_RES_tmp75(s6)
    EVAL_RES_tmp75(s7)
#elif (VECTOR_SIZE == 16)
    EVAL_RES_tmp75(s0)
    EVAL_RES_tmp75(s1)
    EVAL_RES_tmp75(s2)
    EVAL_RES_tmp75(s3)
    EVAL_RES_tmp75(s4)
    EVAL_RES_tmp75(s5)
    EVAL_RES_tmp75(s6)
    EVAL_RES_tmp75(s7)
    EVAL_RES_tmp75(s8)
    EVAL_RES_tmp75(s9)
    EVAL_RES_tmp75(sa)
    EVAL_RES_tmp75(sb)
    EVAL_RES_tmp75(sc)
    EVAL_RES_tmp75(sd)
    EVAL_RES_tmp75(se)
    EVAL_RES_tmp75(sf)
#endif
  }
}

void mod_simple_even_75_and_check_big_factor75_big(const int75_v q, const int75_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
)
/*
This function is a combination of mod_simple_96(), check_big_factor96() and an additional correction step.
If q mod n == 1 then n is a factor and written into the RES array.
q must be less than 100n!
*/
{
  __private float_v qf;
  __private uint_v qi, tmp;
  __private int75_v nn;

  qf = CONVERT_FLOAT_V(q.d4);
  qf = qf * 1073741824.0f;
  
  qi = CONVERT_UINT_V(qf*nf);

#ifdef CHECKS_MODBASECASE
  if(n.d4 != 0 && n.d4 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(10, 106, qi, 18);
  }
#endif
/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi |= 1;
 
  nn.d0  = mad24(n.d0, qi, 1u);
  nn.d1  = nn.d0 >> 15;
  nn.d0 &= 0x7FFF;

#if (TRACE_KERNEL > 1)
	  if (get_global_id(0)==TRACE_TID) printf((__constant char *)"mod_simple_e_75_a#1: q=%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

#if (VECTOR_SIZE == 1)
  if(q.d0 == nn.d0)
#else
  if(any(q.d0 == nn.d0)) /* the lowest word of the final result would be 1 for at least one of the vector components (only in this case n might be a factor) */
#endif
  { // it would be sufficient to calculate the one component that made the above "any" return true. But it would require a bigger EVAL macro ...

#if (TRACE_KERNEL > 1)
	  printf((__constant char *)"mod_simple_e_75_a#2: q=%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

// nn = n * qi
    nn.d1  = mad24(n.d1, qi, nn.d1);
    nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
    nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
    nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
    nn.d1 &= 0x7FFF;
    nn.d2 &= 0x7FFF;
    nn.d3 &= 0x7FFF;

// for the subtraction we don't need to evaluate any borrow: if any component is >0, then we won't have a factor anyway
    tmp  = q.d1 - nn.d1;
    tmp |= q.d2 - nn.d2;
    tmp |= q.d3 - nn.d3;
    tmp |= q.d4 - nn.d4;

#if (TRACE_KERNEL > 1)
     printf((__constant char *)"mod_simple_e_75_a#3: tid=%u, tmp=%u, q=%x:%x:%x:%x:%x, nn=%x:%x:%x:%x:%x\n",
        get_global_id(0), V(tmp), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif
#if (VECTOR_SIZE == 1)
    if(tmp == 0)
    {
      int index;
      index =ATOMIC_INC(RES[0]);
      if(index < 10)                              /* limit to 10 factors per class */
      {
        RES[index*3 + 1]=n.d4;
        RES[index*3 + 2]=mad24(n.d3,0x8000u, n.d2);
        RES[index*3 + 3]=mad24(n.d1,0x8000u, n.d0);
      }
    }
#elif (VECTOR_SIZE == 2)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
#elif (VECTOR_SIZE == 3)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
    EVAL_RES_tmp75(z)
#elif (VECTOR_SIZE == 4)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
    EVAL_RES_tmp75(z)
    EVAL_RES_tmp75(w)
#elif (VECTOR_SIZE == 8)
    EVAL_RES_tmp75(s0)
    EVAL_RES_tmp75(s1)
    EVAL_RES_tmp75(s2)
    EVAL_RES_tmp75(s3)
    EVAL_RES_tmp75(s4)
    EVAL_RES_tmp75(s5)
    EVAL_RES_tmp75(s6)
    EVAL_RES_tmp75(s7)
#elif (VECTOR_SIZE == 16)
    EVAL_RES_tmp75(s0)
    EVAL_RES_tmp75(s1)
    EVAL_RES_tmp75(s2)
    EVAL_RES_tmp75(s3)
    EVAL_RES_tmp75(s4)
    EVAL_RES_tmp75(s5)
    EVAL_RES_tmp75(s6)
    EVAL_RES_tmp75(s7)
    EVAL_RES_tmp75(s8)
    EVAL_RES_tmp75(s9)
    EVAL_RES_tmp75(sa)
    EVAL_RES_tmp75(sb)
    EVAL_RES_tmp75(sc)
    EVAL_RES_tmp75(sd)
    EVAL_RES_tmp75(se)
    EVAL_RES_tmp75(sf)
#endif
  }
}

void mod_simple_75_and_check_big_factor75(const int75_v q, const int75_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
)
/*
This function is a combination of mod_simple_96(), check_big_factor96() and an additional correction step.
If q mod n == 1 then n is a factor and written into the RES array.
q must be less than 100n!
*/
{
  __private float_v qf;
  __private uint_v qi, tmp;
  __private int75_v nn;

  qf = CONVERT_FLOAT_V(mad24(q.d4, 32768u, q.d3));
  qf = qf * 32768.0f;
  
  qi = CONVERT_UINT_V(qf*nf);

#ifdef CHECKS_MODBASECASE
  if(n.d4 != 0 && n.d4 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(10, 107, qi, 19);
  }
#endif
/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi += ((~qi) ^ q.d0) & 1;
 
  nn.d0  = mad24(n.d0, qi, 1u);
  nn.d1  = nn.d0 >> 15;
  nn.d0 &= 0x7FFF;

#if (VECTOR_SIZE == 1)
  if(q.d0 == nn.d0)
#else
  if(any(q.d0 == nn.d0)) /* the lowest word of the final result would be 1 for at least one of the vector components (only in this case n might be a factor) */
#endif
  { // it would be sufficient to calculate the one component that made the above "any" return true. But it would require a bigger EVAL macro ...

#if (TRACE_KERNEL > 1)
	  printf((__constant char *)"mod_simple_75_a: q=%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

// nn = n * qi
    nn.d1  = mad24(n.d1, qi, nn.d1);
    nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
    nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
    nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
    nn.d1 &= 0x7FFF;
    nn.d2 &= 0x7FFF;
    nn.d3 &= 0x7FFF;

// for the subtraction we don't need to evaluate any borrow: if any component is >0, then we won't have a factor anyway
    tmp  = q.d1 - nn.d1;
    tmp |= q.d2 - nn.d2;
    tmp |= q.d3 - nn.d3;
    tmp |= q.d4 - nn.d4;

#if (TRACE_KERNEL > 3)
    if (any( tmp == 0)) printf((__constant char *)"mod_simple_75_a: tid=%u, tmp=%u, nn=%x:%x:%x:%x:%x\n",
        get_global_id(1), V(tmp), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

#if (VECTOR_SIZE == 1)
    if(tmp == 0)
    {
      int index;
      index =ATOMIC_INC(RES[0]);
      if(index < 10)                              /* limit to 10 factors per class */
      {
        RES[index*3 + 1]=n.d4;
        RES[index*3 + 2]=mad24(n.d3,0x8000u, n.d2);
        RES[index*3 + 3]=mad24(n.d1,0x8000u, n.d0);
      }
    }
#elif (VECTOR_SIZE == 2)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
#elif (VECTOR_SIZE == 3)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
    EVAL_RES_tmp75(z)
#elif (VECTOR_SIZE == 4)
    EVAL_RES_tmp75(x)
    EVAL_RES_tmp75(y)
    EVAL_RES_tmp75(z)
    EVAL_RES_tmp75(w)
#elif (VECTOR_SIZE == 8)
    EVAL_RES_tmp75(s0)
    EVAL_RES_tmp75(s1)
    EVAL_RES_tmp75(s2)
    EVAL_RES_tmp75(s3)
    EVAL_RES_tmp75(s4)
    EVAL_RES_tmp75(s5)
    EVAL_RES_tmp75(s6)
    EVAL_RES_tmp75(s7)
#elif (VECTOR_SIZE == 16)
    EVAL_RES_tmp75(s0)
    EVAL_RES_tmp75(s1)
    EVAL_RES_tmp75(s2)
    EVAL_RES_tmp75(s3)
    EVAL_RES_tmp75(s4)
    EVAL_RES_tmp75(s5)
    EVAL_RES_tmp75(s6)
    EVAL_RES_tmp75(s7)
    EVAL_RES_tmp75(s8)
    EVAL_RES_tmp75(s9)
    EVAL_RES_tmp75(sa)
    EVAL_RES_tmp75(sb)
    EVAL_RES_tmp75(sc)
    EVAL_RES_tmp75(sd)
    EVAL_RES_tmp75(se)
    EVAL_RES_tmp75(sf)
#endif
  }
}


void calculate_FC75(const uint exponent, const uint tid, const __global uint * restrict k_tab, const int75_t k_base, __private int75_v * restrict const f)
{
  __private int75_t exp75;
  __private uint_v t, t1;
  __private int75_v k;


  // exp75.d4=0;exp75.d3=0;  // not used, PERF: we can skip d2 as well, if we limit exponent to 2^29
  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"calculate_FC75: exp=%d, x2=%x:%x:%x, k_base=%x:%x:%x:%x:%x\n",
        exponent, exp75.d2, exp75.d1, exp75.d0, k_base.d4, k_base.d3, k_base.d2, k_base.d1, k_base.d0);
#endif

#if (VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
#elif (VECTOR_SIZE == 3)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
#elif (VECTOR_SIZE == 4)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
  t.w  = k_tab[tid+3];
#elif (VECTOR_SIZE == 8)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
#elif (VECTOR_SIZE == 16)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
  t.s8 = k_tab[tid+8];
  t.s9 = k_tab[tid+9];
  t.sa = k_tab[tid+10];
  t.sb = k_tab[tid+11];
  t.sc = k_tab[tid+12];
  t.sd = k_tab[tid+13];
  t.se = k_tab[tid+14];
  t.sf = k_tab[tid+15];
#endif
  t1 = t >> 15;  // t is 24 bits at most
  t  = t & 0x7FFF;

  k.d0  = mad24(t , 4620u, k_base.d0);
  k.d1  = mad24(t1, 4620u, k_base.d1) + (k.d0 >> 15);
  k.d0 &= 0x7FFF;
  k.d2  = (k.d1 >> 15) + k_base.d2;
  k.d1 &= 0x7FFF;
  k.d3  = (k.d2 >> 15) + k_base.d3;
  k.d2 &= 0x7FFF;
  k.d4  = (k.d3 >> 15) + k_base.d4;  // PERF: k.d4 = 0, normally. Can we limit k to 2^60?
  k.d3 &= 0x7FFF;
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"calculate_FC75: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x:%x:%x\n",
        tid, V(t), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0));
#endif
		// f = 2 * k * exp + 1
  f->d0 = mad24(k.d0, exp75.d0, 1u);

  f->d1 = mad24(k.d1, exp75.d0, f->d0 >> 15);
  f->d1 = mad24(k.d0, exp75.d1, f->d1);
  f->d0 &= 0x7FFF;

  f->d2 = mad24(k.d2, exp75.d0, f->d1 >> 15);
  f->d2 = mad24(k.d1, exp75.d1, f->d2);
  f->d2 = mad24(k.d0, exp75.d2, f->d2);  // PERF: if we limit exp at kernel compile time to 2^29, then we can skip exp75.d2 here and above.
  f->d1 &= 0x7FFF;

  f->d3 = mad24(k.d3, exp75.d0, f->d2 >> 15);
  f->d3 = mad24(k.d2, exp75.d1, f->d3);
  f->d3 = mad24(k.d1, exp75.d2, f->d3);
//  f->d3 = mad24(k.d0, exp75.d3, f->d3);    // exp75.d3 = 0
  f->d2 &= 0x7FFF;

  f->d4 = mad24(k.d4, exp75.d0, f->d3 >> 15);  // PERF: see above
  f->d4 = mad24(k.d3, exp75.d1, f->d4);
  f->d4 = mad24(k.d2, exp75.d2, f->d4);
  f->d3 &= 0x7FFF;
}


void mod_simple_90(int90_v * const res, const int90_v q, const int90_v n, const float_v nf
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
assumes q < 12n (12n includes "optional mul 2")
*/
{
  __private float_v qf;
  __private uint_v qi;
  __private int90_v nn;

  qf = CONVERT_FLOAT_V(mad24(q.d5, 32768u, q.d4));  // q.d3 needed?
  qf = qf * 1073741824.0f;
  
  qi = CONVERT_UINT_V(qf*nf);

#ifdef CHECKS_MODBASECASE
/* both barrett based kernels are made for factor candidates above 2^64,
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
  if(n.d5 != 0 && n.d5 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 101, qi, 12);
  }
#endif
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"mod_simple_90#1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d5), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

// nn = n * qi
  nn.d0  = mul24(n.d0, qi);
  nn.d1  = mad24(n.d1, qi, nn.d0 >> 15);
  nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
  nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
  nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
  nn.d5  = mad24(n.d5, qi, nn.d4 >> 15);
  nn.d0 &= 0x7FFF;
  nn.d1 &= 0x7FFF;
  nn.d2 &= 0x7FFF;
  nn.d3 &= 0x7FFF;
  nn.d4 &= 0x7FFF;

  res->d0 = q.d0 - nn.d0;
  res->d1 = q.d1 - nn.d1 + AS_UINT_V(res->d0 > 0x7FFF);
  res->d2 = q.d2 - nn.d2 + AS_UINT_V(res->d1 > 0x7FFF);
  res->d3 = q.d3 - nn.d3 + AS_UINT_V(res->d2 > 0x7FFF);
  res->d4 = q.d4 - nn.d4 + AS_UINT_V(res->d3 > 0x7FFF);
  res->d5 = q.d5 - nn.d5 + AS_UINT_V(res->d4 > 0x7FFF);
  res->d0 &= 0x7FFF;
  res->d1 &= 0x7FFF;
  res->d2 &= 0x7FFF;
  res->d3 &= 0x7FFF;
  res->d4 &= 0x7FFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_simple_90#2: nn=%x:%x:%x:%x:%x:%x, res=%x:%x:%x:%x:%x:%x\n",
        V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d5), V(res->d4), V(res->d3), V(res->d2), V(res->d1), V(res->d0));
#endif
}

void mod_simple_even_90_and_check_big_factor90(const int90_v q, const int90_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
)
/*
This function is a combination of mod_simple_90(), check_big_factor90() and an additional correction step.
If q mod n == 1 then n is a factor and written into the RES array.
q must be less than 100n!
*/
{
  __private float_v qf;
  __private uint_v qi, tmp;
  __private int90_v nn;

  qf = CONVERT_FLOAT_V(mad24(q.d5, 32768u, q.d4));  // q.d3 needed?
  qf = qf * 1073741824.0f;
  
  qi = CONVERT_UINT_V(qf*nf);
#ifdef CHECKS_MODBASECASE
  if(n.d5 != 0 && n.d5 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(10, 108, qi, 20);
  }
#endif

/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi |= 1;
 
  nn.d0  = mad24(n.d0, qi, 1u);
  nn.d1  = nn.d0 >> 15;
  nn.d0 &= 0x7FFF;
#if (TRACE_KERNEL > 1)
	  if (get_global_id(0)==TRACE_TID) printf((__constant char *)"mod_simple_e_90_a#1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d5), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

#if (VECTOR_SIZE == 1)
  if(q.d0 == nn.d0)
#else
  if(any(q.d0 == nn.d0)) /* the lowest word of the final result would be 1 for at least one of the vector components (only in this case n might be a factor) */
#endif
  { // it would be sufficient to calculate the one component that made the above "any" return true. But it would require a bigger EVAL macro ...

#if (TRACE_KERNEL > 1)
	  printf((__constant char *)"mod_simple_e_90_a#2: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d5), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

// nn = n * qi
    nn.d1  = mad24(n.d1, qi, nn.d1);
    nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
    nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
    nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
    nn.d5  = mad24(n.d5, qi, nn.d4 >> 15);
    nn.d1 &= 0x7FFF;
    nn.d2 &= 0x7FFF;
    nn.d3 &= 0x7FFF;
    nn.d4 &= 0x7FFF;

// for the subtraction we don't need to evaluate any borrow: if any component is >0, then we won't have a factor anyway
    tmp  = q.d1 - nn.d1;
    tmp |= q.d2 - nn.d2;
    tmp |= q.d3 - nn.d3;
    tmp |= q.d4 - nn.d4;
    tmp |= q.d5 - nn.d5;

#if (TRACE_KERNEL > 1)
    if (any( tmp == 0)) printf((__constant char *)"mod_simple_e_90_a#3: tid=%u, tmp=%u, nn=%x:%x:%x:%x:%x:%x\n",
        get_global_id(0), V(tmp), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

#if (VECTOR_SIZE == 1)
    if(tmp == 0)
    {
      int index;
      index =ATOMIC_INC(RES[0]);
      if(index < 10)                              /* limit to 10 factors per class */
      {
        RES[index*3 + 1]=mad24(n.d5,0x8000u, n.d4);
        RES[index*3 + 2]=mad24(n.d3,0x8000u, n.d2);
        RES[index*3 + 3]=mad24(n.d1,0x8000u, n.d0);
      }
    }
#elif (VECTOR_SIZE == 2)
    EVAL_RES_tmp90(x)
    EVAL_RES_tmp90(y)
#elif (VECTOR_SIZE == 3)
    EVAL_RES_tmp90(x)
    EVAL_RES_tmp90(y)
    EVAL_RES_tmp90(z)
#elif (VECTOR_SIZE == 4)
    EVAL_RES_tmp90(x)
    EVAL_RES_tmp90(y)
    EVAL_RES_tmp90(z)
    EVAL_RES_tmp90(w)
#elif (VECTOR_SIZE == 8)
    EVAL_RES_tmp90(s0)
    EVAL_RES_tmp90(s1)
    EVAL_RES_tmp90(s2)
    EVAL_RES_tmp90(s3)
    EVAL_RES_tmp90(s4)
    EVAL_RES_tmp90(s5)
    EVAL_RES_tmp90(s6)
    EVAL_RES_tmp90(s7)
#elif (VECTOR_SIZE == 16)
    EVAL_RES_tmp90(s0)
    EVAL_RES_tmp90(s1)
    EVAL_RES_tmp90(s2)
    EVAL_RES_tmp90(s3)
    EVAL_RES_tmp90(s4)
    EVAL_RES_tmp90(s5)
    EVAL_RES_tmp90(s6)
    EVAL_RES_tmp90(s7)
    EVAL_RES_tmp90(s8)
    EVAL_RES_tmp90(s9)
    EVAL_RES_tmp90(sa)
    EVAL_RES_tmp90(sb)
    EVAL_RES_tmp90(sc)
    EVAL_RES_tmp90(sd)
    EVAL_RES_tmp90(se)
    EVAL_RES_tmp90(sf)
#endif
  }
}

void mod_simple_90_and_check_big_factor90(const int90_v q, const int90_v n, const float_v nf, __global uint * const RES
#ifdef CHECKS_MODBASECASE
                  , const int bit_max64, const uint limit, __global uint * restrict modbasecase_debug
#endif
)
/*
This function is a combination of mod_simple_90(), check_big_factor90() and an additional correction step.
If q mod n == 1 then n is a factor and written into the RES array.
q must be less than 100n!
*/
{
  __private float_v qf;
  __private uint_v qi, tmp;
  __private int90_v nn;

  qf = CONVERT_FLOAT_V(mad24(q.d5, 32768u, q.d4));  // q.d3 needed?
  qf = qf * 1073741824.0f;
  
  qi = CONVERT_UINT_V(qf*nf);
#ifdef CHECKS_MODBASECASE
  if(n.d5 != 0 && n.d5 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(10, 109, qi, 21);
  }
#endif

/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi += ((~qi) ^ q.d0) & 1;
 
  nn.d0  = mad24(n.d0, qi, 1u);
  nn.d1  = nn.d0 >> 15;
  nn.d0 &= 0x7FFF;

#if (TRACE_KERNEL > 1)
	  if(get_global_id(0)==TRACE_TID) printf((__constant char *)"mod_simple_90_a: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d5), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif
#if (VECTOR_SIZE == 1)
  if(q.d0 == nn.d0)
#else
  if(any(q.d0 == nn.d0)) /* the lowest word of the final result would be 1 for at least one of the vector components (only in this case n might be a factor) */
#endif
  { // it would be sufficient to calculate the one component that made the above "any" return true. But it would require a bigger EVAL macro ...

#if (TRACE_KERNEL > 1)
	  printf((__constant char *)"mod_simple_90_a: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x:%x, qi=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d5), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(qi));
#endif

// nn = n * qi
    nn.d1  = mad24(n.d1, qi, nn.d1);
    nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
    nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
    nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
    nn.d5  = mad24(n.d5, qi, nn.d4 >> 15);
    nn.d1 &= 0x7FFF;
    nn.d2 &= 0x7FFF;
    nn.d3 &= 0x7FFF;
    nn.d4 &= 0x7FFF;

// for the subtraction we don't need to evaluate any borrow: if any component is >0, then we won't have a factor anyway
    tmp  = q.d1 - nn.d1;
    tmp |= q.d2 - nn.d2;
    tmp |= q.d3 - nn.d3;
    tmp |= q.d4 - nn.d4;
    tmp |= q.d5 - nn.d5;

#if (TRACE_KERNEL > 3)
    if (any( tmp == 0)) printf((__constant char *)"mod_simple_90_a: tid=%u, tmp=%u, nn=%x:%x:%x:%x:%x:%x\n",
        get_global_id(1), V(tmp), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

#if (VECTOR_SIZE == 1)
    if(tmp == 0)
    {
      int index;
      index =ATOMIC_INC(RES[0]);
      if(index < 10)                              /* limit to 10 factors per class */
      {
        RES[index*3 + 1]=mad24(n.d5,0x8000u, n.d4);
        RES[index*3 + 2]=mad24(n.d3,0x8000u, n.d2);
        RES[index*3 + 3]=mad24(n.d1,0x8000u, n.d0);
      }
    }
#elif (VECTOR_SIZE == 2)
    EVAL_RES_tmp90(x)
    EVAL_RES_tmp90(y)
#elif (VECTOR_SIZE == 3)
    EVAL_RES_tmp90(x)
    EVAL_RES_tmp90(y)
    EVAL_RES_tmp90(z)
#elif (VECTOR_SIZE == 4)
    EVAL_RES_tmp90(x)
    EVAL_RES_tmp90(y)
    EVAL_RES_tmp90(z)
    EVAL_RES_tmp90(w)
#elif (VECTOR_SIZE == 8)
    EVAL_RES_tmp90(s0)
    EVAL_RES_tmp90(s1)
    EVAL_RES_tmp90(s2)
    EVAL_RES_tmp90(s3)
    EVAL_RES_tmp90(s4)
    EVAL_RES_tmp90(s5)
    EVAL_RES_tmp90(s6)
    EVAL_RES_tmp90(s7)
#elif (VECTOR_SIZE == 16)
    EVAL_RES_tmp90(s0)
    EVAL_RES_tmp90(s1)
    EVAL_RES_tmp90(s2)
    EVAL_RES_tmp90(s3)
    EVAL_RES_tmp90(s4)
    EVAL_RES_tmp90(s5)
    EVAL_RES_tmp90(s6)
    EVAL_RES_tmp90(s7)
    EVAL_RES_tmp90(s8)
    EVAL_RES_tmp90(s9)
    EVAL_RES_tmp90(sa)
    EVAL_RES_tmp90(sb)
    EVAL_RES_tmp90(sc)
    EVAL_RES_tmp90(sd)
    EVAL_RES_tmp90(se)
    EVAL_RES_tmp90(sf)
#endif
  }
}

void calculate_FC90(const uint exponent, const uint tid, const __global uint * restrict k_tab, const int75_t k_base, __private int90_v * restrict const f)
{
  __private int90_t exp90;
  __private uint_v t, t1;
  __private int90_v k;


  // exp90.d4=0;exp90.d3=0;  // not used, PERF: we can skip d2 as well, if we limit exp to 2^29
  exp90.d2=exponent>>29;exp90.d1=(exponent>>14)&0x7FFF;exp90.d0=(exponent<<1)&0x7FFF;	// exp90 = 2 * exponent

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"calculate_FC90: exp=%d, x2=%x:%x:%x, k_base=%x:%x:%x:%x:%x\n",
        exponent, exp90.d2, exp90.d1, exp90.d0, k_base.d4, k_base.d3, k_base.d2, k_base.d1, k_base.d0);
#endif

#if (VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
#elif (VECTOR_SIZE == 3)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
#elif (VECTOR_SIZE == 4)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
  t.w  = k_tab[tid+3];
#elif (VECTOR_SIZE == 8)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
#elif (VECTOR_SIZE == 16)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
  t.s8 = k_tab[tid+8];
  t.s9 = k_tab[tid+9];
  t.sa = k_tab[tid+10];
  t.sb = k_tab[tid+11];
  t.sc = k_tab[tid+12];
  t.sd = k_tab[tid+13];
  t.se = k_tab[tid+14];
  t.sf = k_tab[tid+15];
#endif
  t1 = t >> 15;  // t is 24 bits at most
  t  = t & 0x7FFF;

  k.d0  = mad24(t , 4620u, k_base.d0);
  k.d1  = mad24(t1, 4620u, k_base.d1) + (k.d0 >> 15);
  k.d0 &= 0x7FFF;
  k.d2  = (k.d1 >> 15) + k_base.d2;
  k.d1 &= 0x7FFF;
  k.d3  = (k.d2 >> 15) + k_base.d3;
  k.d2 &= 0x7FFF;
  k.d4  = (k.d3 >> 15) + k_base.d4;  // PERF: k.d4 = 0, normally. Can we limit k to 2^60?
  k.d3 &= 0x7FFF;
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"calculate_FC90: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x:%x:%x\n",
        tid, V(t), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0));
#endif
		// f = 2 * k * exp + 1
  f->d0 = mad24(k.d0, exp90.d0, 1u);

  f->d1 = mad24(k.d1, exp90.d0, f->d0 >> 15);
  f->d1 = mad24(k.d0, exp90.d1, f->d1);
  f->d0 &= 0x7FFF;

  f->d2 = mad24(k.d2, exp90.d0, f->d1 >> 15);
  f->d2 = mad24(k.d1, exp90.d1, f->d2);
  f->d2 = mad24(k.d0, exp90.d2, f->d2);  // PERF: if we limit exp at kernel compile time to 2^29, then we can skip exp90.d2 here and above.
  f->d1 &= 0x7FFF;

  f->d3 = mad24(k.d3, exp90.d0, f->d2 >> 15);
  f->d3 = mad24(k.d2, exp90.d1, f->d3);
  f->d3 = mad24(k.d1, exp90.d2, f->d3);
//  f->d3 = mad24(k.d0, exp90.d3, f->d3);    // exp90.d3 = 0
  f->d2 &= 0x7FFF;

  f->d4 = mad24(k.d4, exp90.d0, f->d3 >> 15);  // PERF: see above
  f->d4 = mad24(k.d3, exp90.d1, f->d4);
  f->d4 = mad24(k.d2, exp90.d2, f->d4);
  f->d3 &= 0x7FFF;

//  f->d5 = mad24(k.d5, exp90.d0, f->d4 >> 15);  // k.d5 = 0
  f->d5 = mad24(k.d4, exp90.d1, f->d4 >> 15);
  f->d5 = mad24(k.d3, exp90.d2, f->d5);
  f->d4 &= 0x7FFF;
}