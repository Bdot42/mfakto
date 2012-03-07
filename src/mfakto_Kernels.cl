/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2011  Oliver Weihe (o.weihe@t-online.de)
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

// In Catalyst 11.10 and 11.11, not all parameters were passed to the kernel
// -> replace user-defined struct with uint8
#define WA_FOR_CATALYST11_10_BUG

// TRACE_KERNEL: higher is more trace, 0-5 currently used
#define TRACE_KERNEL 0

// If above tracing is on, only the thread with the ID below will trace
#define TRACE_TID 0

// defines how many factor candidates the barrett kernels will process in parallel per thread
// this is now defined via commandline to the OpenCL compiler
//#define BARRETT_VECTOR_SIZE 4

/***********************************
 * DONT CHANGE ANYTHING BELOW THIS *
 ***********************************/

#if (TRACE_KERNEL > 0) || defined (CHECKS_MODBASECASE)
// available on all platforms so far ...
#pragma  OPENCL EXTENSION cl_amd_printf : enable
//#pragma  OPENCL EXTENSION cl_khr_fp64 : enable
#endif


// HD4xxx does not have atomics, but mfakto will work on these GPUs as well.
// Without atomics, the factors found may be scrambled when more than one
// factor is found per grid => if the reported factor(s) are not accepted
// by primenet, then run the bitlevel again with the smallest possible grid size,
// or run it on at least HD5...

#ifdef cl_khr_global_int32_base_atomics 
#pragma  OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMIC_INC(x) atom_inc(&x)
#else
// No atomic operations available - using simple ++
#define ATOMIC_INC(x) ((x)++)
#endif

/* 96bit (3x 32bit) integer
D= d0 + d1*(2^32) + d2*(2^64) */
typedef struct _int96_1t
{
  uint d0,d1,d2;
}int96_1t;

/* 192bit (6x 32bit) integer
D=d0 + d1*(2^32) + d2*(2^64) + ... */
typedef struct _int192_1t
{
  uint d0,d1,d2,d3,d4,d5;
}int192_1t;

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct _int72_t
{
  uint d0,d1,d2;
}int72_t;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct _int144_t
{
  uint d0,d1,d2,d3,d4,d5;
}int144_t;


#ifdef CL_GPU_SIEVE
#include "sieve.cl"
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

#include "barrett.cl"   // one kernel file for 32-bit-barrett of different vector sizes (1, 2, 4, 8, 16)
#define EVAL_RES(x) EVAL_RES_b(x)  // no check for f==1 if running the "big" version
#include "mul24.cl" // one kernel file for 24-bit-kernels of different vector sizes (1, 2, 4, 8, 16)

#define _63BIT_MUL24_K
#undef EVAL_RES
#define EVAL_RES(x) EVAL_RES_a(x)
#include "mul24.cl" // include again, now for small factors < 64 bit


// this kernel is only used for a quick test at startup - no need to be correct ;-)
// currently this kernel is used for testing what happens without atomics when multiple factors are found
__kernel void mod_128_64_k(const ulong hi, const ulong lo, const ulong q,
                           const float qr, __global uint *res
)
{
  __private uint i,f;

  f = get_global_id(0);

  f++; // let the reported results start with 1

//  barrier(CLK_GLOBAL_MEM_FENCE);
#if (TRACE_KERNEL > 1)
    printf("kernel tracing level %d enabled\n", TRACE_KERNEL);
#endif

  if (1 == 1)
  {
    i=ATOMIC_INC(res[0]);

#pragma  OPENCL EXTENSION cl_amd_printf : enable
    printf("thread %d: i=%d, res[0]=%d\n", get_global_id(0), i, res[0]);

    if(i<10)				/* limit to 10 results */
    {
      res[i*3 + 1]=f;
      res[i*3 + 2]=f;
      res[i*3 + 3]=f;
    }
  }

}
