/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)
                                Bertram Franz (bertramf@gmx.net)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
                                
You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
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

// unfortunately required until Catalyst 11.6 (fixed in 11.7)
#undef OpenCL_CRASH_BUG_WA

// TRACE_KERNEL: higher is more trace, 0-5 currently used
//#define TRACE_KERNEL 4

// If above tracing is on, only the thread with the ID below will trace
#define TRACE_TID 5

// defines how many factor candidates the barrett kernels will process in parallel per thread
//#define BARRETT_VECTOR_SIZE 4

// HD4xxx does not have atomics, undefine this to make mfakto work on those.
// without atomics, the factors found may be scrambled when more than one
// factor is found per grid => if the reported factor(s) are not accepted
// by primenet, then run the bitlevel again with the smallest possible grid size,
// or run it on at least HD5...
#define ATOMICS


/***********************************
 * DONT CHANGE ANYTHING BELOW THIS *
 ***********************************/

#if (TRACE_KERNEL > 0) || defined (CHECKS_MODBASECASE)
// available on all platforms so far ...
#pragma  OPENCL EXTENSION cl_amd_printf : enable
//#pragma  OPENCL EXTENSION cl_khr_fp64 : enable
#endif


#ifdef cl_khr_global_int32_base_atomics 
#pragma  OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMIC_INC(x) atom_inc(&x)
#else
#pragma warning "No atomic operations available - using simple ++"
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

#ifdef BARRETT_VECTOR_SIZE
#include "barrett.cl"   // one kernel file for different vector sizes (1, 2, 4, 8, 16)
#else
/****************************************
 ****************************************
 * 32-bit-stuff for the 92/96-bit-kernel
 * Fallback to old no-vector-implementation ... 
 ****************************************
 ****************************************/
/* 96bit (3x 32bit) integer
D= d0 + d1*(2^32) + d2*(2^64) */
typedef struct _int96_t
{
  uint d0,d1,d2;
}int96_t;

/* 192bit (6x 32bit) integer
D=d0 + d1*(2^32) + d2*(2^64) + ... */
typedef struct _int192_t
{
  uint d0,d1,d2,d3,d4,d5;
}int192_t;

void div_192_96(int96_1t *res, int192_1t q, int96_1t n, float nf);
void div_160_96(int96_1t *res, int192_1t q, int96_1t n, float nf);
void mul_96(int96_1t *res, int96_1t a, int96_1t b);
void mul_96_192_no_low2(int192_1t *res, int96_1t a, int96_1t b);
void mul_96_192_no_low3(int192_1t *res, int96_1t a, int96_1t b);
#endif  // Barrett

/****************************************
 ****************************************
 * 24-bit-stuff for the 71-bit-kernel
 *
 ****************************************
 ****************************************/

#define EVAL_RES(comp) \
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


/****************************************
 ****************************************
 * 64-bit-stuff for the 64-bit-kernel
 *
 ****************************************
 ****************************************/

void square_64_128(ulong *res_hi, ulong *res_lo, const ulong in
#if (TRACE_KERNEL > 1)
                   , __private uint tid
#endif
)
{
  *res_hi = mul_hi(in, in);
  *res_lo = in * in;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("square_64_128: %llx ^ 2 = %llx : %llx\n", in, *res_hi, *res_lo);
#endif
}

int gte_128(ulong v1_hi, ulong v1_lo, ulong v2_hi, ulong v2_lo)
{
  if (v1_hi == v2_hi)
    return (v1_lo >= v2_lo);
  return (v1_hi >= v2_hi);
}

void sub_128(ulong *v1_hi, ulong *v1_lo, ulong v2_hi, ulong v2_lo
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("sub_128:   %llx:%llx - %llx:%llx = ", *v1_hi, *v1_lo, v2_hi, v2_lo);
#endif
  *v1_hi = *v1_hi - v2_hi - ((*v1_lo < v2_lo) ? 1 : 0);
  *v1_lo = *v1_lo - v2_lo;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("%llx:%llx\n", *v1_hi, *v1_lo);
#endif
}

void sub_if_gte_128(ulong *v1_hi, ulong *v1_lo, ulong v2_hi, ulong v2_lo
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{ /* if (v1 >= v2) v1=v1-v2 */
  ulong tmp_hi, tmp_lo;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("sub_if_gte_128:   %llx:%llx - %llx:%llx = ", *v1_hi, *v1_lo, v2_hi, v2_lo);
#endif
  tmp_lo = *v1_lo - v2_lo;
  tmp_hi = *v1_hi - v2_hi - ((*v1_lo < v2_lo) ? 1 : 0);


  *v1_hi = (tmp_hi > *v1_hi) ? *v1_hi : tmp_hi;
  *v1_lo = (tmp_hi > *v1_hi) ? *v1_lo : tmp_lo;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("%llx:%llx\n", *v1_hi, *v1_lo);
#endif
}

void square_96_192_64(ulong *res_hi, ulong *res_mid, ulong *res_lo, const ulong in_hi, const ulong in_lo
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{
  __private ulong tmp1, tmp2;   // (in_hi + in_lo) ^2 = in_hi^2 + 2*in_hi*in_lo + in_lo^2
  // PERF: better when using private copies for *res*?
  // PERF: better using 32-bit parts? Or 24-bit?
  *res_lo  = in_lo * in_lo;
  tmp1     = in_lo * in_hi;
  tmp2     = mul_hi(in_lo, in_lo);
  *res_mid = tmp1 << 1;
  *res_hi  = in_hi * in_hi +
             ((*res_mid < tmp1) ? 1 : 0) +   // "carry" from previous left-shift
             (mul_hi(in_lo, in_hi) << 1);    // shift cannot overflow as in_hi uses only 32 of 64 bit.
  *res_mid = *res_mid + tmp2;
  *res_hi  = *res_hi + ((*res_mid < tmp2) ? 1 : 0); // "carry" from above
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("square_96_192: %llx : %llx ^ 2 = %llx : %llx : %llx\n", in_hi, in_lo, *res_hi, *res_mid, *res_lo);
#endif
 }

// modulo by division
ulong mod_128_64_d(__private ulong hi, __private ulong lo, const ulong q, const uint lshift
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{
  // some day I'll implement a fast 64-bit and a fast 96-bit kernel
  return 0;
}

// modulo by shift - cmp - sub
ulong mod_128_64_s(__private ulong hi, __private ulong lo, const ulong q, const uint lshift
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{
  __private int i = clz(q) - clz(hi); // hi is i bitpositions larger than q;  
  __const ulong mask= 0x8000000000000000 ;  // first bit of ulong

#if (TRACE_KERNEL > 2)
  if (q&mask)
  {
    if (tid==TRACE_TID) printf("ERROR: q >= 2^63: %llx (mask=%llx)\n", q, mask);
  }
  if (tid==TRACE_TID) printf("mod_128_64_s: i=%d: hi=%llx, lo=%llx, q=%llx, mask=%llx, shift=%u\n", i, hi, lo, q, mask, lshift);
#endif

#ifdef BETTER_BE_SAFE_THAN_SORRY
  __private ulong a  = q << ( (i>0) ? i : 0);  // a = q shifted to ~same magnitude as hi
  for ( ; i>0 ; i--)  
  {
    hi = hi - ( (hi>a) ? a : 0 );  // subtract multiples of q
    a = a >> 1;                    // slowly shift back until we have q again
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_128_64_s: i=%d: hi= %llx, a=%llx\n", i, hi, a);
#endif
  }
#endif

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("mod_128_64_s: hi= %llx, lo=%llx, q=%llx\n", hi, lo, q);
#endif

 for (i=0; i<64; i++)  //process the 64 bits of lo.  PERF: unroll loop later
  {
    hi = (hi << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster?
    lo = lo << 1;
    hi = hi - ( (hi>q) ? q : 0 );  // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_128_64_s: i=%d: hi= %llx, lo=%llx, a=%llx\n", i, hi, lo, q);
#endif
  }
  hi = hi << lshift;
  hi = hi - ( (hi>q) ? q : 0 );  // subtract q

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mod_128_64_s: return %llx\n", hi);
#endif
  return hi;
}

// modulo by shift - cmp - sub
void mod_192_96_s(__private ulong hi, __private ulong mid, __private ulong lo,
                  __private ulong q_hi, __private ulong q_lo,
                  __private uint lshift,
                  __private ulong *r_hi, __private ulong *r_lo
#if (TRACE_KERNEL > 1)
                  , __private uint tid
#endif
)
{
  __private long i = clz(q_hi) - clz(hi);  // hi is i bitpositions larger than q (q at least 2^63)
  const ulong mask= 0x8000000000000000 ;  // first bit of ulong
  __private ulong a_hi, a_lo;

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: %llx:%llx:%llx mod %llx:%llx, shift=%u\n", i, hi, mid, lo, q_hi, q_lo, lshift);
#endif

  if (i>0)
  {
    a_hi = (q_hi << i) | (q_lo >> (64-i));  // a = q shifted to ~same magnitude as hi
    a_lo = q_lo << i;
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%lld: hi= %llx:%llx, a=%llx:%llx\n", i, hi, mid, a_hi, a_lo);
#endif
  }
  for ( ; i>0 ; i--)  
  {
      sub_if_gte_128(&hi, &mid, a_hi, a_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );     // subtract multiples of q
      a_lo = (a_lo >> 1) | (a_hi << 63);    // slowly shift back until we have q again
      a_hi = a_hi >> 1;
#if (TRACE_KERNEL > 2)
      if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx:%llx, a=%llx:%llx\n", i, hi, mid, a_hi, a_lo);
#endif
  }
  sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
      if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx:%llx, a=%llx:%llx\n", i, hi, mid, q_hi, q_lo);
#endif

//#pragma unroll 2
  for (i=0; i<64; i++)  //process the 64 bits of lo.
  {
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
  }
  if (lshift)
  {
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1);
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
  }
  *r_hi = hi;
  *r_lo = mid;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mod_192_96_s: return %llx:%llx\n", hi, mid);
#endif
}

__kernel void mfakto_cl_64(__private uint exp, __private ulong k_base, __global uint *k_tab, __private ulong4 b_pre_shift, __private int bit_max64, __global uint *RES)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  // __private long shiftcount = b_pre_shift.w;  // how many bits of exp are left to be processed
  __private ulong pp, hi, lo, k, q, r;
	__private uint tid, mask;
//	__private float qr; /* a little less  than 1/q */

	tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  pp = exp;             // 32 -> 64 bit
  k = k_tab[tid];       // 32 -> 64 bit
  k = k*4620 + k_base;  // NUM_CLASSES
	q = (pp<<1) * k + 1;  // q = 2*k*exp+1
	                      // the first bits of exp are processed on the host w/o modulo,
                        // as the result of the squaring was less than the FC anyway.
  /* now only to bit_max^2 of the kernel 
                        // preprocessing is now done as long as it fits into 192 bits w/o modulo

  mod_192_96_s(b_pre_shift.z, b_pre_shift.y, b_pre_shift.x, q, 0, 0, &hi, &lo   // 192 bit with q < 2^64 is more than the mod can handle
#if (TRACE_KERNEL > 1)                              // Do a mod (q<<64) first to bring b down to 128 bits
                   , tid
#endif
              );  // initial modulo of the precomputed residue  */

  r = mod_128_64_s(b_pre_shift.y, b_pre_shift.x, q, 0
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                  ); // r = hi:lo  % q
 // and now again the real modulo

	mask = 1<<(b_pre_shift.w); /* the 1 in mask now points to the bit-pos after the first modulo was necessary */
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_64: tid=%ld: p=%llx, k=%llx, q=%llx, mask=%llx, r=%llx\n", tid, pp, k, q, mask, r);
#endif
	while (mask)
	{
	  square_64_128(&hi, &lo, r
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 ); /*hi:lo = (r * r); */
    r = mod_128_64_s(hi, lo, q, ( pp&mask ) ? 1 : 0
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                    ); // r = hi:lo << 0 or 1   % q
	  mask = mask >> 1;   // next bit of p
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_64: q=%llx, mask=%llx, r=%llx\n", q, mask, r);
#endif
	}
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakto_cl_64: tid=%ld: q=%llx, k=%llx, r=%llx\n", tid, q, k, r);
#endif

/* finally check if we found a factor and write the factor to RES[] */
  if(r==1)
  {
#if (TRACE_KERNEL > 0)  // will trace for any thread
      printf("mfakto_cl_64: tid=%ld found factor: q=%llx, k=%llx, r=%llx\n", tid, q, k, r);
#endif
      tid=ATOMIC_INC(RES[0]);
      if(tid<10)				/* limit to 10 factors per class */
      {
        RES[tid*3 + 1]= 0;
        RES[tid*3 + 2]= (uint) (q >> 32);
        RES[tid*3 + 3]= (uint) q & 0xFFFFFFFF;
      }
  }
}

// this kernel is only used for a quick test at startup - no need to be correct ;-)
// currently this kernel is used for testing what happens without atomics when multiple factors are found
__kernel void mod_128_64_k(const ulong hi, const ulong lo, const ulong q,
                           const float qr, __global uint *res
#if (TRACE_KERNEL > 1)
                         , __private uint tid
#endif
)
{
  __private uint i,f;

  f = get_global_id(0);

  f++; // let the reported results start with 1

//  barrier(CLK_GLOBAL_MEM_FENCE);

  if (1 == 1)
  {
    i=ATOMIC_INC(res[0]);

//#pragma  OPENCL EXTENSION cl_amd_printf : enable
//    printf("thread %d: i=%d, res[0]=%d\n", get_global_id(0), i, res[0]);

    if(i<10)				/* limit to 10 results */
    {
      res[i*3 + 1]=f;
      res[i*3 + 2]=f;
      res[i*3 + 3]=f;
    }
  }

}

__kernel void mfakto_cl_95(__private uint exp, __private ulong k_base,
                           __global uint *k_tab, __private ulong4 b_pre_shift,
                           __private int bit_max64, __global uint *RES)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  // __private long shiftcount = b_pre_shift.w;  // how many bits of exp are left to be processed
  __private ulong q_lo, q_hi, r_lo, r_hi, pp, hi, mid, lo, k;
	__private uint tid, mask;

	tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  pp = exp;             // 32 -> 64 bit
  k = k_tab[tid];       // 32 -> 64 bit
  k = k*4620 + k_base;  // NUM_CLASSES
	q_lo = pp<<1;
  q_hi = mul_hi(q_lo, k);
  q_lo = q_lo * k + 1;  // q = 2*k*exp+1
	lo  = b_pre_shift.x;  // the first bits of exp are processed on the host w/o modulo,
  mid = b_pre_shift.y;  // as the result of the squaring was less than the FC anyway.
  hi  = b_pre_shift.z;  // preprocessing is done as long as it fits into 192 bits w/o modulo

  mod_192_96_s(hi, mid, lo, q_hi, q_lo, 0, &r_hi, &r_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);  // initial modulo of the precomputed residue

	mask = 1<<(b_pre_shift.w); /* the 1 on mask now points to the bit-pos after the first modulo was necessary */
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_95: tid=%ld: p=%llx, k=%llx, q=%llx:%llx, mask=%llx, r=%llx:%llx\n", tid, pp, k, q_hi, q_lo, mask, r_hi, r_lo);
#endif
    
	while (mask)
	{
	  square_96_192_64(&hi, &mid, &lo, r_hi, r_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 ); /*hi:mid:lo = (r * r); */
    mod_192_96_s(hi, mid, lo, q_hi, q_lo, ( exp&mask ) ? 1 : 0, &r_hi, &r_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                );
	  mask = mask >> 1;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_95: q=%llx:%llx, mask=%llx, r=%llx:%llx\n", q_hi, q_lo, mask, r_hi, r_lo);
#endif
	}
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakto_cl_95: tid=%ld: q=%llx:%llx, k=%llx, r=%llx:%llx\n", tid, q_hi, q_lo, k, r_hi, r_lo);
#endif

/* finally check if we found a factor and write the factor to RES[] */
  if((r_hi==0) && (r_lo==1))
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
      printf("mfakto_cl_95: tid=%ld found factor: q=%llx:%llx, k=%llx, r=%llx:%llx\n", tid, q_hi, q_lo, k, r_hi, r_lo);
#endif
      tid=ATOMIC_INC(RES[0]);
      if(tid<10)				/* limit to 10 factors per class */
      {
        RES[tid*3 + 1]= (uint) q_hi & 0xFFFFFFFF;
        RES[tid*3 + 2]= (uint) (q_lo >> 32);
        RES[tid*3 + 3]= (uint) q_lo & 0xFFFFFFFF;
      }
  }
}


/*===========uint exp, ulong k_base, __global uint *k_tab, ulong4 b_pre_shift, __global uint *RES============================
__kernel void mfakto_cl_barrett92_64(__private uint exp, __private ulong k_base,
                                  __global uint *k_tab, __private ulong4 b_pre_shift,
                                  __private int bit_max64, __global uint *RES)*/
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/

#ifndef BARRETT_VECTOR_SIZE

/****************************************
 ****************************************
 * 32-bit based 79- and 92-bit barrett-kernels
 *
 ****************************************
 ****************************************/

int cmp_ge_96(int96_1t a, int96_1t b)
/* checks if a is greater or equal than b */
{
  if(a.d2 == b.d2)
  {
    if(a.d1 == b.d1)return(a.d0 >= b.d0);
    else            return(a.d1 >  b.d1);
  }
  else              return(a.d2 >  b.d2);
}


void sub_96(int96_1t *res, int96_1t a, int96_1t b)
/* a must be greater or equal b!
res = a - b */
{
  /*
  res->d0 = __sub_cc (a.d0, b.d0);
  res->d1 = __subc_cc(a.d1, b.d1);
  res->d2 = __subc   (a.d2, b.d2);
  */
  uint carry= b.d0 > a.d0;

  res->d0 = a.d0 - b.d0;
  res->d1 = a.d1 - b.d1 - (carry ? 1 : 0);
  carry   = (res->d1 > a.d1) || ((res->d1 == a.d1) && carry);
  res->d2 = a.d2 - b.d2 - (carry ? 1 : 0);
}

int96_1t sub_if_gte_96(int96_1t a, int96_1t b)
/* return (a>b)?a-b:a */
{
  int96_1t tmp;
  /* do the subtraction and use tmp.d2 to decide if the result is valid (if a was > b) */

  uint carry= b.d0 > a.d0;

  tmp.d0 = a.d0 - b.d0;
  tmp.d1 = a.d1 - b.d1 - (carry ? 1 : 0);
  carry   = (tmp.d1 > a.d1) || ((tmp.d1 == a.d1) && carry);
  tmp.d2 = a.d2 - b.d2 - (carry ? 1 : 0);

  return (tmp.d2 > a.d2) ? a : tmp;
}


void mul_96(int96_1t *res, int96_1t a, int96_1t b)
/* res = a * b */
{
  /*
  res->d0 = __umul32  (a.d0, b.d0);

  res->d1 = __add_cc(__umul32hi(a.d0, b.d0), __umul32  (a.d1, b.d0));
  res->d2 = __addc  (__umul32  (a.d2, b.d0), __umul32hi(a.d1, b.d0));
  
  res->d1 = __add_cc(res->d1,                __umul32  (a.d0, b.d1));
  res->d2 = __addc  (res->d2,                __umul32hi(a.d0, b.d1));

  res->d2+= __umul32  (a.d0, b.d2);

  res->d2+= __umul32  (a.d1, b.d1);
  */
  uint tmp;
#ifdef OpenCL_CRASH_BUG_WA
  uint carry;  /* FIX: not needed when OpenCL drivers are fixed.
                  until then, the other code below crashes.
                  The Workaround is a lot slower ... */
#endif

  res->d0  = a.d0 * b.d0;
  res->d1  = mul_hi(a.d0, b.d0);

  res->d2  = mul_hi(a.d1, b.d0);

  tmp = a.d1 * b.d0;
  res->d1 += tmp;
#ifdef OpenCL_CRASH_BUG_WA
  carry    = (tmp > res->d1)? 1 : 0;
#else
  res->d2 += (tmp > res->d1)? 1 : 0;
#endif

  res->d2 += mul_hi(a.d0, b.d1);

  tmp = a.d0 * b.d1;
  res->d1 += tmp;
#ifdef OpenCL_CRASH_BUG_WA
  carry   += (tmp > res->d1)? 1 : 0;
  for (tmp=0; tmp<carry; tmp++, res->d2++);
#else
  res->d2 += (tmp > res->d1)? 1 : 0;
#endif

  res->d2 += a.d0 * b.d2 + a.d1 * b.d1 + a.d2 * b.d0;
}


void mul_96_192_no_low2(int192_1t *res, int96_1t a, int96_1t b)
/*
res ~= a * b
res.d0 and res.d1 are NOT computed. Carry from res.d1 to res.d2 is ignored,
too. So the digits res.d{2-5} might differ from mul_96_192(). In
mul_96_192() are two carries from res.d1 to res.d2. So ignoring the digits
res.d0 and res.d1 the result of mul_96_192_no_low() is 0 to 2 lower than
of mul_96_192().
 */
{
  /*
  res->d2 = __umul32  (a.d2, b.d0);
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d2 = __add_cc (res->d2, __umul32hi(a.d1, b.d0));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d2 = __add_cc (res->d2, __umul32hi(a.d0, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d1, b.d2));  
  res->d5 = __addc   (      0,                      0);

  res->d2 = __add_cc (res->d2, __umul32  (a.d0, b.d2));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
  res->d5 = __addc   (res->d5, __umul32hi(a.d2, b.d2));

  res->d2 = __add_cc (res->d2, __umul32  (a.d1, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
  */
  uint tmp;
  
  res->d2  = mul_hi(a.d1, b.d0);

  tmp      = mul_hi(a.d0, b.d1);
  res->d2 += tmp;
  res->d3  = (tmp > res->d2)? 1 : 0;

  tmp      = a.d2 * b.d0;
  res->d2 += tmp;
  res->d3 += (tmp > res->d2)? 1 : 0;

  tmp      = a.d1 * b.d1;
  res->d2 += tmp;
  res->d3 += (tmp > res->d2)? 1 : 0;

  tmp      = a.d0 * b.d2;
  res->d2 += tmp;
  res->d3 += (tmp > res->d2)? 1 : 0;


  tmp      = mul_hi(a.d2, b.d0);
  res->d3 += tmp;
  res->d4  = (tmp > res->d3)? 1 : 0;

  tmp      = mul_hi(a.d1, b.d1);
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;

  tmp      = mul_hi(a.d0, b.d2);
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;

  tmp      = a.d2 * b.d1;
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;

  tmp      = a.d1 * b.d2;
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;


  tmp      = mul_hi(a.d2, b.d1);
  res->d4 += tmp;
  res->d5  = (tmp > res->d4)? 1 : 0;

  tmp      = mul_hi(a.d1, b.d2);
  res->d4 += tmp;
  res->d5 += (tmp > res->d4)? 1 : 0;

  tmp      = a.d2 * b.d2;
  res->d4 += tmp;
  res->d5 += (tmp > res->d4)? 1 : 0;


  res->d5 += mul_hi(a.d2, b.d2);
}


void mul_96_192_no_low3(int192_1t *res, int96_1t a, int96_1t b)
/*
res ~= a * b
res.d0, res.d1 and res.d2 are NOT computed. Carry to res.d3 is ignored,
too. So the digits res.d{3-5} might differ from mul_96_192(). In
mul_96_192() are four carries from res.d2 to res.d3. So ignoring the digits
res.d0, res.d1 and res.d2 the result of mul_96_192_no_low() is 0 to 4 lower
than of mul_96_192().
 */
{
  /*
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d3 = __add_cc (res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d3 = __add_cc (res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc   (res->d4, __umul32hi(a.d1, b.d2)); // no carry propagation to d5 needed: 0xFFFF.FFFF * 0xFFFF.FFFF + 0xFFFF.FFFF + 0xFFFF.FFFE = 0xFFFF.FFFF.FFFF.FFFE
//  res->d4 = __addc_cc(res->d4, __umul32hi(a.d1, b.d2));  
//  res->d5 = __addc   (      0,                      0);

  res->d3 = __add_cc (res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
//  res->d5 = __addc   (res->d5, __umul32hi(a.d2, b.d2));
  res->d5 = __addc   (      0, __umul32hi(a.d2, b.d2));

  res->d3 = __add_cc (res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
  */
  uint tmp;

  res->d3  = mul_hi(a.d2, b.d0);

  tmp      = mul_hi(a.d1, b.d1);
  res->d3 += tmp;
  res->d4  = (tmp > res->d3)? 1 : 0;

  tmp      = mul_hi(a.d0, b.d2);
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;

  tmp      = a.d2 * b.d1;
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;

  tmp      = a.d1 * b.d2;
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;


  tmp      = mul_hi(a.d2, b.d1);
  res->d4 += tmp;
  res->d5  = (tmp > res->d4)? 1 : 0;

  tmp      = mul_hi(a.d1, b.d2);
  res->d4 += tmp;
  res->d5 += (tmp > res->d4)? 1 : 0;

  tmp      = a.d2 * b.d2;
  res->d4 += tmp;
  res->d5 += (tmp > res->d4)? 1 : 0;


  res->d5 += mul_hi(a.d2, b.d2);
}


void square_96_192(int192_1t *res, int96_1t a)
/* res = a^2 = a.d0^2 + a.d1^2 + a.d2^2 + 2(a.d0*a.d1 + a.d0*a.d2 + a.d1*a.d2) */
{
/*
highest possible value for x * x is 0xFFFFFFF9
this occurs for x = {479772853, 1667710795, 2627256501, 3815194443}
Adding x*x to a few carries will not cascade the carry
*/
  uint tmp;

  res->d0  = a.d0 * a.d0;

  res->d1  = mul_hi(a.d0, a.d0);

  tmp      = a.d0 * a.d1;
  res->d1 += tmp;
  res->d2  = (tmp > res->d1)? 1 : 0;
  res->d1 += tmp;
  res->d2 += (tmp > res->d1)? 1 : 0;


  res->d2 += a.d1 * a.d1;  // no carry possible

  tmp      = mul_hi(a.d0, a.d1);
  res->d2 += tmp;
  res->d3  = (tmp > res->d2)? 1 : 0;
  res->d2 += tmp;
  res->d3 += (tmp > res->d2)? 1 : 0;

  tmp      = a.d0 * a.d2;
  res->d2 += tmp;
  res->d3 += (tmp > res->d2)? 1 : 0;
  res->d2 += tmp;
  res->d3 += (tmp > res->d2)? 1 : 0;


  tmp      = mul_hi(a.d1, a.d1);
  res->d3 += tmp;
  res->d4  = (tmp > res->d3)? 1 : 0;

  tmp      = mul_hi(a.d0, a.d2);
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;

  tmp      = a.d1 * a.d2;
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;


  res->d4 += a.d2 * a.d2; // no carry possible

  tmp      = mul_hi(a.d1, a.d2);
  res->d4 += tmp;
  res->d5  = (tmp > res->d4)? 1 : 0;
  res->d4 += tmp;
  res->d5 += (tmp > res->d4)? 1 : 0;


  res->d5 += mul_hi(a.d2, a.d2);
}


void square_96_160(int192_1t *res, int96_1t a)
/* res = a^2 */
/* this is a stripped down version of square_96_192, it doesn't compute res.d5
and is a little bit faster.
For correct results a must be less than 2^80 (a.d2 less than 2^16) */
{
/*
highest possible value for x * x is 0xFFFFFFF9
this occurs for x = {479772853, 1667710795, 2627256501, 3815194443}
Adding x*x to a few carries will not cascade the carry
*/
  uint tmp, TWOad2 = a.d2 << 1; // a.d2 < 2^16 so this always fits

  res->d0  = a.d0 * a.d0;

  res->d1  = mul_hi(a.d0, a.d0);

  tmp      = a.d0 * a.d1;
  res->d1 += tmp;
  res->d2  = (tmp > res->d1)? 1 : 0;
  res->d1 += tmp;
  res->d2 += (tmp > res->d1)? 1 : 0;


  res->d2 += a.d1 * a.d1;  // no carry possible

  tmp      = mul_hi(a.d0, a.d1);
  res->d2 += tmp;
  res->d3  = (tmp > res->d2)? 1 : 0;
  res->d2 += tmp;
  res->d3 += (tmp > res->d2)? 1 : 0;

  tmp      = a.d0 * TWOad2;  
  res->d2 += tmp;
  res->d3 += (tmp > res->d2)? 1 : 0;


  tmp      = mul_hi(a.d1, a.d1);
  res->d3 += tmp;
  res->d4  = (tmp > res->d3)? 1 : 0;

  tmp      = mul_hi(a.d0, TWOad2);
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;

  tmp      = a.d1 * TWOad2;
  res->d3 += tmp;
  res->d4 += (tmp > res->d3)? 1 : 0;


  res->d4 += a.d2 * a.d2; // no carry possible

  res->d4 += mul_hi(a.d1, TWOad2);
}


void shl_96(int96_1t *a)
/* shiftleft a one bit */
{
  a->d2 = (a->d2 << 1) + (a->d1 >> 31);
  a->d1 = (a->d1 << 1) + (a->d0 >> 31);
  a->d0 = a->d0 << 1;
}


#undef DIV_160_96
#ifndef CHECKS_MODBASECASE
void div_192_96(int96_1t *res, int192_1t q, int96_1t n, float nf)
#else
void div_192_96(int96_1t *res, int192_1t q, int96_1t n, float nf, uint *modbasecase_debug)
#endif
/* res = q / n (integer division) */
{
  float qf;
  uint qi, tmp, carry;
  int192_1t nn;
  int96_1t tmp96;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf= convert_float_rtz(q.d5);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d4);
#else
  #ifdef CHECKS_MODBASECASE
    q.d5 = 0;	// later checks in debug code will test if q.d5 is 0 or not but 160bit variant ignores q.d5
  #endif
  qf= convert_float_rtz(q.d4);
#endif  
  qf*= 2097152.0f;

  qi=convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 11;

// nn = n * qi
  nn.d2  = n.d0 * qi;
  nn.d3  = mul_hi(n.d0, qi);
  tmp    = n.d1 * qi;
  nn.d3 += tmp;
  nn.d4  = (tmp > nn.d3)? 1 : 0;
  tmp    = mul_hi(n.d1, qi);
  nn.d4 += tmp;
#ifndef DIV_160_96
  nn.d5  = (tmp > nn.d4)? 1 : 0;
  tmp    = n.d2 * qi;
  nn.d4 += tmp;
  nn.d5 += (tmp > nn.d4)? 1 : 0;
  nn.d5 += mul_hi(n.d2, qi);
#else
  nn.d4 += n.d2 * qi;
#endif

// shiftleft nn 11 bits
#ifndef DIV_160_96
  nn.d5 = (nn.d5 << 11) + (nn.d4 >> 21);
#endif
  nn.d4 = (nn.d4 << 11) + (nn.d3 >> 21);
  nn.d3 = (nn.d3 << 11) + (nn.d2 >> 21);
  nn.d2 =  nn.d2 << 11;

//  q = q - nn
  carry= (nn.d2 > q.d2);
  q.d2 = q.d2 - nn.d2;

  tmp  = q.d3 - nn.d3 - (carry ? 1 : 0);
  carry= (tmp > q.d3) || (carry && (tmp == q.d3));
  q.d3 = tmp;

#ifndef DIV_160_96
  tmp  = q.d4 - nn.d4 - (carry ? 1 : 0);
  carry= (tmp > q.d4) || (carry && (tmp == q.d4));
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d5 - (carry ? 1 : 0);
#else
  q.d4 = q.d4 - nn.d4 - (carry ? 1 : 0);
#endif
/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef DIV_160_96
  qf= convert_float_rtz(q.d5);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d4);
#else
  qf= convert_float_rtz(q.d4);
#endif
  qf= qf * 4294967296.0f + convert_float_rtz(q.d3);
  qf*= 512.0f;

  qi= convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);

  res->d1 =  qi << 23;
  res->d2 += qi >>  9;

// nn = n * qi
  nn.d1 = n.d0 * qi;
  nn.d2  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d2 += tmp;
  nn.d3  = (tmp > nn.d2)? 1 : 0;
  tmp    = mul_hi(n.d1, qi);
  nn.d3 += tmp;
  nn.d4  = (tmp > nn.d3)? 1 : 0;
  tmp    = n.d2* qi;
  nn.d3 += tmp;
  nn.d4 += (tmp > nn.d3)? 1 : 0;
  nn.d4 += mul_hi(n.d2, qi);

  // shiftleft nn 23 bits
#ifdef CHECKS_MODBASECASE
  nn.d5 =                  nn.d4 >> 9;
#endif  
  nn.d4 = (nn.d4 << 23) + (nn.d3 >> 9);
  nn.d3 = (nn.d3 << 23) + (nn.d2 >> 9);
  nn.d2 = (nn.d2 << 23) + (nn.d1 >> 9);
  nn.d1 =  nn.d1 << 23;

// q = q - nn
  carry= (nn.d1 > q.d1);
  q.d1 = q.d1 - nn.d1;

  tmp  = q.d2 - nn.d2 - (carry ? 1 : 0);
  carry= (tmp > q.d2) || (carry && (tmp == q.d2));
  q.d2 = tmp;

  tmp  = q.d3 - nn.d3 - (carry ? 1 : 0);
  carry= (tmp > q.d3) || (carry && (tmp == q.d3));
  q.d3 = tmp;

#ifdef CHECKS_MODBASECASE
  tmp  = q.d4 - nn.d4 - (carry ? 1 : 0);
  carry= (tmp > q.d4) || (carry && (tmp == q.d4));
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d5 - (carry ? 1 : 0);
#else
  q.d4 = q.d4 - nn.d4 - (carry ? 1 : 0);
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= convert_float_rtz(q.d4);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d3);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi= convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

  tmp     = (qi << 3);
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + (qi >> 29) + ((tmp > res->d1)? 1 : 0);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;

// nn = n * qi
  
  nn.d1 = n.d0 * qi;
  nn.d2  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d2 += tmp;
  nn.d3  = (tmp > nn.d2)? 1 : 0;
  tmp    = mul_hi(n.d1, qi);
  nn.d3 += tmp;
  nn.d4  = (tmp > nn.d3)? 1 : 0;
  tmp    = n.d2* qi;
  nn.d3 += tmp;
  nn.d4 += (tmp > nn.d3)? 1 : 0;
  nn.d4 += mul_hi(n.d2, qi);

//  q = q - nn
  carry= (nn.d1 > q.d1);
  q.d1 = q.d1 - nn.d1;

  tmp  = q.d2 - nn.d2 - (carry ? 1 : 0);
  carry= (tmp > q.d2) || (carry && (tmp == q.d2));
  q.d2 = tmp;

  tmp  = q.d3 - nn.d3 - (carry ? 1 : 0);
  carry= (tmp > q.d3) || (carry && (tmp == q.d3));
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 - (carry ? 1 : 0);

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 4);

  qf= convert_float_rtz(q.d4);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d3);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d2);
  qf*= 131072.0f;
  
  qi= convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  tmp     = qi >> 17;
  res->d0 = qi << 15;
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + ((tmp > res->d1)? 1 : 0);
  
// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = (tmp > nn.d1)? 1 : 0;
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
  nn.d3  = (tmp > nn.d2)? 1 : 0;
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += (tmp > nn.d2)? 1 : 0;
  nn.d3 += mul_hi(n.d2, qi);

// shiftleft nn 15 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                  nn.d3 >> 17;
#endif
  nn.d3 = (nn.d3 << 15) + (nn.d2 >> 17);
  nn.d2 = (nn.d2 << 15) + (nn.d1 >> 17);
  nn.d1 = (nn.d1 << 15) + (nn.d0 >> 17);
  nn.d0 =  nn.d0 << 15;

//  q = q - nn
  carry= (nn.d0 > q.d0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - (carry ? 1 : 0);
  carry= (tmp > q.d1) || (carry && (tmp == q.d1));
  q.d1 = tmp;

  tmp  = q.d2 - nn.d2 - (carry ? 1 : 0);
  carry= (tmp > q.d2) || (carry && (tmp == q.d2));
  q.d2 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d3 = q.d3 - nn.d3 - (carry ? 1 : 0);
#else
  tmp  = q.d3 - nn.d3 - (carry ? 1 : 0);
  carry= (tmp > q.d3) || (carry && (tmp == q.d3));
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 - (carry ? 1 : 0);
#endif

/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 6);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= convert_float_rtz(q.d3);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d2);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d1);
  
  qi= convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 += qi;
  carry    = (qi > res->d0)? 1 : 0;
  res->d1 += carry;
  res->d2 += (carry > res->d1)? 1 : 0;
  
// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = (tmp > nn.d1)? 1 : 0;
#ifndef CHECKS_MODBASECASE
  nn.d2 += mul_hi(n.d1, qi) + n.d2* qi;
#else
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
  nn.d3  = (tmp > nn.d2)? 1 : 0;
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += (tmp > nn.d2)? 1 : 0;
  nn.d3 += mul_hi(n.d2, qi);
#endif  

//  q = q - nn
  carry= (nn.d0 > q.d0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - (carry ? 1 : 0);
  carry= (tmp > q.d1) || (carry && (tmp == q.d1));
  q.d1 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d2 = q.d2 - nn.d2 - (carry ? 1 : 0);
#else
  tmp  = q.d2 - nn.d2 - (carry ? 1 : 0);
  carry= (tmp > q.d2) || (carry && (tmp == q.d2));
  q.d2 = tmp;

  q.d3 = q.d3 - nn.d3 - (carry ? 1 : 0);
#endif

//  res->d0=q.d0;
//  res->d1=q.d1;
//  res->d2=q.d2;
  tmp96.d0=q.d0;
  tmp96.d1=q.d1;
  tmp96.d2=q.d2;
  
  MODBASECASE_NONZERO_ERROR(q.d5, 6, 5, 9);
  MODBASECASE_NONZERO_ERROR(q.d4, 6, 4, 10);
  MODBASECASE_NONZERO_ERROR(q.d3, 6, 3, 11);

/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
  if(cmp_ge_96(tmp96,n))
  {
    res->d0 += 1;
    carry    = (res->d0 == 0)? 1 : 0;
    res->d1 += carry;
    res->d2 += (carry > res->d1)? 1 : 0;
  }
}


#define DIV_160_96
#ifndef CHECKS_MODBASECASE
void div_160_96(int96_1t *res, int192_1t q, int96_1t n, float nf)
#else
void div_160_96(int96_1t *res, int192_1t q, int96_1t n, float nf, uint *modbasecase_debug)
#endif
/* res = q / n (integer division) */
/* the code of div_160_96() is an EXACT COPY of div_192_96(), the only
difference is that the 160bit version ignores the most significant
word of q (q.d5) because it assumes it is 0. This is controlled by defining
DIV_160_96 here. */
{
  float qf;
  uint qi, tmp, carry;
  int192_1t nn;
  int96_1t tmp96;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf= convert_float_rtz(q.d5);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d4);
#else
  #ifdef CHECKS_MODBASECASE
    q.d5 = 0;	// later checks in debug code will test if q.d5 is 0 or not but 160bit variant ignores q.d5
  #endif
  qf= convert_float_rtz(q.d4);
#endif  
  qf*= 2097152.0f;

  qi=convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 11;

// nn = n * qi
  nn.d2  = n.d0 * qi;
  nn.d3  = mul_hi(n.d0, qi);
  tmp    = n.d1 * qi;
  nn.d3 += tmp;
  nn.d4  = (tmp > nn.d3)? 1 : 0;
  tmp    = mul_hi(n.d1, qi);
  nn.d4 += tmp;
#ifndef DIV_160_96
  nn.d5  = (tmp > nn.d4)? 1 : 0;
  tmp    = n.d2 * qi;
  nn.d4 += tmp;
  nn.d5 += (tmp > nn.d4)? 1 : 0;
  nn.d5 += mul_hi(n.d2, qi);
#else
  nn.d4 += n.d2 * qi;
#endif

// shiftleft nn 11 bits
#ifndef DIV_160_96
  nn.d5 = (nn.d5 << 11) + (nn.d4 >> 21);
#endif
  nn.d4 = (nn.d4 << 11) + (nn.d3 >> 21);
  nn.d3 = (nn.d3 << 11) + (nn.d2 >> 21);
  nn.d2 =  nn.d2 << 11;

//  q = q - nn
  carry= (nn.d2 > q.d2);
  q.d2 = q.d2 - nn.d2;

  tmp  = q.d3 - nn.d3 - (carry ? 1 : 0);
  carry= (tmp > q.d3) || (carry && (tmp == q.d3));
  q.d3 = tmp;

#ifndef DIV_160_96
  tmp  = q.d4 - nn.d4 - (carry ? 1 : 0);
  carry= (tmp > q.d4) || (carry && (tmp == q.d4));
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d5 - (carry ? 1 : 0);
#else
  q.d4 = q.d4 - nn.d4 - (carry ? 1 : 0);
#endif
/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef DIV_160_96
  qf= convert_float_rtz(q.d5);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d4);
#else
  qf= convert_float_rtz(q.d4);
#endif
  qf= qf * 4294967296.0f + convert_float_rtz(q.d3);
  qf*= 512.0f;

  qi= convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);

  res->d1 =  qi << 23;
  res->d2 += qi >>  9;

// nn = n * qi
  nn.d1 = n.d0 * qi;
  nn.d2  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d2 += tmp;
  nn.d3  = (tmp > nn.d2)? 1 : 0;
  tmp    = mul_hi(n.d1, qi);
  nn.d3 += tmp;
  nn.d4  = (tmp > nn.d3)? 1 : 0;
  tmp    = n.d2* qi;
  nn.d3 += tmp;
  nn.d4 += (tmp > nn.d3)? 1 : 0;
  nn.d4 += mul_hi(n.d2, qi);

  // shiftleft nn 23 bits
#ifdef CHECKS_MODBASECASE
  nn.d5 =                  nn.d4 >> 9;
#endif  
  nn.d4 = (nn.d4 << 23) + (nn.d3 >> 9);
  nn.d3 = (nn.d3 << 23) + (nn.d2 >> 9);
  nn.d2 = (nn.d2 << 23) + (nn.d1 >> 9);
  nn.d1 =  nn.d1 << 23;

// q = q - nn
  carry= (nn.d1 > q.d1);
  q.d1 = q.d1 - nn.d1;

  tmp  = q.d2 - nn.d2 - (carry ? 1 : 0);
  carry= (tmp > q.d2) || (carry && (tmp == q.d2));
  q.d2 = tmp;

  tmp  = q.d3 - nn.d3 - (carry ? 1 : 0);
  carry= (tmp > q.d3) || (carry && (tmp == q.d3));
  q.d3 = tmp;

#ifdef CHECKS_MODBASECASE
  tmp  = q.d4 - nn.d4 - (carry ? 1 : 0);
  carry= (tmp > q.d4) || (carry && (tmp == q.d4));
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d5 - (carry ? 1 : 0);
#else
  q.d4 = q.d4 - nn.d4 - (carry ? 1 : 0);
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= convert_float_rtz(q.d4);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d3);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi= convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

  tmp     = (qi << 3);
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + (qi >> 29) + ((tmp > res->d1)? 1 : 0);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;

// nn = n * qi
  
  nn.d1 = n.d0 * qi;
  nn.d2  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d2 += tmp;
  nn.d3  = (tmp > nn.d2)? 1 : 0;
  tmp    = mul_hi(n.d1, qi);
  nn.d3 += tmp;
  nn.d4  = (tmp > nn.d3)? 1 : 0;
  tmp    = n.d2* qi;
  nn.d3 += tmp;
  nn.d4 += (tmp > nn.d3)? 1 : 0;
  nn.d4 += mul_hi(n.d2, qi);

//  q = q - nn
  carry= (nn.d1 > q.d1);
  q.d1 = q.d1 - nn.d1;

  tmp  = q.d2 - nn.d2 - (carry ? 1 : 0);
  carry= (tmp > q.d2) || (carry && (tmp == q.d2));
  q.d2 = tmp;

  tmp  = q.d3 - nn.d3 - (carry ? 1 : 0);
  carry= (tmp > q.d3) || (carry && (tmp == q.d3));
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 - (carry ? 1 : 0);

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 4);

  qf= convert_float_rtz(q.d4);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d3);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d2);
  qf*= 131072.0f;
  
  qi= convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  tmp     = qi >> 17;
  res->d0 = qi << 15;
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + ((tmp > res->d1)? 1 : 0);
  
// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = (tmp > nn.d1)? 1 : 0;
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
  nn.d3  = (tmp > nn.d2)? 1 : 0;
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += (tmp > nn.d2)? 1 : 0;
  nn.d3 += mul_hi(n.d2, qi);

// shiftleft nn 15 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                  nn.d3 >> 17;
#endif
  nn.d3 = (nn.d3 << 15) + (nn.d2 >> 17);
  nn.d2 = (nn.d2 << 15) + (nn.d1 >> 17);
  nn.d1 = (nn.d1 << 15) + (nn.d0 >> 17);
  nn.d0 =  nn.d0 << 15;

//  q = q - nn
  carry= (nn.d0 > q.d0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - (carry ? 1 : 0);
  carry= (tmp > q.d1) || (carry && (tmp == q.d1));
  q.d1 = tmp;

  tmp  = q.d2 - nn.d2 - (carry ? 1 : 0);
  carry= (tmp > q.d2) || (carry && (tmp == q.d2));
  q.d2 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d3 = q.d3 - nn.d3 - (carry ? 1 : 0);
#else
  tmp  = q.d3 - nn.d3 - (carry ? 1 : 0);
  carry= (tmp > q.d3) || (carry && (tmp == q.d3));
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 - (carry ? 1 : 0);
#endif

/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 6);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= convert_float_rtz(q.d3);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d2);
  qf= qf * 4294967296.0f + convert_float_rtz(q.d1);
  
  qi= convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 += qi;
  carry    = (qi > res->d0)? 1 : 0;
  res->d1 += carry;
  res->d2 += (carry > res->d1)? 1 : 0;
  
// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = (tmp > nn.d1)? 1 : 0;
#ifndef CHECKS_MODBASECASE
  nn.d2 += mul_hi(n.d1, qi) + n.d2* qi;
#else
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
  nn.d3  = (tmp > nn.d2)? 1 : 0;
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += (tmp > nn.d2)? 1 : 0;
  nn.d3 += mul_hi(n.d2, qi);
#endif  

//  q = q - nn
  carry= (nn.d0 > q.d0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - (carry ? 1 : 0);
  carry= (tmp > q.d1) || (carry && (tmp == q.d1));
  q.d1 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d2 = q.d2 - nn.d2 - (carry ? 1 : 0);
#else
  tmp  = q.d2 - nn.d2 - (carry ? 1 : 0);
  carry= (tmp > q.d2) || (carry && (tmp == q.d2));
  q.d2 = tmp;

  q.d3 = q.d3 - nn.d3 - (carry ? 1 : 0);
#endif

//  res->d0=q.d0;
//  res->d1=q.d1;
//  res->d2=q.d2;
  tmp96.d0=q.d0;
  tmp96.d1=q.d1;
  tmp96.d2=q.d2;
  
  MODBASECASE_NONZERO_ERROR(q.d5, 6, 5, 9);
  MODBASECASE_NONZERO_ERROR(q.d4, 6, 4, 10);
  MODBASECASE_NONZERO_ERROR(q.d3, 6, 3, 11);

/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
  if(cmp_ge_96(tmp96,n))
  {
    res->d0 += 1;
    carry    = (res->d0 == 0)? 1 : 0;
    res->d1 += carry;
    res->d2 += (carry > res->d1)? 1 : 0;
  }
}
#undef DIV_160_96




#ifndef CHECKS_MODBASECASE
void mod_simple_96(int96_1t *res, int96_1t q, int96_1t n, float nf
#if (TRACE_KERNEL > 1)
                  , __private uint tid
#endif
)
#else
void mod_simple_96(int96_1t *res, int96_1t q, int96_1t n, float nf, int bit_max64, unsigned int limit, unsigned int *modbasecase_debug)
#endif
/*
res = q mod n
used for refinement in barrett modular multiplication
assumes q < 6n (6n includes "optional mul 2")
*/
{
  float qf;
  uint qi;
  int96_1t nn;
  uint tmp, carry;

  qf = convert_float_rtz(q.d2);
  qf = qf * 4294967296.0f + convert_float_rtz(q.d1);
  
  qi = convert_uint(qf*nf);

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
  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 100, qi, 12);
  }
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_simple_96: q=%x:%x:%x, n=%x:%x:%x, nf=%.9G, qf=%f, qi=%x\n",
        q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, nf, qf, qi);
#endif

  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = (tmp > nn.d1)? 1 : 0;
  nn.d2 += mul_hi(n.d1, qi) + n.d2* qi;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mod_simple_96: nn=%x:%x:%x\n",
        nn.d2, nn.d1, nn.d0);
#endif

  carry= (nn.d0 > q.d0);
  res->d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - (carry ? 1 : 0);
  carry= (tmp > q.d1) || (carry && (tmp == q.d1));
  res->d1 = tmp;

  res->d2 = q.d2 - nn.d2 - (carry ? 1 : 0);
}

#ifndef CHECKS_MODBASECASE
__kernel void mfakto_cl_barrett92(__private uint exp, __private int96_1t k, __global uint *k_tab,
         __private int shiftcount, __private int192_1t b, __global uint *RES, __private int bit_max64)
#else
__kernel void mfakto_cl_barrett92(__private uint exp, __private int96_1t k, __global uint *k_tab,
         __private int shiftcount, __private int192_1t b, __global uint *RES, __private int bit_max64, __global uint *modbasecase_debug)
#endif
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int96_1t exp96,f;
  __private int96_1t a, u;
  __private int192_1t tmp192;
  __private int96_1t tmp96;
  __private float ff;
  __private int bit_max64_32 = 32 - bit_max64; /* used for bit shifting... */
  __private uint t, tid, tmp, carry;

	tid = get_global_id(0)+get_global_size(0)*get_global_id(1);

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp<<1;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_barrett92: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0, k.d2, k.d1, k.d0);
#endif

  t    = k_tab[tid];
  tmp  = t * 4620; // NUM_CLASSES
  k.d0 = k.d0 + tmp;
  k.d1 = k.d1 + mul_hi(t, 4620) + ((tmp > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x, tmp=%x\n",
        tid, t, k.d2, k.d1, k.d0, tmp);
#endif
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0  = k.d0 * exp96.d0 + 1;

  f.d1  = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  tmp   = mul_hi(k.d0, exp96.d0);
  f.d1 += tmp;
  f.d2 += (tmp > f.d1)? 1 : 0;

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mul_hi(k.d1, exp96.d0) + ((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= convert_float_rtz(f.d2);
  ff= ff * 4294967296.0f + convert_float_rtz(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we always underestimate the quotient
        
  // OpenCL shifts 32-bit values by 31 at most
  tmp192.d5 = (0x40000000 >> (31 - bit_max64)) >> (31 - bit_max64);	// tmp192 = 2^(2*bit_max)
  tmp192.d4 = (1 << bit_max64) << bit_max64;   // 1 << (b << 1) = (1 << b) << b
  tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#if (TRACE_KERNEL > 4)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: tmp=%x:%x:0:0:0:0, bit_max64=%d, bit_max64_32=%d\n",
        tmp192.d5, tmp192.d4, bit_max64, bit_max64_32);
#endif

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);						// u = floor(tmp192 / f)
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(tmp192 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: u=%x:%x:%x, ff=%G\n",
        u.d2, u.d1, u.d0, ff);
#endif

  a.d0 = (b.d2 >> bit_max64) + (b.d3 << bit_max64_32);			// a = b / (2^bit_max)
  a.d1 = (b.d3 >> bit_max64) + (b.d4 << bit_max64_32);
  a.d2 = (b.d4 >> bit_max64) + (b.d5 << bit_max64_32);

  mul_96_192_no_low2(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2, a.d1, a.d0, tmp192.d5, tmp192.d4, tmp192.d3, tmp192.d2);
#endif

  a.d0 = (tmp192.d2 >> bit_max64) + (tmp192.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = (tmp192.d3 >> bit_max64) + (tmp192.d4 << bit_max64_32);
  a.d2 = (tmp192.d4 >> bit_max64) + (tmp192.d5 << bit_max64_32);

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2, a.d1, a.d0, tmp96.d2, tmp96.d1, tmp96.d0);
#endif
  carry= (tmp96.d0 > b.d0);
  tmp96.d0 = b.d0 - tmp96.d0;

  tmp  = b.d1 - tmp96.d1 - (carry ? 1 : 0);
  carry= (tmp > b.d1) || (carry && (tmp == b.d1));
  tmp96.d1 = tmp;

  tmp96.d2 = b.d2 - tmp96.d2 - (carry ? 1 : 0);	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2, b.d1, b.d0, tmp96.d2, tmp96.d1, tmp96.d0);
#endif
#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 6;
  if(bit_max64 == 1) limit = 8;						// bit_max == 65, due to decreased accuracy of mul_96_192_no_low2() above we need a higher threshold
  if(bit_max64 == 2) limit = 7;						// bit_max == 66, ...
  mod_simple_96(&a, tmp96, f, ff, bit_max64, limit, modbasecase_debug);
#endif
  
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2, tmp96.d1, tmp96.d0, f.d2, f.d1, f.d0, a.d2, a.d1, a.d0 );
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_192(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2, a.d1, a.d0, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0 );
#endif
    a.d0 = (b.d2 >> bit_max64) + (b.d3 << bit_max64_32);		// a = b / (2^bit_max)
    a.d1 = (b.d3 >> bit_max64) + (b.d4 << bit_max64_32);
    a.d2 = (b.d4 >> bit_max64) + (b.d5 << bit_max64_32);

    mul_96_192_no_low2(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2, a.d1, a.d0, tmp192.d5, tmp192.d4, tmp192.d3, tmp192.d2);
#endif
    a.d0 = (tmp192.d2 >> bit_max64) + (tmp192.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = (tmp192.d3 >> bit_max64) + (tmp192.d4 << bit_max64_32);
    a.d2 = (tmp192.d4 >> bit_max64) + (tmp192.d5 << bit_max64_32);

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2, a.d1, a.d0, tmp96.d2, tmp96.d1, tmp96.d0);
#endif
    carry= (tmp96.d0 > b.d0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - (carry ? 1 : 0);
    carry= (tmp > b.d1) || (carry && (tmp == b.d1));
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - (carry ? 1 : 0);	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2, b.d1, b.d0, tmp96.d2, tmp96.d1, tmp96.d0);
#endif
    if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    if(bit_max64 == 1) limit = 8;					// bit_max == 65, due to decreased accuracy of mul_96_192_no_low2() above we need a higher threshold
    if(bit_max64 == 2) limit = 7;					// bit_max == 66, ...
    mod_simple_96(&a, tmp96, f, ff, bit_max64, limit, modbasecase_debug);
#endif

    exp<<=1;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("loopend: exp=%d, tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        exp, tmp96.d2, tmp96.d1, tmp96.d0, f.d2, f.d1, f.d0, a.d2, a.d1, a.d0 );
#endif
  }
  
  if(cmp_ge_96(a,f))				// final adjustment in case a >= f
  {
    sub_96(&a, a, f);
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x \n",
         a.d2, a.d1, a.d0 );
#endif
  }

#if defined CHECKS_MODBASECASE && defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= 200
  if(cmp_ge_96(a,f) && f.d2)
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif
  

/* finally check if we found a factor and write the factor to RES[] */
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("mfakto_cl_barrett92: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2, f.d1, f.d0, k.d2, k.d1, k.d0);
#endif
/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f != 1 */  
      tid=ATOMIC_INC(RES[0]);
    if(tid<10)				/* limit to 10 factors per class */
    {
      RES[tid*3 + 1]=f.d2;
      RES[tid*3 + 2]=f.d1;
      RES[tid*3 + 3]=f.d0;
    }
  }
}

#ifndef CHECKS_MODBASECASE
__kernel void mfakto_cl_barrett79(__private uint exp, __private int96_1t k, __global uint *k_tab,
         __private int shiftcount, __private int192_1t b, __global uint *RES, __private int bit_max64)
#else
__kernel void mfakto_cl_barrett79(__private uint exp, __private int96_1t k, __global uint *k_tab,
         __private int shiftcount, __private int192_1t b, __global uint *RES, __private int bit_max64, __global uint *modbasecase_debug)
#endif
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private int96_1t exp96,f;
  __private int96_1t a, u;
  __private int192_1t tmp192;
  __private int96_1t tmp96;
  __private float ff;
  __private uint t, tid, tmp, carry;

	tid = get_global_id(0)+get_global_size(0)*get_global_id(1);

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp<<1;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0, k.d2, k.d1, k.d0);
#endif

  t    = k_tab[tid];
  tmp  = t * 4620; // NUM_CLASSES
  k.d0 = k.d0 + tmp;
  k.d1 = k.d1 + mul_hi(t, 4620) + ((tmp > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x, tmp=%x\n",
        tid, t, k.d2, k.d1, k.d0, tmp);
#endif

//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0  = k.d0 * exp96.d0 + 1;

  f.d1  = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  tmp   = mul_hi(k.d0, exp96.d0);
  f.d1 += tmp;
  f.d2 += (tmp > f.d1)? 1 : 0;

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mul_hi(k.d1, exp96.d0) + ((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount);
#endif

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= convert_float_rtz(f.d2);
  ff= ff * 4294967296.0f + convert_float_rtz(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we allways underestimate the quotient

  tmp192.d4 = 0xFFFFFFFF;						// tmp is nearly 2^(81)
  tmp192.d3 = 0xFFFFFFFF;
  tmp192.d2 = 0xFFFFFFFF;
  tmp192.d1 = 0xFFFFFFFF;
  tmp192.d0 = 0xFFFFFFFF;

#ifndef CHECKS_MODBASECASE
  div_160_96(&u,tmp192,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_160_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: u=%x:%x:%x, ff=%G\n",
        u.d2, u.d1, u.d0, ff);
#endif

  a.d0 = b.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = b.d3;
  a.d2 = b.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2, a.d1, a.d0, tmp192.d5, tmp192.d4, tmp192.d3);
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2, a.d1, a.d0, tmp96.d2, tmp96.d1, tmp96.d0);
#endif

  carry= (tmp96.d0 > b.d0);
  tmp96.d0 = b.d0 - tmp96.d0;

  tmp  = b.d1 - tmp96.d1 - (carry ? 1 : 0);
  carry= (tmp > b.d1) || (carry && (tmp == b.d1));
  tmp96.d1 = tmp;

  tmp96.d2 = b.d2 - tmp96.d2 - (carry ? 1 : 0);	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2, b.d1, b.d0, tmp96.d2, tmp96.d1, tmp96.d0);
#endif

#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 6;
  if(bit_max64 == 15) limit = 9;					// bit_max == 79, due to decreased accuracy of mul_96_192_no_low3() above we need a higher threshold
  mod_simple_96(&a, tmp96, f, ff, 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2, tmp96.d1, tmp96.d0, f.d2, f.d1, f.d0, a.d2, a.d1, a.d0 );
#endif
 
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2, a.d1, a.d0, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0 );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2, a.d1, a.d0, tmp192.d5, tmp192.d4, tmp192.d3);
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2, a.d1, a.d0, tmp96.d2, tmp96.d1, tmp96.d0);
#endif

    carry= (tmp96.d0 > b.d0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - (carry ? 1 : 0);
    carry= (tmp > b.d1) || (carry && (tmp == b.d1));
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - (carry ? 1 : 0);	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2, b.d1, b.d0, tmp96.d2, tmp96.d1, tmp96.d0);
#endif

    if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    if(bit_max64 == 15) limit = 9;					// bit_max == 79, due to decreased accuracy of mul_96_192_no_low3() above we need a higher threshold
    mod_simple_96(&a, tmp96, f, ff, 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2, tmp96.d1, tmp96.d0, f.d2, f.d1, f.d0, a.d2, a.d1, a.d0 );
#endif

    exp<<=1;
  }

  if(cmp_ge_96(a,f))							// final adjustment in case a >= f
  {
    sub_96(&a, a, f);
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x \n",
         a.d2, a.d1, a.d0 );
#endif
  }
  
#if defined CHECKS_MODBASECASE
  if(cmp_ge_96(a,f) && f.d2)						// factors < 2^64 are not supported by this kernel
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif
  
/* finally check if we found a factor and write the factor to RES[] */
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("mfakto_cl_barrett: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2, f.d1, f.d0, k.d2, k.d1, k.d0);
#endif

/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f = 1! */  
      tid=ATOMIC_INC(RES[0]);
    if(tid<10)				/* limit to 10 factors per class */
    {
      RES[tid*3 + 1]=f.d2;
      RES[tid*3 + 2]=f.d1;
      RES[tid*3 + 3]=f.d0;
    }
  }
}
#endif // barrett

/*******************************
 ******* 24-bit-stuff **********
 *******************************/

void mul_24_48(uint *res_hi, uint *res_lo, uint a, uint b)
/* res_hi*(2^24) + res_lo = a * b */
{ // PERF: inline its use
/* thats how it should be, but the mul24_hi is missing ...
  *res_lo = mul24(a,b) & 0xFFFFFF;
  *res_hi = mul24_hi(a,b) >> 8;       // PERF: check for mul24_hi
  */
  *res_lo  = mul24(a,b);
//  *res_hi  = (mul_hi(a,b) << 8) | (*res_lo >> 24);       // PERF: check for mul24_hi
  *res_hi  = mad24(mul_hi(a,b), 256, (*res_lo >> 24));
  *res_lo &= 0xFFFFFF;
}


void copy_72(int72_t *a, int72_t b)
/* a = b */
{
  a->d0 = b.d0;
  a->d1 = b.d1;
  a->d2 = b.d2;
}

int gte_72(int72_t a, int72_t b)
/* returns
0  if a < b
1  if a >= b */
{
if (a.d2 == b.d2)
  if (a.d1 == b.d1)
    return (a.d0 > b.d0);
  else
    return (a.d1 > b.d1);
else
  return (a.d2 > b.d2);
}

int cmp_72(int72_t a, int72_t b)
/* returns
-1 if a < b
0  if a = b
1  if a > b */
{
  if(a.d2 < b.d2)return -1;
  if(a.d2 > b.d2)return 1;
  if(a.d1 < b.d1)return -1;
  if(a.d1 > b.d1)return 1;
  if(a.d0 < b.d0)return -1;
  if(a.d0 > b.d0)return 1;
  return 0;
}


void sub_72(int72_t *res, int72_t a, int72_t b)
/* a must be greater or equal b!
res = a - b */
{
  /*
  res->d0 = __sub_cc (a.d0, b.d0) & 0xFFFFFF;
  res->d1 = __subc_cc(a.d1, b.d1) & 0xFFFFFF;
  res->d2 = __subc   (a.d2, b.d2) & 0xFFFFFF;
  */
  res->d0 = (a.d0 - b.d0) & 0xFFFFFF;
  res->d1 = (a.d1 - b.d1 - ((b.d0 > a.d0) ? 1 : 0));
  res->d2 = (a.d2 - b.d2 - ((res->d1 > a.d1) ? 1 : 0)) & 0xFFFFFF;
  res->d1&= 0xFFFFFF;
}

int72_t sub_if_gte_72(int72_t a, int72_t b)
/* return (a>b)?a-b:a */
{
  int72_t tmp;
  /* do the subtraction and use tmp.d2 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0) & 0xFFFFFF;
  tmp.d1 = (a.d1 - b.d1 - ((b.d0 > a.d0) ? 1 : 0));
  tmp.d2 = (a.d2 - b.d2 - ((tmp.d1 > a.d1) ? 1 : 0)) & 0xFFFFFF;
  tmp.d1&= 0xFFFFFF;

  return (tmp.d2 > a.d2) ? a : tmp;
}

void mul_72(int72_t *res, int72_t a, int72_t b)
/* res = (a * b) mod (2^72) */
{
  uint hi,lo;

  mul_24_48(&hi, &lo, a.d0, b.d0);
  res->d0 = lo;
  res->d1 = hi;

  mul_24_48(&hi, &lo, a.d1, b.d0);
  res->d1 += lo;
  res->d2 = hi;

  mul_24_48(&hi, &lo, a.d0, b.d1);
  res->d1 += lo;
  res->d2 += hi;

  res->d2 = mad24(a.d2, b.d0, res->d2);

  res->d2 = mad24(a.d1, b.d1, res->d2);

  res->d2 = mad24(a.d0, b.d2, res->d2);

//  no need to carry res->d0

  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d2 &= 0xFFFFFF;
}


void square_72_144(int144_t *res, int72_t a)
/* res = a^2 */
{
  uint tmp;

  tmp      =  mul24(a.d0, a.d0);
  res->d1  =  mad24(mul_hi(a.d0, a.d0), 256, (tmp >> 24));
  res->d0  =  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
  res->d2  =  mad24(mul_hi(a.d1, a.d0), 512, (tmp >> 23));
  res->d1 += (tmp << 1) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d0);
  res->d3  =  mad24(mul_hi(a.d2, a.d0), 512, (tmp >> 23));
  res->d2 += (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 +=  mad24(mul_hi(a.d1, a.d1), 256, (tmp >> 24));
  res->d2 +=  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  =  mad24(mul_hi(a.d2, a.d1), 512, (tmp >> 23));
  res->d3 += (tmp << 1) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d2);
  res->d5  =  mad24(mul_hi(a.d2, a.d2), 256, (tmp >> 24));
  res->d4 +=  tmp       & 0xFFFFFF;

/*  res->d0 doesn't need carry */
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;
/*  res->d5 doesn't need carry */
}


void square_72_144_shl(int144_t *res, int72_t a)
/* res = 2* a^2 */
{ // PERF: use local copy for intermediate res->...?
  uint tmp;

  tmp      =  mul24(a.d0, a.d0);
  res->d1  =  mad24(mul_hi(a.d0, a.d0), 512, (tmp >> 23));
  res->d0  = (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
  res->d2  =  mad24(mul_hi(a.d1, a.d0), 1024, (tmp >> 22));
  res->d1 += (tmp << 2) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d0);
  res->d3  =  mad24(mul_hi(a.d2, a.d0), 1024, (tmp >> 22));
  res->d2 += (tmp << 2) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 +=  mad24(mul_hi(a.d1, a.d1), 512, (tmp >> 23));
  res->d2 += (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  =  mad24(mul_hi(a.d2, a.d1), 1024, (tmp >> 22));
  res->d3 += (tmp << 2) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d2);
  res->d5  =  mad24(mul_hi(a.d2, a.d2), 512, (tmp >> 23));
  res->d4 += (tmp << 1) & 0xFFFFFF;

/*  res->d0 doesn't need carry */
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;
/*  res->d5 doesn't need carry */
}


void mod_144_72(int72_t *res, int144_t q, int72_t n, float nf
#if (TRACE_KERNEL > 1)
                   , __private uint tid
#endif
#ifdef CHECKS_MODBASECASE
                   , __global uint *modbasecase_debug
#endif

)
/* res = q mod n */
{
  float qf;
  uint  qi, tmp;
  int144_t nn={0};

/********** Step 1, Offset 2^51 (2*24 + 3) **********/
  qf= convert_float_rte(q.d5);
  qf= qf * 16777216.0f + convert_float_rte(q.d4);
//  qf= qf * 16777216.0f + convert_float_rte(q.d3);
  qf*= 2097152.0f * 16777216.0f;

  qi=convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

#if (TRACE_KERNEL > 3)
// Bug in 11.6: floats are printed wrong in the first 3 params - insert dummies
  if (tid==TRACE_TID) printf("mod_%d%d_%d#1: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
//    if (tid==TRACE_TID) printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", qf, qf, qf, qf, qf, qf, qf, qf, qf, qf, qf, qf);
 //   if (tid==TRACE_TID) printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", nf, nf, nf, nf, nf, nf, nf, nf, nf, nf, nf, nf);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, qi);
#endif

//  nn.d0=0;
//  nn.d1=0;
// nn = n * qi AND shiftleft 3 bits at once, carry is done later
  tmp    =  mul24(n.d0, qi);
  nn.d3  =  mad24(mul_hi(n.d0, qi), 2048, (tmp >> 21));
  nn.d2  = (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d1, qi);
  nn.d4  =  mad24(mul_hi(n.d1, qi), 2048, (tmp >> 21));
  nn.d3 += (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d2, qi);
  nn.d5  =  mad24(mul_hi(n.d2, qi), 2048, (tmp >> 21));
  nn.d4 += (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
#endif


/* do carry */
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;

//  MODBASECASE_NN_BIG_ERROR(0xFFFFFF, 1, nn.d5, 1);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#1: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
#endif

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d2 = __sub_cc (q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5); */
  q.d2 = q.d2 - nn.d2;
  q.d3 = q.d3 - nn.d3 - ((q.d2 > 0xFFFFFF)?1:0);
  q.d4 = q.d4 - nn.d4 - ((q.d3 > 0xFFFFFF)?1:0);
  q.d5 = q.d5 - nn.d5 - ((q.d4 > 0xFFFFFF)?1:0);
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 2, Offset 2^31 (1*24 + 7) **********/
  qf= convert_float_rte(q.d5);
  qf= qf * 16777216.0f + convert_float_rte(q.d4);
  qf= qf * 16777216.0f + convert_float_rte(q.d3);
  qf= qf * 16777216.0f + convert_float_rte(q.d2);
  qf*= 131072.0f;

  qi=convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 2);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#2: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
    //if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", 0.0f, 1.0f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#2: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, qi);
#endif

//  nn.d0=0;
// nn = n * qi AND shiftleft 7 bits at once, carry is done later

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d0, qi);
  nn.d2  =  mad24(mul_hi(n.d0, qi), 32768, (tmp >> 17));
  nn.d1  = (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d1, qi);
  nn.d3  =  mad24(mul_hi(n.d1, qi), 32768, (tmp >> 17));
  nn.d2 += (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d2, qi);
  nn.d4  =  mad24(mul_hi(n.d2, qi), 32768, (tmp >> 17));
  nn.d3 += (tmp << 7) & 0xFFFFFF;
#if (TRACE_KERNEL > 2) || defined(CHECKS_MODBASECASE)
  nn.d5=0;
#endif
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

/* do carry */
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
#if (TRACE_KERNEL > 2) || defined(CHECKS_MODBASECASE)
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#2: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
#endif

/* q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d1 = __sub_cc (q.d1, nn.d1) & 0xFFFFFF;
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE  
  q.d4 = __subc   (q.d4, nn.d4) & 0xFFFFFF;
#else
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5);
#endif */
  q.d1 = q.d1 - nn.d1;
  q.d2 = q.d2 - nn.d2 - ((q.d1 > 0xFFFFFF)?1:0);
  q.d3 = q.d3 - nn.d3 - ((q.d2 > 0xFFFFFF)?1:0);
  q.d4 = q.d4 - nn.d4 - ((q.d3 > 0xFFFFFF)?1:0);
#ifdef CHECKS_MODBASECASE  
  q.d5 = q.d5 - nn.d5 - ((q.d4 > 0xFFFFFF)?1:0);
#endif
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 3, Offset 2^11 (0*24 + 11) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 3);

  qf= convert_float_rte(q.d4);
  qf= qf * 16777216.0f + convert_float_rte(q.d3);
  qf= qf * 16777216.0f + convert_float_rte(q.d2);
// d1 not needed as d2, 3 and 4 provide enough significant bits
//  qf= qf * 16777216.0f + convert_float_rte(q.d1);
  qf*= 8192.0f * 16777216.0f;

  qi=convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 4);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#3: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
    // if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", -1.0e10f, 3.2e8f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#3: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, qi);
#endif

//nn = n * qi, shiftleft is done later
/*  nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0); */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d0, qi);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d1, qi);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d2, qi);
  nn.d3 = mad24(mul_hi(n.d2, qi), 256, tmp >> 24);
  nn.d2 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  /* do carry */
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#3: before shl(11): nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif
// shiftleft 11 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                             nn.d3>>13;
  nn.d3 = mad24(nn.d3 & 0x1FFF, 2048, nn.d2>>13);
#else  
  nn.d3 = mad24(nn.d3,          2048, nn.d2>>13);	// we don't need to clear top bits here, this is done during q = q - nn
#endif  
  nn.d2 = mad24(nn.d2 & 0x1FFF, 2048, nn.d1>>13);
  nn.d1 = mad24(nn.d1 & 0x1FFF, 2048, nn.d0>>13);
  nn.d0 = ((nn.d0 & 0x1FFF)<<11);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
#endif

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d0 = __sub_cc (q.d0, nn.d0) & 0xFFFFFF;
  q.d1 = __subc_cc(q.d1, nn.d1) & 0xFFFFFF;
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE
  q.d3 = __subc   (q.d3, nn.d3) & 0xFFFFFF;
#else
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc   (q.d4, nn.d4);
#endif */
  q.d0 = q.d0 - nn.d0;
  q.d1 = q.d1 - nn.d1 - ((q.d0 > 0xFFFFFF)?1:0);
  q.d2 = q.d2 - nn.d2 - ((q.d1 > 0xFFFFFF)?1:0);
  q.d3 = q.d3 - nn.d3 - ((q.d2 > 0xFFFFFF)?1:0);
#ifdef CHECKS_MODBASECASE
  q.d4 = q.d4 - nn.d4 - ((q.d3 > 0xFFFFFF)?1:0);
#endif
  q.d0 &= 0xFFFFFF;
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;

/********** Step 4, Offset 2^0 (0*24 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 5);
  MODBASECASE_NONZERO_ERROR(q.d4, 4, 4, 6);

  qf= convert_float_rte(q.d3);
  qf= qf * 16777216.0f + convert_float_rte(q.d2);
  qf= qf * 16777216.0f + convert_float_rte(q.d1);
//  qf= qf * 16777216.0f + convert_float_rte(q.d0);
  qf= qf * 16777216.0f;

  qi=convert_uint(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 7);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#4: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
    //if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", qf, nf, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, qi);
#endif

  /* nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE
  nn.d2 = __addc   (mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi));
#else
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0);
#endif */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d0, qi);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d1, qi);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

#ifndef CHECKS_MODBASECASE  
  nn.d2 = mad24(n.d2, qi, nn.d2);
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;  // carry
#else
  tmp   = mul24(n.d2, qi);
  nn.d3 = mad24(mul_hi(n.d2, qi), 256, tmp >> 24);
  nn.d2 += tmp & 0xFFFFFF;
// do carry
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#4.3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
#endif

/* q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d0 = __sub_cc (q.d0, nn.d0) & 0xFFFFFF;
  q.d1 = __subc_cc(q.d1, nn.d1) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE  
  q.d2 = __subc   (q.d2, nn.d2) & 0xFFFFFF;
#else
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc   (q.d3, nn.d3);
#endif */

  q.d0 = q.d0 - nn.d0;
  q.d1 = q.d1 - nn.d1 - ((q.d0 > 0xFFFFFF)?1:0);
  q.d2 = q.d2 - nn.d2 - ((q.d1 > 0xFFFFFF)?1:0);
#ifdef CHECKS_MODBASECASE  
  q.d3 = q.d3 - nn.d3 - ((q.d2 > 0xFFFFFF)?1:0);
#endif

  res->d0 = q.d0 & 0xFFFFFF;
  res->d1 = q.d1 & 0xFFFFFF;
  res->d2 = q.d2 & 0xFFFFFF;
  
  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 8);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 9);
  MODBASECASE_NONZERO_ERROR(q.d3, 5, 3, 10);


#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, res->d2, res->d1, res->d0, n.d2, n.d1, n.d0, qi);
#endif

}


__kernel void mfakto_cl_71(__private uint exp, __private int72_t k, 
                           __global uint *k_tab, __private int shiftcount,
                           __private int144_t b, __global uint *RES
#ifdef CHECKS_MODBASECASE  
                         , __global uint *modbasecase_debug
#endif
                           )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  int72_t exp72,f;
  int72_t a;
  int tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  float ff;

  exp72.d2=0;exp72.d1=exp>>23;exp72.d0=(exp&0x7FFFFF)<<1;	// exp72 = 2 * exp

  mul_24_48(&(a.d1),&(a.d0),k_tab[tid],4620); // NUM_CLASSES
  k.d0 += a.d0;
  k.d1 += a.d1;
  k.d1 += k.d0 >> 24; k.d0 &= 0xFFFFFF;
  k.d2 += k.d1 >> 24; k.d1 &= 0xFFFFFF;		// k = k + k_tab[tid] * NUM_CLASSES

  mul_72(&f,k,exp72);				// f = 2 * k * exp
  f.d0 += 1;				      	// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_144_72().
Precalculated here since it is the same for all steps in the following loop */
  ff= convert_float(f.d2);
  ff= ff * 16777216.0f + convert_float(f.d1);
  ff= ff * 16777216.0f + convert_float(f.d0);

//  ff=0.9999997f/ff;
//  ff=__int_as_float(0x3f7ffffc) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
  ff=as_float(0x3f7ffffb) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
 
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_71: tid=%ld: p=%x, *2 =%x:%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d, b=%x:%x:%x:%x:%x:%x\n",
                              tid, exp, exp72.d1, exp72.d0, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0);
#endif

  mod_144_72(&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
#ifdef CHECKS_MODBASECASE  
                   , modbasecase_debug
#endif
                   );			// a = b mod f
  exp<<= 32 - shiftcount;
  while(exp)
  {
    if(exp&0x80000000)square_72_144_shl(&b,a);	// b = 2 * a^2 ("optional multiply by 2" in Prime 95 documentation)
    else              square_72_144(&b,a);	// b = a^2
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mfakto_cl_71: exp=%x,  %x:%x:%x ^2 (shl:%d) = %x:%x:%x:%x:%x:%x\n",
                              exp, a.d2, a.d1, a.d0, (exp&0x80000000?1:0), b.d5, b.d4, b.d3, b.d2, b.d1, b.d0);
#endif
    mod_144_72(&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
#ifdef CHECKS_MODBASECASE  
                   , modbasecase_debug
#endif
                   );			// a = b mod f
    exp<<=1;
  }
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakto_cl_71 result: f=%x:%x:%x, a=%x:%x:%x\n",
                              f.d2, f.d1, f.d0, a.d2, a.d1, a.d0);
#endif

  a=sub_if_gte_72(a,f);

/* finally check if we found a factor and write the factor to RES[] */
  if((a.d2|a.d1)==0 && a.d0==1)
  {
    if ((f.d2|f.d1)!=0 || f.d0 != 1)  // happens for k=0 and does not help us ;-)
    {
#if (TRACE_KERNEL > 0)  // trace this for any thread
      printf("mfakto_cl_71: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2, f.d1, f.d0, k.d2, k.d1, k.d0);
#endif
      tid=ATOMIC_INC(RES[0]);
      if(tid<10)				/* limit to 10 factors per class */
      {
        RES[tid*3 + 1]=f.d2;
        RES[tid*3 + 2]=f.d1;
        RES[tid*3 + 3]=f.d0;
      }
    }
  }
}

/***********************************************
 * 8-vector implementation of all
 * 24-bit-stuff for the 71-bit-kernel
 *
 ***********************************************/

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct _int72_8t
{
  uint8 d0,d1,d2;
}int72_8t;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct _int144_8t
{
  uint8 d0,d1,d2,d3,d4,d5;
}int144_8t;


void mul_24_48_8(uint8 *res_hi, uint8 *res_lo, uint8 a, uint8 b)
/* res_hi*(2^24) + res_lo = a * b */
{ // PERF: inline its use
/* thats how it should be, but the mul24_hi is missing ...
  *res_lo = mul24(a,b) & 0xFFFFFF;
  *res_hi = mul24_hi(a,b) >> 8;       // PERF: check for mul24_hi
  */
  *res_lo  = mul24(a,b);
//  *res_hi  = (mul_hi(a,b) << 8) | (*res_lo >> 24);       
  *res_hi  = mad24(mul_hi(a,b), 256, (*res_lo >> 24));       
  *res_lo &= 0xFFFFFF;
}


void copy_72_8(int72_8t *a, int72_8t b)
/* a = b */
{
  a->d0 = b.d0;
  a->d1 = b.d1;
  a->d2 = b.d2;
}

void sub_72_8(int72_8t *res, int72_8t a, int72_8t b)
/* a must be greater or equal b!
res = a - b */
{
  /*
  res->d0 = __sub_cc (a.d0, b.d0) & 0xFFFFFF;
  res->d1 = __subc_cc(a.d1, b.d1) & 0xFFFFFF;
  res->d2 = __subc   (a.d2, b.d2) & 0xFFFFFF;
  */

  res->d0 = (a.d0 - b.d0) & 0xFFFFFF;
  res->d1 = a.d1 - b.d1 - as_uint8((b.d0 > a.d0) ? 1 : 0);
  res->d2 = (a.d2 - b.d2 - as_uint8((res->d1 > a.d1) ? 1 : 0)) & 0xFFFFFF;
  res->d1&= 0xFFFFFF;
}

int72_8t sub_if_gte_72_8(int72_8t a, int72_8t b)
/* return (a>b)?a-b:a */
{
  int72_8t tmp;
  /* do the subtraction and use tmp.d2 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0) & 0xFFFFFF;
  tmp.d1 = (a.d1 - b.d1 - as_uint8((b.d0 > a.d0) ? 1 : 0));
  tmp.d2 = (a.d2 - b.d2 - as_uint8((tmp.d1 > a.d1) ? 1 : 0));
  tmp.d1&= 0xFFFFFF;

  /* tmp valid if tmp.d2 <= a.d2 (separately for each part of the vector) */
  tmp.d0 = (tmp.d2 > a.d2) ? a.d0 : tmp.d0;
  tmp.d1 = (tmp.d2 > a.d2) ? a.d1 : tmp.d1;
  tmp.d2 = (tmp.d2 > a.d2) ? a.d2 : tmp.d2 & 0xFFFFFF;

  return tmp;
}

void mul_72_8(int72_8t *res, int72_8t a, int72_t b)
/* res = (a * b) mod (2^72) */
{
  uint8 hi,lo; // PERF: inline mul_24_48

  mul_24_48_8(&hi, &lo, a.d0, b.d0);
  res->d0 = lo;
  res->d1 = hi;

  mul_24_48_8(&hi, &lo, a.d1, b.d0);
  res->d1 += lo;
  res->d2 = hi;

  mul_24_48_8(&hi, &lo, a.d0, b.d1);
  res->d1 += lo;
  res->d2 += hi;

  res->d2 = mad24(a.d2,b.d0,res->d2);

  res->d2 = mad24(a.d1,b.d1,res->d2);

  res->d2 = mad24(a.d0,b.d2,res->d2);

//  no need to carry res->d0

  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d2 &= 0xFFFFFF;
}


void square_72_144_8(int144_8t *res, int72_8t a)
/* res = a^2 */
{ // PERF: use local copy for intermediate res->...?
  uint8 tmp;

  tmp      =  mul24(a.d0, a.d0);
//  res->d1  = (mul_hi(a.d0, a.d0) << 8) | (tmp >> 24);
  res->d1  =  mad24(mul_hi(a.d0, a.d0), 256, (tmp >> 24));
  res->d0  =  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
//  res->d2  = (mul_hi(a.d1, a.d0) << 9) | (tmp >> 23);
  res->d2  =  mad24(mul_hi(a.d1, a.d0), 512, (tmp >> 23));
  res->d1 += (tmp << 1) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d0);
  res->d3  =  mad24(mul_hi(a.d2, a.d0), 512, (tmp >> 23));
  res->d2 += (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 +=  mad24(mul_hi(a.d1, a.d1), 256, (tmp >> 24));
  res->d2 +=  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  =  mad24(mul_hi(a.d2, a.d1), 512, (tmp >> 23));
  res->d3 += (tmp << 1) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d2);
  res->d5  =  mad24(mul_hi(a.d2, a.d2), 256, (tmp >> 24));
  res->d4 +=  tmp       & 0xFFFFFF;

/*  res->d0 doesn't need carry */
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;
/*  res->d5 doesn't need carry */
}


void square_72_144_8_shl(int144_8t *res, int72_8t a)
/* res = 2* a^2 */
{ // PERF: use local copy for intermediate res->...?
  uint8 tmp;

  tmp      =  mul24(a.d0, a.d0);
  res->d1  =  mad24(mul_hi(a.d0, a.d0), 512, (tmp >> 23));
  res->d0  = (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
  res->d2  =  mad24(mul_hi(a.d1, a.d0), 1024, (tmp >> 22));
  res->d1 += (tmp << 2) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d0);
  res->d3  =  mad24(mul_hi(a.d2, a.d0), 1024, (tmp >> 22));
  res->d2 += (tmp << 2) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 +=  mad24(mul_hi(a.d1, a.d1), 512, (tmp >> 23));
  res->d2 += (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  =  mad24(mul_hi(a.d2, a.d1), 1024, (tmp >> 22));
  res->d3 += (tmp << 2) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d2);
  res->d5  =  mad24(mul_hi(a.d2, a.d2), 512, (tmp >> 23));
  res->d4 += (tmp << 1) & 0xFFFFFF;

/*  res->d0 doesn't need carry */
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;
/*  res->d5 doesn't need carry */
}


void mod_144_72_8(int72_8t *res, int144_8t q, int72_8t n, float8 nf
#if (TRACE_KERNEL > 1)
                   , __private uint tid
#endif
)
/* res = q mod n */
{
  float8 qf;
  uint8  qi, tmp;
  int144_8t nn; // ={0,0,0,0};  // PERF: initialization needed?

/********** Step 1, Offset 2^51 (2*24 + 3) **********/
  qf= convert_float8_rte(q.d5);
  qf= qf * 16777216.0f + convert_float8_rte(q.d4);
//  qf= qf * 16777216.0f + convert_float8_rte(q.d3);
  qf*= 2097152.0f * 16777216.0f;

  qi=convert_uint8(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#1: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.s0, nf.s0, (qf*nf).s0, qi.s0);
//    if (tid==TRACE_TID) printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", qf, qf, qf, qf, qf, qf, qf, qf, qf, qf, qf, qf);
 //   if (tid==TRACE_TID) printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", nf, nf, nf, nf, nf, nf, nf, nf, nf, nf, nf, nf);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

//  nn.d0=0;
//  nn.d1=0;
// nn = n * qi AND shiftleft 3 bits at once, carry is done later
  tmp    =  mul24(n.d0, qi);
  nn.d3  =  mad24(mul_hi(n.d0, qi), 2048, (tmp >> 21));
//  nn.d3  = (mul_hi(n.d0, qi) << 11) | (tmp >> 21);
  nn.d2  = (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d1, qi);
//  nn.d4  = (mul_hi(n.d1, qi) << 11) | (tmp >> 21);
  nn.d4  =  mad24(mul_hi(n.d1, qi), 2048, (tmp >> 21));
  nn.d3 += (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d2, qi);
//  nn.d5  = (mul_hi(n.d2, qi) << 11) | (tmp >> 21);
  nn.d5  =  mad24(mul_hi(n.d2, qi), 2048, (tmp >> 21));
  nn.d4 += (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif


/* do carry */
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#1: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d2 = __sub_cc (q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5); */
  q.d2 = q.d2 - nn.d2;
  q.d3 = q.d3 - nn.d3 - as_uint8((q.d2 > 0xFFFFFF)?1:0);
  q.d4 = q.d4 - nn.d4 - as_uint8((q.d3 > 0xFFFFFF)?1:0);
  q.d5 = q.d5 - nn.d5 - as_uint8((q.d4 > 0xFFFFFF)?1:0);
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 2, Offset 2^31 (1*24 + 7) **********/
  qf= convert_float8_rte(q.d5);
  qf= qf * 16777216.0f + convert_float8_rte(q.d4);
  qf= qf * 16777216.0f + convert_float8_rte(q.d3);
  qf= qf * 16777216.0f + convert_float8_rte(q.d2);
  qf*= 131072.0f;

  qi=convert_uint8(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#2: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.s0, nf.s0, (qf*nf).s0, qi.s0);
    //if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", 0.0f, 1.0f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#2: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

//  nn.d0=0;
// nn = n * qi AND shiftleft 7 bits at once, carry is done later

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d0, qi);
//  nn.d2  = (mul_hi(n.d0, qi) << 15) | (tmp >> 17);
  nn.d2  =  mad24(mul_hi(n.d0, qi), 32768, (tmp >> 17));
  nn.d1  = (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d1, qi);
//  nn.d3  = (mul_hi(n.d1, qi) << 15) | (tmp >> 17);
  nn.d3  =  mad24(mul_hi(n.d1, qi), 32768, (tmp >> 17));
  nn.d2 += (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d2, qi);
//  nn.d4  = (mul_hi(n.d2, qi) << 15) | (tmp >> 17);
  nn.d4  =  mad24(mul_hi(n.d2, qi), 32768, (tmp >> 17));
  nn.d3 += (tmp << 7) & 0xFFFFFF;
#if (TRACE_KERNEL > 2)
  nn.d5=0;
#endif
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

/* do carry */
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
#if (TRACE_KERNEL > 2)
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#2: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

/* q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d1 = __sub_cc (q.d1, nn.d1) & 0xFFFFFF;
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE  
  q.d4 = __subc   (q.d4, nn.d4) & 0xFFFFFF;
#else
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5);
#endif */
  q.d1 = q.d1 - nn.d1;
  q.d2 = q.d2 - nn.d2 - as_uint8((q.d1 > 0xFFFFFF)?1:0);
  q.d3 = q.d3 - nn.d3 - as_uint8((q.d2 > 0xFFFFFF)?1:0);
  q.d4 = q.d4 - nn.d4 - as_uint8((q.d3 > 0xFFFFFF)?1:0);
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 3, Offset 2^11 (0*24 + 11) **********/
  qf= convert_float8_rte(q.d4);
  qf= qf * 16777216.0f + convert_float8_rte(q.d3);
  qf= qf * 16777216.0f + convert_float8_rte(q.d2);
//  qf= qf * 16777216.0f + convert_float8_rte(q.d1);
  qf*= 8192.0f * 16777216.0f;

  qi=convert_uint8(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#3: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
    // if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", -1.0e10f, 3.2e8f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#3: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

//nn = n * qi, shiftleft is done later
/*  nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0); */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d0, qi);
//  nn.d1 = (mul_hi(n.d0, qi) << 8) | (tmp >> 24);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d1, qi);
//  nn.d2 = (mul_hi(n.d1, qi) << 8) | (tmp >> 24);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d2, qi);
//  nn.d3 = (mul_hi(n.d2, qi) << 8) | (tmp >> 24);
  nn.d3 = mad24(mul_hi(n.d2, qi), 256, tmp >> 24);
  nn.d2 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  /* do carry */
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#3: before shl(11): nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif
// shiftleft 11 bits
//  nn.d3 = ( nn.d3          <<11) + (nn.d2>>13);	// we don't need to clear top bits here, this is done during q = q - nn
  nn.d3 = mad24(nn.d3,          2048, nn.d2>>13);	// we don't need to clear top bits here, this is done during q = q - nn
//  nn.d2 = ((nn.d2 & 0x1FFF)<<11) + (nn.d1>>13);
//  nn.d1 = ((nn.d1 & 0x1FFF)<<11) + (nn.d0>>13);
  nn.d2 = mad24(nn.d2 & 0x1FFF, 2048, nn.d1>>13);
  nn.d1 = mad24(nn.d1 & 0x1FFF, 2048, nn.d0>>13);
  nn.d0 = ((nn.d0 & 0x1FFF)<<11);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d0 = __sub_cc (q.d0, nn.d0) & 0xFFFFFF;
  q.d1 = __subc_cc(q.d1, nn.d1) & 0xFFFFFF;
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE
  q.d3 = __subc   (q.d3, nn.d3) & 0xFFFFFF;
#else
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc   (q.d4, nn.d4);
#endif */
  q.d0 = q.d0 - nn.d0;
  q.d1 = q.d1 - nn.d1 - as_uint8((q.d0 > 0xFFFFFF)?1:0);
  q.d2 = q.d2 - nn.d2 - as_uint8((q.d1 > 0xFFFFFF)?1:0);
  q.d3 = q.d3 - nn.d3 - as_uint8((q.d2 > 0xFFFFFF)?1:0);
  q.d0 &= 0xFFFFFF;
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;

/********** Step 4, Offset 2^0 (0*24 + 0) **********/

  qf= convert_float8_rte(q.d3);
  qf= qf * 16777216.0f + convert_float8_rte(q.d2);
  qf= qf * 16777216.0f + convert_float8_rte(q.d1);
//  qf= qf * 16777216.0f + convert_float8_rte(q.d0);
  qf= qf * 16777216.0f;

  qi=convert_uint8(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#4: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.s0, nf.s0, (qf*nf).s0, qi.s0);
    //if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", qf, nf, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

  /* nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE
  nn.d2 = __addc   (mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi));
#else
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0);
#endif */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d0, qi);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d1, qi);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d2 = mad24(n.d2, qi, nn.d2);
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#4.3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

/* q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d0 = __sub_cc (q.d0, nn.d0) & 0xFFFFFF;
  q.d1 = __subc_cc(q.d1, nn.d1) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE  
  q.d2 = __subc   (q.d2, nn.d2) & 0xFFFFFF;
#else
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc   (q.d3, nn.d3);
#endif */

  q.d0 = q.d0 - nn.d0;
  q.d1 = q.d1 - nn.d1 - as_uint8((q.d0 > 0xFFFFFF)?1:0);
  q.d2 = q.d2 - nn.d2 - as_uint8((q.d1 > 0xFFFFFF)?1:0);

  res->d0 = q.d0 & 0xFFFFFF;
  res->d1 = q.d1 & 0xFFFFFF;
  res->d2 = q.d2 & 0xFFFFFF;

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, res->d2.s0, res->d1.s0, res->d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

}


__kernel void mfakto_cl_71_8(__private uint exp, __private int72_t k_base, __global uint *k_tab, __private int shiftcount, __private int144_t b_in, __global uint *RES)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  int72_t   exp72;
  int72_8t  k;  
  int72_8t  a;       // result of the modulo
  int144_8t b;       // result of the squaring;
  int72_8t  f;       // the factor(s) to be tested
  int       tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * 8;
  float8    ff;
  uint8     t;

  exp72.d2=0;exp72.d1=exp>>23;exp72.d0=(exp&0x7FFFFF)<<1;	// exp72 = 2 * exp
  k.d0 = k_base.d0; k.d1 = k_base.d1; k.d2 = k_base.d2;   // widen to vec8
  b.d0 = b_in.d0; b.d1 = b_in.d1; b.d2 = b_in.d2;
  b.d3 = b_in.d3; b.d4 = b_in.d4; b.d5 = b_in.d5;
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];

  mul_24_48_8(&(a.d1), &(a.d0), t, 4620); // NUM_CLASSES
  k.d0 += a.d0;
  k.d1 += a.d1;
  k.d1 += k.d0 >> 24; k.d0 &= 0xFFFFFF;
  k.d2 += k.d1 >> 24; k.d1 &= 0xFFFFFF;		// k = k + k_tab[tid] * NUM_CLASSES

  mul_72_8(&f, k, exp72);				// f = 2 * k * exp
  f.d0 += 1;				      	// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_144_72().
Precalculated here since it is the same for all steps in the following loop */
  ff= convert_float8(f.d2);
  ff= ff * 16777216.0f + convert_float8(f.d1);
  ff= ff * 16777216.0f + convert_float8(f.d0);

//  ff=0.9999997f/ff;
//  ff=__int_as_float(0x3f7ffffc) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
  ff=as_float(0x3f7ffffb) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
 
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_71: tid=%ld: p=%x, *2 =%x:%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d, b=%x:%x:%x:%x:%x:%x\n",
                              tid, exp, exp72.d1, exp72.d0, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0);
#endif

  mod_144_72_8(&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);			// a = b mod f
  exp<<= 32 - shiftcount;
  while(exp)
  {
    if(exp&0x80000000)square_72_144_8_shl(&b,a);	// b = 2 * a^2 ("optional multiply by 2" in Prime 95 documentation)
    else              square_72_144_8(&b,a);	// b = a^2
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mfakto_cl_71: exp=%x,  %x:%x:%x ^2 (shl:%d) = %x:%x:%x:%x:%x:%x\n",
                              exp, a.d2, a.d1, a.d0, (exp&0x80000000?1:0), b.d5, b.d4, b.d3, b.d2, b.d1, b.d0);
#endif
    mod_144_72_8(&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
      );			// a = b mod f
    exp<<=1;
  }
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakto_cl_71 result: f=%x:%x:%x, a=%x:%x:%x\n",
                              f.d2, f.d1, f.d0, a.d2, a.d1, a.d0);
#endif

  a=sub_if_gte_72_8(a,f);

  EVAL_RES(s0)
  EVAL_RES(s1)
  EVAL_RES(s2)
  EVAL_RES(s3)
  EVAL_RES(s4)
  EVAL_RES(s5)
  EVAL_RES(s6)
  EVAL_RES(s7)
}


/***********************************************
 * 4-vector implementation of all
 * 24-bit-stuff for the 71-bit-kernel
 *
 ***********************************************/

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct _int72_4t
{
  uint4 d0,d1,d2;
}int72_4t;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct _int144_4t
{
  uint4 d0,d1,d2,d3,d4,d5;
}int144_4t;


void mul_24_48_4(uint4 *res_hi, uint4 *res_lo, uint4 a, uint4 b)
/* res_hi*(2^24) + res_lo = a * b */
{ // PERF: inline its use
/* thats how it should be, but the mul24_hi is missing ...
  *res_lo = mul24(a,b) & 0xFFFFFF;
  *res_hi = mul24_hi(a,b) >> 8;       // PERF: check for mul24_hi
  */
  *res_lo  = mul24(a,b);
//  *res_hi  = (mul_hi(a,b) << 8) | (*res_lo >> 24);       
  *res_hi  = mad24(mul_hi(a,b), 256, (*res_lo >> 24));       
  *res_lo &= 0xFFFFFF;
}


void copy_72_4(int72_4t *a, int72_4t b)
/* a = b */
{
  a->d0 = b.d0;
  a->d1 = b.d1;
  a->d2 = b.d2;
}

void sub_72_4(int72_4t *res, int72_4t a, int72_4t b)
/* a must be greater or equal b!
res = a - b */
{
  /*
  res->d0 = __sub_cc (a.d0, b.d0) & 0xFFFFFF;
  res->d1 = __subc_cc(a.d1, b.d1) & 0xFFFFFF;
  res->d2 = __subc   (a.d2, b.d2) & 0xFFFFFF;
  */

  res->d0 = (a.d0 - b.d0) & 0xFFFFFF;
  res->d1 = a.d1 - b.d1 - as_uint4((b.d0 > a.d0) ? 1 : 0);
  res->d2 = (a.d2 - b.d2 - as_uint4((res->d1 > a.d1) ? 1 : 0)) & 0xFFFFFF;
  res->d1&= 0xFFFFFF;
}

int72_4t sub_if_gte_72_4(int72_4t a, int72_4t b)
/* return (a>b)?a-b:a */
{
  int72_4t tmp;
  /* do the subtraction and use tmp.d2 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0) & 0xFFFFFF;
  tmp.d1 = (a.d1 - b.d1 - as_uint4((b.d0 > a.d0) ? 1 : 0));
  tmp.d2 = (a.d2 - b.d2 - as_uint4((tmp.d1 > a.d1) ? 1 : 0));
  tmp.d1&= 0xFFFFFF;

  /* tmp valid if tmp.d2 <= a.d2 (separately for each part of the vector) */
  tmp.d0 = (tmp.d2 > a.d2) ? a.d0 : tmp.d0;
  tmp.d1 = (tmp.d2 > a.d2) ? a.d1 : tmp.d1;
  tmp.d2 = (tmp.d2 > a.d2) ? a.d2 : tmp.d2 & 0xFFFFFF;

  return tmp;
}

void mul_72_4(int72_4t *res, int72_4t a, int72_t b)
/* res = (a * b) mod (2^72) */
{
  uint4 hi,lo; // PERF: inline mul_24_48

  mul_24_48_4(&hi, &lo, a.d0, b.d0);
  res->d0 = lo;
  res->d1 = hi;

  mul_24_48_4(&hi, &lo, a.d1, b.d0);
  res->d1 += lo;
  res->d2 = hi;

  mul_24_48_4(&hi, &lo, a.d0, b.d1);
  res->d1 += lo;
  res->d2 += hi;

  res->d2 = mad24(a.d2,b.d0,res->d2);

  res->d2 = mad24(a.d1,b.d1,res->d2);

  res->d2 = mad24(a.d0,b.d2,res->d2);

//  no need to carry res->d0

  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d2 &= 0xFFFFFF;
}


void square_72_144_4(int144_4t *res, int72_4t a)
/* res = a^2 */
{ // PERF: use local copy for intermediate res->...?
  uint4 tmp;

  tmp      =  mul24(a.d0, a.d0);
//  res->d1  = (mul_hi(a.d0, a.d0) << 8) | (tmp >> 24);
  res->d1  =  mad24(mul_hi(a.d0, a.d0), 256, (tmp >> 24));
  res->d0  =  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
//  res->d2  = (mul_hi(a.d1, a.d0) << 9) | (tmp >> 23);
  res->d2  =  mad24(mul_hi(a.d1, a.d0), 512, (tmp >> 23));
  res->d1 += (tmp << 1) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d0);
  res->d3  =  mad24(mul_hi(a.d2, a.d0), 512, (tmp >> 23));
  res->d2 += (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 +=  mad24(mul_hi(a.d1, a.d1), 256, (tmp >> 24));
  res->d2 +=  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  =  mad24(mul_hi(a.d2, a.d1), 512, (tmp >> 23));
  res->d3 += (tmp << 1) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d2);
  res->d5  =  mad24(mul_hi(a.d2, a.d2), 256, (tmp >> 24));
  res->d4 +=  tmp       & 0xFFFFFF;

/*  res->d0 doesn't need carry */
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;
/*  res->d5 doesn't need carry */
}


void square_72_144_4_shl(int144_4t *res, int72_4t a)
/* res = 2* a^2 */
{ // PERF: use local copy for intermediate res->...?
  uint4 tmp;

  tmp      =  mul24(a.d0, a.d0);
  res->d1  =  mad24(mul_hi(a.d0, a.d0), 512, (tmp >> 23));
  res->d0  = (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
  res->d2  =  mad24(mul_hi(a.d1, a.d0), 1024, (tmp >> 22));
  res->d1 += (tmp << 2) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d0);
  res->d3  =  mad24(mul_hi(a.d2, a.d0), 1024, (tmp >> 22));
  res->d2 += (tmp << 2) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 +=  mad24(mul_hi(a.d1, a.d1), 512, (tmp >> 23));
  res->d2 += (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  =  mad24(mul_hi(a.d2, a.d1), 1024, (tmp >> 22));
  res->d3 += (tmp << 2) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d2);
  res->d5  =  mad24(mul_hi(a.d2, a.d2), 512, (tmp >> 23));
  res->d4 += (tmp << 1) & 0xFFFFFF;

/*  res->d0 doesn't need carry */
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;
/*  res->d5 doesn't need carry */
}


void mod_144_72_4(int72_4t *res, int144_4t q, const int72_4t n, const float4 nf
#if (TRACE_KERNEL > 1)
                   , const uint tid
#endif
)
/* res = q mod n */
{
  float4 qf;
  uint4  qi, tmp;
  int144_4t nn; // ={0,0,0,0};  // PERF: initialization needed?

/********** Step 1, Offset 2^51 (2*24 + 3) **********/
  qf= convert_float4_rte(q.d5);
  qf= qf * 16777216.0f + convert_float4_rte(q.d4);
//  qf= qf * 16777216.0f + convert_float4_rte(q.d3);
  qf*= 2097152.0f * 16777216.0f;

  qi=convert_uint4(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#1: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.x, nf.x, (qf*nf).x, qi.x);
//    if (tid==TRACE_TID) printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", qf, qf, qf, qf, qf, qf, qf, qf, qf, qf, qf, qf);
 //   if (tid==TRACE_TID) printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", nf, nf, nf, nf, nf, nf, nf, nf, nf, nf, nf, nf);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.x, q.d4.x, q.d3.x, q.d2.x, q.d1.x, q.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

//  nn.d0=0;
//  nn.d1=0;
// nn = n * qi AND shiftleft 3 bits at once, carry is done later
  tmp    =  mul24(n.d0, qi);
  nn.d3  =  mad24(mul_hi(n.d0, qi), 2048, (tmp >> 21));
//  nn.d3  = (mul_hi(n.d0, qi) << 11) | (tmp >> 21);
  nn.d2  = (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp    =  mul24(n.d1, qi);
//  nn.d4  = (mul_hi(n.d1, qi) << 11) | (tmp >> 21);
  nn.d4  =  mad24(mul_hi(n.d1, qi), 2048, (tmp >> 21));
  nn.d3 += (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp    =  mul24(n.d2, qi);
//  nn.d5  = (mul_hi(n.d2, qi) << 11) | (tmp >> 21);
  nn.d5  =  mad24(mul_hi(n.d2, qi), 2048, (tmp >> 21));
  nn.d4 += (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif


/* do carry */
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#1: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d2 = __sub_cc (q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5); */
  q.d2 = q.d2 - nn.d2;
  q.d3 = q.d3 - nn.d3 - as_uint4((q.d2 > 0xFFFFFF)?1:0);
  q.d4 = q.d4 - nn.d4 - as_uint4((q.d3 > 0xFFFFFF)?1:0);
  q.d5 = q.d5 - nn.d5 - as_uint4((q.d4 > 0xFFFFFF)?1:0);
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 2, Offset 2^31 (1*24 + 7) **********/
  qf= convert_float4_rte(q.d5);
  qf= qf * 16777216.0f + convert_float4_rte(q.d4);
  qf= qf * 16777216.0f + convert_float4_rte(q.d3);
  qf= qf * 16777216.0f + convert_float4_rte(q.d2);
  qf*= 131072.0f;

  qi=convert_uint4(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#2: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.x, nf.x, (qf*nf).x, qi.x);
    //if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", 0.0f, 1.0f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#2: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.x, q.d4.x, q.d3.x, q.d2.x, q.d1.x, q.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

//  nn.d0=0;
// nn = n * qi AND shiftleft 7 bits at once, carry is done later

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp    =  mul24(n.d0, qi);
//  nn.d2  = (mul_hi(n.d0, qi) << 15) | (tmp >> 17);
  nn.d2  =  mad24(mul_hi(n.d0, qi), 32768, (tmp >> 17));
  nn.d1  = (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp    =  mul24(n.d1, qi);
//  nn.d3  = (mul_hi(n.d1, qi) << 15) | (tmp >> 17);
  nn.d3  =  mad24(mul_hi(n.d1, qi), 32768, (tmp >> 17));
  nn.d2 += (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp    =  mul24(n.d2, qi);
//  nn.d4  = (mul_hi(n.d2, qi) << 15) | (tmp >> 17);
  nn.d4  =  mad24(mul_hi(n.d2, qi), 32768, (tmp >> 17));
  nn.d3 += (tmp << 7) & 0xFFFFFF;
#if (TRACE_KERNEL > 2)
  nn.d5=0;
#endif
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

/* do carry */
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
#if (TRACE_KERNEL > 2)
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#2: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

/* q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d1 = __sub_cc (q.d1, nn.d1) & 0xFFFFFF;
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE  
  q.d4 = __subc   (q.d4, nn.d4) & 0xFFFFFF;
#else
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5);
#endif */
  q.d1 = q.d1 - nn.d1;
  q.d2 = q.d2 - nn.d2 - as_uint4((q.d1 > 0xFFFFFF)?1:0);
  q.d3 = q.d3 - nn.d3 - as_uint4((q.d2 > 0xFFFFFF)?1:0);
  q.d4 = q.d4 - nn.d4 - as_uint4((q.d3 > 0xFFFFFF)?1:0);
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 3, Offset 2^11 (0*24 + 11) **********/
  qf= convert_float4_rte(q.d4);
  qf= qf * 16777216.0f + convert_float4_rte(q.d3);
  qf= qf * 16777216.0f + convert_float4_rte(q.d2);
//  qf= qf * 16777216.0f + convert_float4_rte(q.d1);
  qf*= 8192.0f * 16777216.0f;

  qi=convert_uint4(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#3: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
    // if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", -1.0e10f, 3.2e8f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#3: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.x, q.d4.x, q.d3.x, q.d2.x, q.d1.x, q.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

//nn = n * qi, shiftleft is done later
/*  nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0); */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp   = mul24(n.d0, qi);
//  nn.d1 = (mul_hi(n.d0, qi) << 8) | (tmp >> 24);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp   = mul24(n.d1, qi);
//  nn.d2 = (mul_hi(n.d1, qi) << 8) | (tmp >> 24);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp   = mul24(n.d2, qi);
//  nn.d3 = (mul_hi(n.d2, qi) << 8) | (tmp >> 24);
  nn.d3 = mad24(mul_hi(n.d2, qi), 256, tmp >> 24);
  nn.d2 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  /* do carry */
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#3: before shl(11): nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif
// shiftleft 11 bits
//  nn.d3 = ( nn.d3          <<11) + (nn.d2>>13);	// we don't need to clear top bits here, this is done during q = q - nn
  nn.d3 = mad24(nn.d3,          2048, nn.d2>>13);	// we don't need to clear top bits here, this is done during q = q - nn
//  nn.d2 = ((nn.d2 & 0x1FFF)<<11) + (nn.d1>>13);
//  nn.d1 = ((nn.d1 & 0x1FFF)<<11) + (nn.d0>>13);
  nn.d2 = mad24(nn.d2 & 0x1FFF, 2048, nn.d1>>13);
  nn.d1 = mad24(nn.d1 & 0x1FFF, 2048, nn.d0>>13);
  nn.d0 = ((nn.d0 & 0x1FFF)<<11);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d0 = __sub_cc (q.d0, nn.d0) & 0xFFFFFF;
  q.d1 = __subc_cc(q.d1, nn.d1) & 0xFFFFFF;
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE
  q.d3 = __subc   (q.d3, nn.d3) & 0xFFFFFF;
#else
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc   (q.d4, nn.d4);
#endif */
  q.d0 = q.d0 - nn.d0;
  q.d1 = q.d1 - nn.d1 - as_uint4((q.d0 > 0xFFFFFF)?1:0);
  q.d2 = q.d2 - nn.d2 - as_uint4((q.d1 > 0xFFFFFF)?1:0);
  q.d3 = q.d3 - nn.d3 - as_uint4((q.d2 > 0xFFFFFF)?1:0);
  q.d0 &= 0xFFFFFF;
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;

/********** Step 4, Offset 2^0 (0*24 + 0) **********/

  qf= convert_float4_rte(q.d3);
  qf= qf * 16777216.0f + convert_float4_rte(q.d2);
  qf= qf * 16777216.0f + convert_float4_rte(q.d1);
  qf= qf * 16777216.0f;
//  qf= qf * 16777216.0f + convert_float4_rte(q.d0);

  qi=convert_uint4(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#4: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.x, nf.x, (qf*nf).x, qi.x);
    //if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", qf, nf, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.x, q.d4.x, q.d3.x, q.d2.x, q.d1.x, q.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

  /* nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE
  nn.d2 = __addc   (mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi));
#else
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0);
#endif */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp   = mul24(n.d0, qi);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  tmp   = mul24(n.d1, qi);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x);
#endif

  nn.d2 = mad24(n.d2, qi, nn.d2);
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#4.3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.x, nn.d4.x, nn.d3.x, nn.d2.x, nn.d1.x, nn.d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

/* q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d0 = __sub_cc (q.d0, nn.d0) & 0xFFFFFF;
  q.d1 = __subc_cc(q.d1, nn.d1) & 0xFFFFFF;
#ifndef CHECKS_MODBASECASE  
  q.d2 = __subc   (q.d2, nn.d2) & 0xFFFFFF;
#else
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc   (q.d3, nn.d3);
#endif */

  q.d0 = q.d0 - nn.d0;
  q.d1 = q.d1 - nn.d1 - as_uint4((q.d0 > 0xFFFFFF)?1:0);
  q.d2 = q.d2 - nn.d2 - as_uint4((q.d1 > 0xFFFFFF)?1:0);

  res->d0 = q.d0 & 0xFFFFFF;
  res->d1 = q.d1 & 0xFFFFFF;
  res->d2 = q.d2 & 0xFFFFFF;

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.x, q.d4.x, q.d3.x, res->d2.x, res->d1.x, res->d0.x, n.d2.x, n.d1.x, n.d0.x, qi.x);
#endif

}


__kernel void mfakto_cl_71_4(uint exp, __private int72_t k_base,
                   __global uint * restrict k_tab, __private int shiftcount,
                   __private int144_t b_in, __global uint * restrict RES)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  __private int72_t   exp72;
  __private int72_4t  k;  
  __private int72_4t  a;       // result of the modulo
  __private int144_4t b;       // result of the squaring;
  __private int72_4t  f;       // the factor(s) to be tested
  __private int       tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * 4;
  __private float4    ff;
  __private uint4     t;

  exp72.d2=0;exp72.d1=exp>>23;exp72.d0=(exp&0x7FFFFF)<<1;	// exp72 = 2 * exp
  k.d0 = k_base.d0; k.d1 = k_base.d1; k.d2 = k_base.d2;   // widen to vec4
  b.d0 = b_in.d0; b.d1 = b_in.d1; b.d2 = b_in.d2;
  b.d3 = b_in.d3; b.d4 = b_in.d4; b.d5 = b_in.d5;
  t.x = k_tab[tid];
  t.y = k_tab[tid+1];
  t.z = k_tab[tid+2];
  t.w = k_tab[tid+3];

  mul_24_48_4(&(a.d1), &(a.d0), t, 4620); // NUM_CLASSES
  k.d0 += a.d0;
  k.d1 += a.d1;
  k.d1 += k.d0 >> 24; k.d0 &= 0xFFFFFF;
  k.d2 += k.d1 >> 24; k.d1 &= 0xFFFFFF;		// k = k + k_tab[tid] * NUM_CLASSES

  mul_72_4(&f, k, exp72);				// f = 2 * k * exp
  f.d0 += 1;				      	// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_144_72().
Precalculated here since it is the same for all steps in the following loop */
  ff= convert_float4(f.d2);
  ff= ff * 16777216.0f + convert_float4(f.d1);
  ff= ff * 16777216.0f + convert_float4(f.d0);

//  ff=0.9999997f/ff;
//  ff=__int_as_float(0x3f7ffffc) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
  ff=as_float(0x3f7ffffb) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
 
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_71: tid=%ld: p=%x, *2 =%x:%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d, b=%x:%x:%x:%x:%x:%x\n",
                              tid, exp, exp72.d1, exp72.d0, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0);
#endif

  mod_144_72_4(&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);			// a = b mod f
  exp<<= 32 - shiftcount;
  while(exp)
  {
    if(exp&0x80000000)square_72_144_4_shl(&b,a);	// b = 2 * a^2 ("optional multiply by 2" in Prime 95 documentation)
    else              square_72_144_4(&b,a);	// b = a^2
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mfakto_cl_71: exp=%x,  %x:%x:%x ^2 (shl:%d) = %x:%x:%x:%x:%x:%x\n",
                              exp, a.d2, a.d1, a.d0, (exp&0x80000000?1:0), b.d5, b.d4, b.d3, b.d2, b.d1, b.d0);
#endif
    mod_144_72_4(&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
      );			// a = b mod f
    exp<<=1;
  }
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakto_cl_71 result: f=%x:%x:%x, a=%x:%x:%x\n",
                              f.d2, f.d1, f.d0, a.d2, a.d1, a.d0);
#endif

  a=sub_if_gte_72_4(a,f);

/* finally check if we found a factor and write the factor to RES[] */
  EVAL_RES(x)
  EVAL_RES(y)
  EVAL_RES(z)
  EVAL_RES(w)
}

