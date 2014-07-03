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

Version 0.14
*/

/****************************************
 ****************************************
 * 24-bit-stuff for the 72-bit-barrett-kernel
 * included by main kernel file
 ****************************************
 ****************************************/



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

/* all types already defined:
   int72_t, int144_t  ... scalar
   int72_v, int144_v  ... vector
   */

void div_144_72(int72_v * const res, __private int144_v q, const int72_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , __global uint * restrict modbasecase_debug
#endif
                  );
void mul_72(int72_v * const res, const int72_v a, const int72_t b);
void mul_72_144_no_low2(int144_v *const res, const int72_v a, const int72_v b);
void mul_72_144_no_low3(int144_v *const res, const int72_v a, const int72_v b);


/****************************************
 ****************************************
 * 24-bit based 72-bit barrett-kernels
 *
 ****************************************
 ****************************************/

/* from mul24.cl:
   int72_v sub_if_gte_72(const int72_v a, const int72_v b)
   void mul_72(int72_v * const res, const int72_v a, const int72_t b)
   void square_72_144(int144_v * const res, const int72_v a)
 */

void inc_if_ge_72(int72_v * const res, const int72_v a, const int72_v b)
{ /* if (a >= b) res++ */
  __private uint_v ge;

  ge = AS_UINT_V(a.d2 == b.d2);
  ge = AS_UINT_V(ge ? ((a.d1 == b.d1) ? (a.d0 >= b.d0) : (a.d1 > b.d1)) : (a.d2 > b.d2));

  res->d0 -= ge;
  res->d1 += res->d0 >> 24;
  res->d2 += res->d1 >> 24;
  res->d0 &= 0xFFFFFF;
  res->d1 &= 0xFFFFFF;
}

void mul_72_v(int72_v * const res, const int72_v a, const int72_v b)
/* res = (a * b) mod (2^72) */
{
  uint_v hi,lo;

  mul_24_48(&hi, &lo, a.d0, b.d0);
  res->d0 = lo;
  res->d1 = hi;

  mul_24_48(&hi, &lo, a.d1, b.d0);
  res->d1 += lo;
  res->d2 = hi;

  mul_24_48(&hi, &lo, a.d0, b.d1);
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

void mul_72_144_no_low2(int144_v * const res, const int72_v a, const int72_v b)
/*
res ~= a * b
res.d0 and res.d1 are NOT computed. Carry from res.d1 to res.d2 is ignored,
too. So the digits res.d{2-5} might differ from mul_72_144(). In
mul_72_144() are two carries from res.d1 to res.d2. So ignoring the digits
res.d0 and res.d1 the result of mul_72_144_no_low() is 0 to 2 lower than
of mul_72_144().
 */
{

  __private uint_v tmp;

  res->d2  = mul_hi(a.d1, b.d0) + mul_hi(a.d0, b.d1);

  tmp      = mul24(a.d2, b.d0);
  res->d3  = mad24(mul_hi(a.d2, b.d0), 256u, tmp >> 24);
  res->d2  = mad24(res->d2, 256u, tmp & 0xFFFFFF);

  tmp      = mul24(a.d1, b.d1);
  res->d3 += mad24(mul_hi(a.d1, b.d1), 256u, tmp >> 24);
  res->d2 += tmp & 0xFFFFFF;

  tmp      = mul24(a.d0, b.d2);
  res->d3 += mad24(mul_hi(a.d0, b.d2), 256u, tmp >> 24);
  res->d2 += tmp & 0xFFFFFF;

  tmp      = mul24(a.d2, b.d1);
  res->d4  = mad24(mul_hi(a.d2, b.d1), 256u, tmp >> 24);
  res->d3 += tmp & 0xFFFFFF;

  tmp      = mul24(a.d1, b.d2);
  res->d4 += mad24(mul_hi(a.d1, b.d2), 256u, tmp >> 24);
  res->d3 += tmp & 0xFFFFFF;

  tmp      = mul24(a.d2, b.d2);
  res->d5  = mad24(mul_hi(a.d2, b.d2), 256u, tmp >> 24);
  res->d4 += tmp & 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;

}


void mul_72_144_no_low3(int144_v * const res, const int72_v a, const int72_v b)
/*
res ~= a * b
res.d0, res.d1 and res.d2 are NOT computed. Carry to res.d3 is ignored,
too. So the digits res.d{3-5} might differ from mul_72_144(). In
mul_72_144() are four carries from res.d2 to res.d3. So ignoring the digits
res.d0, res.d1 and res.d2 the result of mul_72_144_no_low() is 0 to 4 lower
than of mul_72_144().
 */
{

  /* not needed for now, return ..._no_low_2 */

  mul_72_144_no_low2(res, a, b);
}


void shl_72(int72_v * const a)
/* shiftleft a one bit */
{
  /*
  a->d2 = (a->d2 << 1) + (a->d1 >> 23);
  a->d1 = ((a->d1 << 1) + (a->d0 >> 23)) & 0xFFFFFF;
  */
  a->d2 = mad24(a->d2, 2u, a->d1 >> 23);  // keep the higher bit here
  a->d1 = mad24(a->d1, 2u,  a->d0 >> 23) & 0xFFFFFF;
  a->d0 = (a->d0 << 1) & 0xFFFFFF;
}


void div_144_72(int72_v * const res, __private int144_v q, const int72_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , __global uint * restrict modbasecase_debug
#endif
)
/* res = q / n (integer division) */
{
  __private float_v qf;
  __private uint_v qi, tmp;
  __private int144_v nn;

/********** Step 1, Offset 2^51 (2*24 + 3) **********/
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d4);
  qf= qf * 2097152.0f;
//  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);  // test: not needed for large factors?
//  qf*= 2097152.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 3;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#0: qf=%#G, nf=%#G, *=%#G, qi=%d\n", qf.s0, nf.s0, qf.s0*nf.s0, qi.s0);
#endif

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_144_72#1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, res=%x:..:..\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0, res->d2.s0);
#endif

//  nn.d0=0;
//  nn.d1=0;
// nn = n * qi AND shiftleft 3 bits at once, carry is done later
  tmp    =  mul24(n.d0, qi);
  nn.d3  =  mad24(mul_hi(n.d0, qi), 2048u, (tmp >> 21));
//  nn.d3  = (mul_hi(n.d0, qi) << 11) | (tmp >> 21);
  nn.d2  = (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#1.1: nn=..:..:%x:%x:..:..\n",
        nn.d3.s0, nn.d2.s0);
#endif

  tmp    =  mul24(n.d1, qi);
//  nn.d4  = (mul_hi(n.d1, qi) << 11) | (tmp >> 21);
  nn.d4  =  mad24(mul_hi(n.d1, qi), 2048u, (tmp >> 21));
//  nn.d3 += (tmp << 3) & 0xFFFFFF;
  nn.d3  =  mad24((tmp & 0x1FFFFF), 8u, nn.d3);
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#1.2: nn=..:%x:%x:%x:..:..\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

  tmp    =  mul24(n.d2, qi);
//  nn.d5  = (mul_hi(n.d2, qi) << 11) | (tmp >> 21);
  nn.d5  =  mad24(mul_hi(n.d2, qi), 2048u, (tmp >> 21));
//  nn.d4 += (tmp << 3) & 0xFFFFFF;
  nn.d4  =  mad24((tmp & 0x1FFFFF), 8u, nn.d4);
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#1.3: nn=%x:%x:%x:%x:..:..\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif


/* do carry */
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;

  MODBASECASE_NN_BIG_ERROR(0xFFFFFF, 1, nn.d5, 1);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#1: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

/*  q = q - nn */
  q.d2 = q.d2 - nn.d2;
  q.d3 = q.d3 - nn.d3 + AS_UINT_V(q.d2 > 0xFFFFFF);
  q.d4 = q.d4 - nn.d4 + AS_UINT_V(q.d3 > 0xFFFFFF);
  q.d5 = q.d5 - nn.d5 + AS_UINT_V(q.d4 > 0xFFFFFF);
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 2, Offset 2^31 (1*24 + 7) **********/
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d4);
//  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);
  qf= qf * 16777216.0f * 131072.0f;


  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);

  res->d1 =  (qi <<  7) & 0xFFFFFF;
  res->d2 += qi >> 17;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_71#2: qf=%#G, nf=%#G, *=%#G, qi=%d\n", qf.s0, nf.s0, qf.s0*nf.s0, qi.s0);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"div_144_72#2: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, res=%x:%x:%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0, res->d2.s0, res->d1.s0, res->d0.s0);
#endif

//  nn.d0=0;
// nn = n * qi AND shiftleft 7 bits at once, carry is done later

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#2.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d0, qi);
//  nn.d2  = (mul_hi(n.d0, qi) << 15) | (tmp >> 17);
  nn.d2  =  mad24(mul_hi(n.d0, qi), 32768u, (tmp >> 17));
  nn.d1  = (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d1, qi);
//  nn.d3  = (mul_hi(n.d1, qi) << 15) | (tmp >> 17);
  nn.d3  =  mad24(mul_hi(n.d1, qi), 32768u, (tmp >> 17));
//  nn.d2 += (tmp << 7) & 0xFFFFFF;
  nn.d2  =  mad24((tmp & 0x1FFFF), 128u, nn.d2);

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d2, qi);
//  nn.d4  = (mul_hi(n.d2, qi) << 15) | (tmp >> 17);
  nn.d4  =  mad24(mul_hi(n.d2, qi), 32768u, (tmp >> 17));
//  nn.d3 += (tmp << 7) & 0xFFFFFF;
  nn.d3  =  mad24((tmp & 0x1FFFF), 128u, nn.d3);
#if (TRACE_KERNEL > 2) || defined(CHECKS_MODBASECASE)
  nn.d5=0;
#endif

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#2.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

/* do carry */
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
#if (TRACE_KERNEL > 2) || defined(CHECKS_MODBASECASE)
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#2: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
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
  q.d2 = q.d2 - nn.d2 + AS_UINT_V(q.d1 > 0xFFFFFF);
  q.d3 = q.d3 - nn.d3 + AS_UINT_V(q.d2 > 0xFFFFFF);
  q.d4 = q.d4 - nn.d4 + AS_UINT_V(q.d3 > 0xFFFFFF);
#ifdef CHECKS_MODBASECASE
  q.d5 = q.d5 - nn.d5 + AS_UINT_V(q.d4 > 0xFFFFFF);
#endif
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 3, Offset 2^11 (0*24 + 11) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);
//  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d2);
  qf= qf * 16777216.0f * 8192.0f;

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

  res->d0 = (qi << 11) & 0xFFFFFF;
  res->d1 += qi >> 13;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#3: qf=%#G, nf=%#G, *=%#G, qi=%d\n", qf.s0, nf.s0, qf.s0*nf.s0, qi.s0);
    // if (tid==TRACE_TID) printf((__constant char *)"div_144_72: qf=%#G, nf=%#G, qi=%d\n", -1.0e10f, 3.2e8f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"div_144_72#3: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, res=%x:%x:%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0, res->d2.s0, res->d1.s0, res->d0.s0);
#endif

//nn = n * qi, shiftleft is done later
/*  nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0); */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#3.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d0, qi);
//  nn.d1 = (mul_hi(n.d0, qi) << 8) | (tmp >> 24);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256u, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#3.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d1, qi);
//  nn.d2 = (mul_hi(n.d1, qi) << 8) | (tmp >> 24);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256u, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#3.2: nn=(%x:%x:)%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d2, qi);
//  nn.d3 = (mul_hi(n.d2, qi) << 8) | (tmp >> 24);
  nn.d3 = mad24(mul_hi(n.d2, qi), 256u, tmp >> 24);
  nn.d2 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#3.3: nn=(%x:%x:)%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  /* do carry */
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#3: before shl(11): nn=(%x:%x:)%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif
// shiftleft 11 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                             nn.d3>>13;
  nn.d3 = mad24(nn.d3 & 0x1FFF, 2048u, nn.d2>>13);
#else
  nn.d3 = mad24(nn.d3,          2048u, nn.d2>>13);	// we don't need to clear top bits here, this is done during q = q - nn
#endif
  nn.d2 = mad24(nn.d2 & 0x1FFF, 2048u, nn.d1>>13);
  nn.d1 = mad24(nn.d1 & 0x1FFF, 2048u, nn.d0>>13);
  nn.d0 = ((nn.d0 & 0x1FFF)<<11);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#3: nn=(%x:%x:)%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
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
  q.d1 = q.d1 - nn.d1 + AS_UINT_V(q.d0 > 0xFFFFFF);
  q.d2 = q.d2 - nn.d2 + AS_UINT_V(q.d1 > 0xFFFFFF);
  q.d3 = q.d3 - nn.d3 + AS_UINT_V(q.d2 > 0xFFFFFF);
#ifdef CHECKS_MODBASECASE
  q.d4 = q.d4 - nn.d4 + AS_UINT_V(q.d3 > 0xFFFFFF);
#endif
  q.d0 &= 0xFFFFFF;
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;

/********** Step 4, Offset 2^0 (0*24 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 5);
  MODBASECASE_NONZERO_ERROR(q.d4, 4, 4, 6);

  qf= CONVERT_FLOAT_V(q.d3);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d2);
//  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d1);
  qf= qf * 16777216.0f;

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  res->d0 += qi;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#4: qf=%#G, nf=%#G, *=%#G, qi=%d\n", qf.s0, nf.s0, qf.s0*nf.s0, qi.s0);
#endif

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_144_72#4: q=(%x:%x:)%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

  // skip the last part - it will change the result by one at most - we can live with a result that is off by one
  // but handle the missing carries
  res->d1 += res->d0 >> 24;
  res->d0 &= 0xFFFFFF;
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;
}


void mod_simple_72(int72_v * const res, const int72_v q, const int72_v n, const float_v nf
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
  __private int72_v nn;

  qf = CONVERT_FLOAT_V(q.d2);
  qf = mad(qf, 16777216.0f, CONVERT_FLOAT_V(q.d1));
//  qf = qf * 16777216.0f + CONVERT_FLOAT_V(q.d1);

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
  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 100, qi, 12);
  }
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mod_simple_72: q=%x:%x:%x, n=%x:%x:%x, nf=%G, qf=%G, qi=%x\n",
        q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, nf.s0, qf.s0, qi.s0);
#endif

  // qi < 8 bit, so no mul_hi is needed (+3% total speed)
  nn.d0 = mul24(n.d0, qi);
  nn.d1 = mad24(n.d1, qi, nn.d0 >> 24);
  nn.d2 = mad24(n.d2, qi, nn.d1 >> 24);
  nn.d0 &= 0xFFFFFF;
  nn.d1 &= 0xFFFFFF;


#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"mod_simple_72: nn=%x:%x:%x\n",
        nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  res->d0 = (q.d0 - nn.d0) & 0xFFFFFF;
  res->d1 = (q.d1 - nn.d1 + AS_UINT_V(nn.d0 > q.d0));
  res->d2 = (q.d2 - nn.d2 + AS_UINT_V(res->d1 > q.d1));
  res->d1 &= 0xFFFFFF;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"mod_simple_72: return %x:%x:%x\n",
        res->d2.s0, res->d1.s0, res->d0.s0);
#endif
}

__kernel void cl_barrett24_70(__private uint exp, const int72_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int144_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod and is precomputed on host ONCE.

bit_max64 is bit_max - 64! (1 .. 8)
*/
{
  __private int72_t exp72;
  __private int72_v a, u, f, k;
  __private int144_v b, tmp144;
  __private int72_v tmp72;
  __private float_v ff;
  __private uint bit_max48 = 16 + bit_max64; /* = bit_max - 48, used for bit shifting... */
  __private uint bit_max48_24 = 24 - bit_max48; /* = 72 - bit_max, used for bit shifting... */
  __private uint bit_max48_24_mult = 1 << bit_max48_24; /* = 2 ^ (72 - bit_max), used for bit shifting... */
  __private uint tid;
  __private uint_v t;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int144_t bb={0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp72.d2=0;exp72.d1=exp>>23;exp72.d0=(exp+exp)&0xFFFFFF;	// exp72 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett24_70: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, shift=%d\n",
        exp, exp72.d1, exp72.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, shiftcount);
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

  mul_24_48(&(a.d1), &(a.d0), t, 4620); // NUM_CLASSES
  k.d0 = k_base.d0 + a.d0;
  k.d1 = k_base.d1 + a.d1;
  k.d1 += k.d0 >> 24; k.d0 &= 0xFFFFFF;
  k.d2 = k_base.d2 + (k.d1 >> 24); k.d1 &= 0xFFFFFF;		// k = k + k_tab[tid] * NUM_CLASSES

  mul_72(&f, k, exp72);				// f = 2 * k * exp
  f.d0 += 1;				      	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett24_70: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif
/*
ff = f as float, needed in mod and div.
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= mad(ff, 16777216.0f, CONVERT_FLOAT_RTP_V(f.d1));
  // f.d0 not needed as d1 and d2 provide more than 23 bits precision

  ff= as_float(0x3f7ffffd) / ff;	// we rounded ff towards plus infinity, and round all other results towards zero.

  // OpenCL shifts 32-bit values by 31 at most. bit_max64 can be 0 .. 8
  tmp144.d5 = 256 << (bit_max64 <<1);     	// tmp144 = 2^(2*bit_max), may use the 2^144 bit
  tmp144.d4 = 0; tmp144.d3 = 0; tmp144.d2 = 0; tmp144.d1 = 0; tmp144.d0 = 0;

  div_144_72(&u,tmp144,f,ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
#ifdef CHECKS_MODBASECASE
                  , modbasecase_debug
#endif
                  );						// u = floor(tmp144 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett24_70: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

  //a.d0 = (bb.d2 >> bit_max48) + (bb.d3 << bit_max48_24) & 0xFFFFFF;
  //a.d1 = (bb.d3 >> bit_max48) + (bb.d4 << bit_max48_24) & 0xFFFFFF;
  //a.d2 = (bb.d4 >> bit_max48) + (bb.d5 << bit_max48_24) & 0xFFFFFF;
  a.d0 = mad24(bit_max48_24_mult, bb.d3, bb.d2 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bit_max48_24_mult, bb.d4, bb.d3 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bit_max48_24_mult, bb.d5, bb.d4 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)

  mul_72_144_no_low2(&tmp144, a, u);					// tmp144 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett24_70: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp144.d5.s0, tmp144.d4.s0, tmp144.d3.s0, tmp144.d2.s0);
#endif

//  a.d0 = (tmp144.d2 >> bit_max64) + (tmp144.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
//  a.d1 = (tmp144.d3 >> bit_max64) + (tmp144.d4 << bit_max64_32);
//  a.d2 = (tmp144.d4 >> bit_max64) + (tmp144.d5 << bit_max64_32);
  a.d0 = mad24(tmp144.d3, bit_max48_24_mult, tmp144.d2 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)
  a.d1 = mad24(tmp144.d4, bit_max48_24_mult, tmp144.d3 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)
  a.d2 = mad24(tmp144.d5, bit_max48_24_mult, tmp144.d4 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)

  mul_72_v(&tmp72, a, f);							// tmp72 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett24_70: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp72.d2.s0, tmp72.d1.s0, tmp72.d0.s0);
#endif
  // bb.d0-bb.d1 are zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp72.d0 = ( -tmp72.d0) & 0xFFFFFF;
  tmp72.d1 = ( -tmp72.d1 + AS_UINT_V(tmp72.d0 > 0)) & 0xFFFFFF;
  tmp72.d2 = ( bb.d2-tmp72.d2 + AS_UINT_V((tmp72.d0 | tmp72.d1) > 0)) & 0xFFFFFF; // if either d0 or d1 are non-zero we'll have to borrow

	 // we do not need the upper digits of b and tmp72 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett24_70: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp72.d2.s0, tmp72.d1.s0, tmp72.d0.s0);
#endif
#ifndef CHECKS_MODBASECASE
  mod_simple_72(&a, tmp72, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 6;
  if(bit_max64 == 1) limit = 8;						// bit_max == 65, due to decreased accuracy of mul_72_144_no_low2() above we need a higher threshold
  if(bit_max64 == 2) limit = 7;						// bit_max == 66, ...
  mod_simple_72(&a, tmp72, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max64, limit, modbasecase_debug);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett24_70: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp72.d2.s0, tmp72.d1.s0, tmp72.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_72_144(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
    a.d0 = mad24(b.d3, bit_max48_24_mult, b.d2 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)
    a.d1 = mad24(b.d4, bit_max48_24_mult, b.d3 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)
    a.d2 = mad24(b.d5, bit_max48_24_mult, b.d4 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)

    mul_72_144_no_low2(&tmp144, a, u);					// tmp144 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp144.d5.s0, tmp144.d4.s0, tmp144.d3.s0, tmp144.d2.s0);
#endif
    a.d0 = mad24(tmp144.d3, bit_max48_24_mult, tmp144.d2 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)
    a.d1 = mad24(tmp144.d4, bit_max48_24_mult, tmp144.d3 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)
    a.d2 = mad24(tmp144.d5, bit_max48_24_mult, tmp144.d4 >> bit_max48) & 0xFFFFFF;			// a = b / (2^bit_max)

    mul_72_v(&tmp72, a, f);						// tmp72 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp72.d2.s0, tmp72.d1.s0, tmp72.d0.s0);
#endif
    tmp72.d0 = (b.d0 - tmp72.d0) & 0xFFFFFF;
    tmp72.d1 = (b.d1 - tmp72.d1 + AS_UINT_V(tmp72.d0 > b.d0));
    tmp72.d2 = (b.d2 - tmp72.d2 + AS_UINT_V(tmp72.d1 > b.d1));
    tmp72.d1 &= 0xFFFFFF;
    tmp72.d2 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp72.d2.s0, tmp72.d1.s0, tmp72.d0.s0);
#endif
    if(exp&0x80000000)shl_72(&tmp72);					// "optional multiply by 2" in Prime 95 documentation, may use the 2^72 bit

#ifndef CHECKS_MODBASECASE
    mod_simple_72(&a, tmp72, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    if(bit_max64 == 1) limit = 8;					// bit_max == 65, due to decreased accuracy of mul_72_144_no_low2() above we need a higher threshold
    if(bit_max64 == 2) limit = 7;					// bit_max == 66, ...
    mod_simple_72(&a, tmp72, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max64, limit, modbasecase_debug);
#endif

    exp+=exp;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        exp, tmp72.d2.s0, tmp72.d1.s0, tmp72.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  }


#ifndef CHECKS_MODBASECASE
  a = sub_if_gte_72(a,f);	// final adjustment in case a >= f
#else
  tmp72 = sub_if_gte_72(a,f);
  a = sub_if_gte_72(tmp72,f);
  if( (tmp72.d2 != a.d2) || (tmp72.d1 != a.d1) || (tmp72.d0 != a.d0))
  {
    printf((__constant char *)"EEEEEK, final a was >= f\n");
  }
#endif

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett24_70: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf((__constant char *)"cl_barrett24_70: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
