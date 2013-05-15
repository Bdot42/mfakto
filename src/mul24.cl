/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2012  Oliver Weihe (o.weihe@t-online.de)
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

/* This file will be included TWICE by the main kernel file, once with
   _63BIT_MUL24_K defined and once undefined. Some definitions are shared and
   only included once. Do not define typedefs etc for the second time this file
   is included for the small factor kernel */

#ifndef _63BIT_MUL24_K

/***********************************************
 * vector implementations of all sizes
 * for the 71-bit-kernel
 *
 ***********************************************/


void mul_24_48(uint_v * const res_hi, uint_v * const res_lo, const uint_v a, const uint_v b)
/* res_hi*(2^24) + res_lo = a * b */
{ // PERF: inline its use
/* thats how it should be, but the mul24_hi is missing ...
  *res_lo = mul24(a,b) & 0xFFFFFF;
  *res_hi = mul24_hi(a,b) >> 8;       // PERF: check for mul24_hi
  */
  *res_lo  = mul24(a,b);
//  *res_hi  = (mul_hi(a,b) << 8) | (*res_lo >> 24);       
  *res_hi  = mad24(mul_hi(a,b), 256u, (*res_lo >> 24));       
  *res_lo &= 0xFFFFFF;
}


int72_v sub_if_gte_72(const int72_v a, const int72_v b)
/* return (a>b)?a-b:a */
{
  int72_v tmp;
  /* do the subtraction and use tmp.d2 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0) & 0xFFFFFF;
  tmp.d1 = (a.d1 - b.d1 + AS_UINT_V(b.d0 > a.d0));
  tmp.d2 = (a.d2 - b.d2 + AS_UINT_V(tmp.d1 > a.d1));
  tmp.d1&= 0xFFFFFF;

  /* tmp valid if tmp.d2 <= a.d2 (separately for each part of the vector) */
  tmp.d0 = (tmp.d2 > a.d2) ? a.d0 : tmp.d0;
  tmp.d1 = (tmp.d2 > a.d2) ? a.d1 : tmp.d1;
  tmp.d2 = (tmp.d2 > a.d2) ? a.d2 : tmp.d2;  //  & 0xFFFFFF not necessary as tmp.d4 is <= a.d4

  return tmp;
}

void mul_72(int72_v * const res, const int72_v a, const int72_t b)
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


void square_72_144(int144_v * const res, const int72_v a)
/* res = a^2 */
{
  __private uint_v tmp;

  tmp      =  mul24(a.d0, a.d0);
//  res->d1  = (mul_hi(a.d0, a.d0) << 8) | (tmp >> 24);
  res->d1  =  mad24(mul_hi(a.d0, a.d0), 256u, (tmp >> 24));
  res->d0  =  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
//  res->d2  = (mul_hi(a.d1, a.d0) << 9) | (tmp >> 23);
  res->d2  =  mad24(mul_hi(a.d1, a.d0), 512u, (tmp >> 23));
//  res->d1 += (tmp << 1) & 0xFFFFFF;
  res->d1  =  mad24(tmp & 0x7FFFFF, 2u, res->d1);

  tmp      =  mul24(a.d2, a.d0);
  res->d3  =  mad24(mul_hi(a.d2, a.d0), 512u, (tmp >> 23));
//  res->d2 += (tmp << 1) & 0xFFFFFF;
  res->d2  =  mad24(tmp & 0x7FFFFF, 2u, res->d2);
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 +=  mad24(mul_hi(a.d1, a.d1), 256u, (tmp >> 24));
  res->d2 +=  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  =  mad24(mul_hi(a.d2, a.d1), 512u, (tmp >> 23));
//  res->d3 += (tmp << 1) & 0xFFFFFF;
  res->d3  =  mad24(tmp & 0x7FFFFF, 2u, res->d3);

  tmp      =  mul24(a.d2, a.d2);
  res->d5  =  mad24(mul_hi(a.d2, a.d2), 256u, (tmp >> 24));
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


void square_72_144_shl(int144_v * const res, const int72_v a)
/* res = 2* a^2 */
{
  __private uint_v tmp;

  tmp      =  mul24(a.d0, a.d0);
  res->d1  =  mad24(mul_hi(a.d0, a.d0), 512u, (tmp >> 23));
  res->d0  = (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
  res->d2  =  mad24(mul_hi(a.d1, a.d0), 1024u, (tmp >> 22));
//  res->d1 += (tmp << 2) & 0xFFFFFF;
  res->d1  =  mad24(tmp & 0x3FFFFF, 4u, res->d1);

  tmp      =  mul24(a.d2, a.d0);
  res->d3  =  mad24(mul_hi(a.d2, a.d0), 1024u, (tmp >> 22));
//  res->d2 += (tmp << 2) & 0xFFFFFF;
  res->d2  =  mad24(tmp & 0x3FFFFF, 4u, res->d2);
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 +=  mad24(mul_hi(a.d1, a.d1), 512u, (tmp >> 23));
//  res->d2 += (tmp << 1) & 0xFFFFFF;
  res->d2  =  mad24(tmp & 0x7FFFFF, 2u, res->d2);
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  =  mad24(mul_hi(a.d2, a.d1), 1024u, (tmp >> 22));
//  res->d3 += (tmp << 2) & 0xFFFFFF;
  res->d3  =  mad24(tmp & 0x3FFFFF, 4u, res->d3);

  tmp      =  mul24(a.d2, a.d2);
  res->d5  =  mad24(mul_hi(a.d2, a.d2), 512u, (tmp >> 23));
//  res->d4 += (tmp << 1) & 0xFFFFFF;
  res->d4  =  mad24(tmp & 0x7FFFFF, 2u, res->d4);

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
#endif

#ifdef _63BIT_MUL24_K
void mod_144_64
#else
void mod_144_72
#endif
    (int72_v * const res, __private int144_v q, const int72_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                   , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                   , __global uint * restrict modbasecase_debug
#endif

)
/* res = q mod n */
{
  __private float_v  qf;
  __private uint_v   qi, tmp;
  __private int144_v nn; // ={0,0,0,0};  // PERF: initialization needed?

/********** Step 1, Offset 2^51 (2*24 + 3) **********/
#ifdef _63BIT_MUL24_K
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d4);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);
  qf*= 2097152.0f;
#else
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d4);
  qf= qf * 16777216.0f * 2097152.0f;
//  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);  // test: not needed for large factors?
//  qf*= 2097152.0f;
#endif

  qi=CONVERT_UINT_V(qf*nf);
  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mod_%d%d_%d#1: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.s0, nf.s0, qf.s0*nf.s0, qi.s0);
#endif

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

//  nn.d0=0;
//  nn.d1=0;
// nn = n * qi AND shiftleft 3 bits at once, carry is done later
  tmp    =  mul24(n.d0, qi);
  nn.d3  =  mad24(mul_hi(n.d0, qi), 2048u, (tmp >> 21));
//  nn.d3  = (mul_hi(n.d0, qi) << 11) | (tmp >> 21);
  nn.d2  = (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#1.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d1, qi);
//  nn.d4  = (mul_hi(n.d1, qi) << 11) | (tmp >> 21);
  nn.d4  =  mad24(mul_hi(n.d1, qi), 2048u, (tmp >> 21));
//  nn.d3 += (tmp << 3) & 0xFFFFFF;
  nn.d3  =  mad24((tmp & 0x1FFFFF), 8u, nn.d3);
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#1.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d2, qi);
//  nn.d5  = (mul_hi(n.d2, qi) << 11) | (tmp >> 21);
  nn.d5  =  mad24(mul_hi(n.d2, qi), 2048u, (tmp >> 21));
//  nn.d4 += (tmp << 3) & 0xFFFFFF;
  nn.d4  =  mad24((tmp & 0x1FFFFF), 8u, nn.d4);
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#1.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif


/* do carry */
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;

//  MODBASECASE_NN_BIG_ERROR(0xFFFFFF, 1, nn.d5, 1);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#1: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d2 = __sub_cc (q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5); */
  q.d2 = q.d2 - nn.d2;
  q.d3 = q.d3 - nn.d3 + AS_UINT_V(q.d2 > 0xFFFFFF);
  q.d4 = q.d4 - nn.d4 + AS_UINT_V(q.d3 > 0xFFFFFF);
  q.d5 = q.d5 - nn.d5 + AS_UINT_V(q.d4 > 0xFFFFFF);
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 2, Offset 2^31 (1*24 + 7) **********/
#ifdef _63BIT_MUL24_K
  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d2);
  qf*= 131072.0f;
#else
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d4);
//  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);
  qf= qf * 16777216.0f * 16777216.0f * 131072.0f;
#endif

  qi=CONVERT_UINT_V(qf*nf);
  MODBASECASE_QI_ERROR(1<<22, 2, qi, 2);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mod_%d%d_%d#2: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.s0, nf.s0, qf.s0*nf.s0, qi.s0);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#2: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

//  nn.d0=0;
// nn = n * qi AND shiftleft 7 bits at once, carry is done later

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#2.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d0, qi);
//  nn.d2  = (mul_hi(n.d0, qi) << 15) | (tmp >> 17);
  nn.d2  =  mad24(mul_hi(n.d0, qi), 32768u, (tmp >> 17));
  nn.d1  = (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp    =  mul24(n.d1, qi);
//  nn.d3  = (mul_hi(n.d1, qi) << 15) | (tmp >> 17);
  nn.d3  =  mad24(mul_hi(n.d1, qi), 32768u, (tmp >> 17));
//  nn.d2 += (tmp << 7) & 0xFFFFFF;
  nn.d2  =  mad24((tmp & 0x1FFFF), 128u, nn.d2);

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
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
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#2.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

/* do carry */
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
#if (TRACE_KERNEL > 2) || defined(CHECKS_MODBASECASE)
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#2: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
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
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 3);
#ifdef _63BIT_MUL24_K
  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d2);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d1);
  qf*= 8192.0f;
#else
  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d3);
//  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d2);
  qf= qf * 16777216.0f * 16777216.0f * 8192.0f;
#endif

  qi=CONVERT_UINT_V(qf*nf);
  MODBASECASE_QI_ERROR(1<<22, 3, qi, 4);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mod_%d%d_%d#3: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.s0, nf.s0, qf.s0*nf.s0, qi.s0);
    // if (tid==TRACE_TID) printf((__constant char *)"mod_144_72: qf=%#G, nf=%#G, qi=%d\n", -1.0e10f, 3.2e8f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#3: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

//nn = n * qi, shiftleft is done later
/*  nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0); */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#3.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d0, qi);
//  nn.d1 = (mul_hi(n.d0, qi) << 8) | (tmp >> 24);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256u, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#3.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d1, qi);
//  nn.d2 = (mul_hi(n.d1, qi) << 8) | (tmp >> 24);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256u, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#3.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d2, qi);
//  nn.d3 = (mul_hi(n.d2, qi) << 8) | (tmp >> 24);
  nn.d3 = mad24(mul_hi(n.d2, qi), 256u, tmp >> 24);
  nn.d2 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#3.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  /* do carry */
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#3: before shl(11): nn=%x:%x:%x:%x:%x:%x\n",
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
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
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

#ifdef _63BIT_MUL24_K
  qf= CONVERT_FLOAT_V(q.d3);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d2);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d1);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d0);
#else
  qf= CONVERT_FLOAT_V(q.d3);
  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d2);
//  qf= qf * 16777216.0f + CONVERT_FLOAT_V(q.d1);
  qf= qf * 16777216.0f * 16777216.0f;
#endif

  qi=CONVERT_UINT_V(qf*nf);
  MODBASECASE_QI_ERROR(1<<22, 4, qi, 7);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mod_%d%d_%d#4: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf.s0, nf.s0, qf.s0*nf.s0, qi.s0);
    //if (tid==TRACE_TID) printf((__constant char *)"mod_144_72: qf=%#G, nf=%#G, qi=%d\n", qf, nf, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
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
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#4.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d0, qi);
  nn.d1 = mad24(mul_hi(n.d0, qi), 256u, tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#4.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  tmp   = mul24(n.d1, qi);
  nn.d2 = mad24(mul_hi(n.d1, qi), 256u, tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#4.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

#ifndef CHECKS_MODBASECASE  
  nn.d2 = mad24(n.d2, qi, nn.d2);
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;  // carry
#else
  tmp   = mul24(n.d2, qi);
  nn.d3 = mad24(mul_hi(n.d2, qi), 256u, tmp >> 24);
  nn.d2 += tmp & 0xFFFFFF;
// do carry
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#4.3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
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
  q.d1 = q.d1 - nn.d1 + AS_UINT_V(q.d0 > 0xFFFFFF);
  q.d2 = q.d2 - nn.d2 + AS_UINT_V(q.d1 > 0xFFFFFF);
#ifdef CHECKS_MODBASECASE  
  q.d3 = q.d3 - nn.d3 + AS_UINT_V(q.d2 > 0xFFFFFF);
#endif

  res->d0 = q.d0 & 0xFFFFFF;
  res->d1 = q.d1 & 0xFFFFFF;
  res->d2 = q.d2 & 0xFFFFFF;

  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 8);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 9);
  MODBASECASE_NONZERO_ERROR(q.d3, 5, 3, 10);

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5.s0, q.d4.s0, q.d3.s0, res->d2.s0, res->d1.s0, res->d0.s0, n.d2.s0, n.d1.s0, n.d0.s0, qi.s0);
#endif

}

#ifdef _63BIT_MUL24_K
__kernel void mfakto_cl_63
#else
__kernel void mfakto_cl_71
#endif
      (__private uint exp, const int72_t k_base, const __global uint * restrict k_tab, const int shiftcount
#ifdef WA_FOR_CATALYST11_10_BUG
                           , const uint8 b_in
#else
                           , __private int144_t b_in
#endif
                           , __global uint * restrict RES
#ifdef CHECKS_MODBASECASE
                           , __global uint * restrict modbasecase_debug
#endif
)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  __private int72_t  exp72;
  __private int72_v  k;  
  __private int72_v  a;       // result of the modulo
  __private int144_v b;       // result of the squaring;
  __private int72_v  f;       // the factor(s) to be tested
  __private int      tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;
  __private float_v  ff;
  __private uint_v   t;

  exp72.d2=0;exp72.d1=exp>>23;exp72.d0=(exp+exp)&0xFFFFFF;	// exp72 = 2 * exp
  k.d0 = k_base.d0; k.d1 = k_base.d1; k.d2 = k_base.d2;   // widen to vectored "k"
#ifdef WA_FOR_CATALYST11_10_BUG
  b.d0 = 0; b.d1 = b_in.s1; b.d2 = b_in.s2;
  b.d3 = b_in.s3; b.d4 = b_in.s4; b.d5 = b_in.s5;
#else
  b.d0 = 0; b.d1 = b_in.d1; b.d2 = b_in.d2;
  b.d3 = b_in.d3; b.d4 = b_in.d4; b.d5 = b_in.d5;
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
  k.d0 += a.d0;
  k.d1 += a.d1;
  k.d1 += k.d0 >> 24; k.d0 &= 0xFFFFFF;
  k.d2 += k.d1 >> 24; k.d1 &= 0xFFFFFF;		// k = k + k_tab[tid] * NUM_CLASSES

  mul_72(&f, k, exp72);				// f = 2 * k * exp
  f.d0 += 1;				      	// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_144_72().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_V(f.d2);
  ff= mad(ff, 16777216.0f, CONVERT_FLOAT_V(f.d1));
#ifdef _63BIT_MUL24_K
  ff= mad(ff, 16777216.0f, CONVERT_FLOAT_V(f.d0));
#else // if f>48 bit then d2 and d1 provide enough precision
  ff= ff * 16777216.0f;
#endif

//  ff=0.9999997f/ff;
//  ff=__int_as_float(0x3f7ffffc) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
  ff=as_float(0x3f7ffffb) / ff;	// just a little bit below 1.0f so we always underestimate the quotient
 
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_71: tid=%d: p=%x, *2 =%x:%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d, b=%x:%x:%x:%x:%x:%x\n",
                              tid, exp, exp72.d1, exp72.d0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0);
#endif

#ifdef _63BIT_MUL24_K
  mod_144_64
#else
  mod_144_72
#endif
      (&a,b,f,ff
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
  if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_71: exp=%x,  %x:%x:%x ^2 (shl:%d) = %x:%x:%x:%x:%x:%x\n",
                              exp, a.d2.s0, a.d1.s0, a.d0.s0, (exp&0x80000000?1:0), b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0);
#endif

#ifdef _63BIT_MUL24_K
    mod_144_64
#else
    mod_144_72
#endif
      (&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
#ifdef CHECKS_MODBASECASE  
                   , modbasecase_debug
#endif
                   );			// a = b mod f
    exp+=exp;
  }
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_71 result: f=%x:%x:%x, a=%x:%x:%x\n",
                              f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif

  a=sub_if_gte_72(a,f);

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#ifdef _63BIT_MUL24_K
    if ((f.d2|f.d1)!=0 || f.d0 != 1)
    {
#endif
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf((__constant char *)"mfakto_cl_71: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2, f.d1, f.d0, k.d2, k.d1, k.d0);
#endif

    tid=ATOMIC_INC(RES[0]);
    if(tid<10)				/* limit to 10 factors per class */
    {
      RES[tid*3 + 1]=f.d2;
      RES[tid*3 + 2]=f.d1;
      RES[tid*3 + 3]=f.d0;
    }
#ifdef _63BIT_MUL24_K
    }
#endif
  }
#elif (VECTOR_SIZE == 2)
  EVAL_RES(x)
  EVAL_RES(y)
#elif (VECTOR_SIZE == 3)
  EVAL_RES(x)
  EVAL_RES(y)
  EVAL_RES(z)
#elif (VECTOR_SIZE == 4)
  EVAL_RES(x)
  EVAL_RES(y)
  EVAL_RES(z)
  EVAL_RES(w)
#elif (VECTOR_SIZE == 8)
  EVAL_RES(s0)
  EVAL_RES(s1)
  EVAL_RES(s2)
  EVAL_RES(s3)
  EVAL_RES(s4)
  EVAL_RES(s5)
  EVAL_RES(s6)
  EVAL_RES(s7)
#elif (VECTOR_SIZE == 16)
  EVAL_RES(s0)
  EVAL_RES(s1)
  EVAL_RES(s2)
  EVAL_RES(s3)
  EVAL_RES(s4)
  EVAL_RES(s5)
  EVAL_RES(s6)
  EVAL_RES(s7)
  EVAL_RES(s8)
  EVAL_RES(s9)
  EVAL_RES(sa)
  EVAL_RES(sb)
  EVAL_RES(sc)
  EVAL_RES(sd)
  EVAL_RES(se)
  EVAL_RES(sf)
#endif
}

