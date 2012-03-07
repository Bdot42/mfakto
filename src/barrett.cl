/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)
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

/****************************************
 ****************************************
 * 32-bit-stuff for the 92/76-bit-barrett-kernel
 * included by main kernel file
 ****************************************
 ****************************************/


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

#ifdef CHECKS_MODBASECASE
// this check only works for single vector (i.e. no vector)
#if (BARRETT_VECTOR_SIZE != 1)
#pragma error "CHECKS_MODBASECASE only works with BARRETT_VECTOR_SIZE = 1"
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

#if (BARRETT_VECTOR_SIZE == 1)

typedef struct _int96_t
{
  uint d0,d1,d2;
}int96_t;

typedef struct _int192_t
{
  uint d0,d1,d2,d3,d4,d5;
}int192_t;

#define int_v int
#define uint_v uint
#define float_v float
#define CONVERT_FLOAT_V convert_float_rtz
#define CONVERT_UINT_V convert_uint
#define AS_UINT_V as_uint

#elif (BARRETT_VECTOR_SIZE == 2)
typedef struct _int96_t
{
  uint2 d0,d1,d2;
}int96_t;

typedef struct _int192_t
{
  uint2 d0,d1,d2,d3,d4,d5;
}int192_t;

#define int_v int2
#define uint_v uint2
#define float_v float2
#define CONVERT_FLOAT_V convert_float2_rtz
#define CONVERT_UINT_V convert_uint2
#define AS_UINT_V as_uint2

#elif (BARRETT_VECTOR_SIZE == 4)

typedef struct _int96_t
{
  uint4 d0,d1,d2;
}int96_t;

typedef struct _int192_t
{
  uint4 d0,d1,d2,d3,d4,d5;
}int192_t;

#define int_v int4
#define uint_v uint4
#define float_v float4
#define CONVERT_FLOAT_V convert_float4_rtz
#define CONVERT_UINT_V convert_uint4
#define AS_UINT_V as_uint4

#elif (BARRETT_VECTOR_SIZE == 8)
typedef struct _int96_t
{
  uint8 d0,d1,d2;
}int96_t;

typedef struct _int192_t
{
  uint8 d0,d1,d2,d3,d4,d5;
}int192_t;

#define int_v int8
#define uint_v uint8
#define float_v float8
#define CONVERT_FLOAT_V convert_float8_rtz
#define CONVERT_UINT_V convert_uint8
#define AS_UINT_V as_uint8

#elif (BARRETT_VECTOR_SIZE == 16)
#pragma error "Vector size 16 is so slow, don't use it. If you really want to, remove this pragma."
typedef struct _int96_t
{
  uint16 d0,d1,d2;
}int96_t;

typedef struct _int192_t
{
  uint16 d0,d1,d2,d3,d4,d5;
}int192_t;

#define int_v int16
#define uint_v uint16
#define float_v float16
#define CONVERT_FLOAT_V convert_float16_rtz
#define CONVERT_UINT_V convert_uint16
#define AS_UINT_V as_uint16

#else
#pragma error "invalid BARRETT_VECTOR_SIZE"
#endif

#ifndef CHECKS_MODBASECASE
void div_192_96(int96_t *res, int192_t q, int96_t n, float_v nf);
void div_160_96(int96_t *res, int192_t q, int96_t n, float_v nf);
#else
void div_192_96(int96_t *res, int192_t q, int96_t n, float_v nf, __global uint* modcasebase_debug);
void div_160_96(int96_t *res, int192_t q, int96_t n, float_v nf, __global uint* modcasebase_debug);
#endif
void mul_96(int96_t *res, int96_t a, int96_t b);
void mul_96_192_no_low2(int192_t *res, int96_t a, int96_t b);
void mul_96_192_no_low3(int192_t *res, int96_t a, int96_t b);


/****************************************
 ****************************************
 * 32-bit based 79- and 92-bit barrett-kernels
 *
 ****************************************
 ****************************************/

int96_t sub_if_gte_96(int96_t a, int96_t b)
/* return (a>b)?a-b:a */
{
  int96_t tmp;
  /* do the subtraction and use tmp.d2 to decide if the result is valid (if a was > b) */

  int_v carry= (b.d0 > a.d0);

  tmp.d0 = a.d0 - b.d0;
  tmp.d1 = a.d1 - b.d1 - AS_UINT_V(carry ? 1 : 0);
  carry   = (tmp.d1 > a.d1) || ((tmp.d1 == a.d1) && carry);
  tmp.d2 = a.d2 - b.d2 - AS_UINT_V(carry ? 1 : 0);

  tmp.d0 = (tmp.d2 > a.d2) ? a.d0 : tmp.d0;
  tmp.d1 = (tmp.d2 > a.d2) ? a.d1 : tmp.d1;
  tmp.d2 = (tmp.d2 > a.d2) ? a.d2 : tmp.d2;

  return tmp;
}

void inc_if_ge_96(int96_t *res, int96_t a, int96_t b)
{ /* if (a >= b) res++ */
  uint_v ge, carry;

  ge = AS_UINT_V(a.d2 == b.d2);
  ge = AS_UINT_V(ge ? ((a.d1 == b.d1) ? (a.d0 >= b.d0) : (a.d1 > b.d1)) : (a.d2 > b.d2));

  carry    = AS_UINT_V(ge ? 1 : 0);
  res->d0 += carry;
  carry    = AS_UINT_V((carry > res->d0)? 1 : 0);
  res->d1 += carry;
  res->d2 += AS_UINT_V((carry > res->d1)? 1 : 0);
}

void mul_96(int96_t *res, int96_t a, int96_t b)
/* res = a * b */
{
  uint_v tmp;

  res->d0  = a.d0 * b.d0;
  res->d1  = mul_hi(a.d0, b.d0);

  res->d2  = mul_hi(a.d1, b.d0);

  tmp = a.d1 * b.d0;
  res->d1 += tmp;
  res->d2 += AS_UINT_V((tmp > res->d1)? 1 : 0);

  res->d2 += mul_hi(a.d0, b.d1);

  tmp = a.d0 * b.d1;
  res->d1 += tmp;
  res->d2 += AS_UINT_V((tmp > res->d1)? 1 : 0);

  res->d2 += a.d0 * b.d2 + a.d1 * b.d1 + a.d2 * b.d0;
}


void mul_96_192_no_low2(int192_t *res, int96_t a, int96_t b)
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
  uint_v tmp;
  
  res->d2  = mul_hi(a.d1, b.d0);

  tmp      = mul_hi(a.d0, b.d1);
  res->d2 += tmp;
  res->d3  = AS_UINT_V((tmp > res->d2)? 1 : 0);

  tmp      = a.d2 * b.d0;
  res->d2 += tmp;
  res->d3 += AS_UINT_V((tmp > res->d2)? 1 : 0);

  tmp      = a.d1 * b.d1;
  res->d2 += tmp;
  res->d3 += AS_UINT_V((tmp > res->d2)? 1 : 0);

  tmp      = a.d0 * b.d2;
  res->d2 += tmp;
  res->d3 += AS_UINT_V((tmp > res->d2)? 1 : 0);


  tmp      = mul_hi(a.d2, b.d0);
  res->d3 += tmp;
  res->d4  = AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = mul_hi(a.d1, b.d1);
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = mul_hi(a.d0, b.d2);
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = a.d2 * b.d1;
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = a.d1 * b.d2;
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);


  tmp      = mul_hi(a.d2, b.d1);
  res->d4 += tmp;
  res->d5  = AS_UINT_V((tmp > res->d4)? 1 : 0);

  tmp      = mul_hi(a.d1, b.d2);
  res->d4 += tmp;
  res->d5 += AS_UINT_V((tmp > res->d4)? 1 : 0);

  tmp      = a.d2 * b.d2;
  res->d4 += tmp;
  res->d5 += AS_UINT_V((tmp > res->d4)? 1 : 0);


  res->d5 += mul_hi(a.d2, b.d2);
}


void mul_96_192_no_low3(int192_t *res, int96_t a, int96_t b)
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
  uint_v tmp;

  res->d3  = mul_hi(a.d2, b.d0);

  tmp      = mul_hi(a.d1, b.d1);
  res->d3 += tmp;
  res->d4  = AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = mul_hi(a.d0, b.d2);
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = a.d2 * b.d1;
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = a.d1 * b.d2;
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);


  tmp      = mul_hi(a.d2, b.d1);
  res->d4 += tmp;
  res->d5  = AS_UINT_V((tmp > res->d4)? 1 : 0);

  tmp      = mul_hi(a.d1, b.d2);
  res->d4 += tmp;
  res->d5 += AS_UINT_V((tmp > res->d4)? 1 : 0);

  tmp      = a.d2 * b.d2;
  res->d4 += tmp;
  res->d5 += AS_UINT_V((tmp > res->d4)? 1 : 0);


  res->d5 += mul_hi(a.d2, b.d2);
}


void square_96_192(int192_t *res, int96_t a)
/* res = a^2 = a.d0^2 + a.d1^2 + a.d2^2 + 2(a.d0*a.d1 + a.d0*a.d2 + a.d1*a.d2) */
{
/*
highest possible value for x * x is 0xFFFFFFF9
this occurs for x = {479772853, 1667710795, 2627256501, 3815194443}
Adding x*x to a few carries will not cascade the carry
*/
  uint_v tmp;

  res->d0  = a.d0 * a.d0;

  res->d1  = mul_hi(a.d0, a.d0);

  tmp      = a.d0 * a.d1;
  res->d1 += tmp;
  res->d2  = AS_UINT_V((tmp > res->d1)? 1 : 0);
  res->d1 += tmp;
  res->d2 += AS_UINT_V((tmp > res->d1)? 1 : 0);


  res->d2 += a.d1 * a.d1;  // no carry possible

  tmp      = mul_hi(a.d0, a.d1);
  res->d2 += tmp;
  res->d3  = AS_UINT_V((tmp > res->d2)? 1 : 0);
  res->d2 += tmp;
  res->d3 += AS_UINT_V((tmp > res->d2)? 1 : 0);

  tmp      = a.d0 * a.d2;
  res->d2 += tmp;
  res->d3 += AS_UINT_V((tmp > res->d2)? 1 : 0);
  res->d2 += tmp;
  res->d3 += AS_UINT_V((tmp > res->d2)? 1 : 0);


  tmp      = mul_hi(a.d1, a.d1);
  res->d3 += tmp;
  res->d4  = AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = mul_hi(a.d0, a.d2);
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = a.d1 * a.d2;
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);


  res->d4 += a.d2 * a.d2; // no carry possible

  tmp      = mul_hi(a.d1, a.d2);
  res->d4 += tmp;
  res->d5  = AS_UINT_V((tmp > res->d4)? 1 : 0);
  res->d4 += tmp;
  res->d5 += AS_UINT_V((tmp > res->d4)? 1 : 0);


  res->d5 += mul_hi(a.d2, a.d2);
}


void square_96_160(int192_t *res, int96_t a)
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
  uint_v tmp, TWOad2 = a.d2 << 1; // a.d2 < 2^16 so this always fits

  res->d0  = a.d0 * a.d0;

  res->d1  = mul_hi(a.d0, a.d0);

  tmp      = a.d0 * a.d1;
  res->d1 += tmp;
  res->d2  = AS_UINT_V((tmp > res->d1)? 1 : 0);
  res->d1 += tmp;
  res->d2 += AS_UINT_V((tmp > res->d1)? 1 : 0);


  res->d2 += a.d1 * a.d1;  // no carry possible

  tmp      = mul_hi(a.d0, a.d1);
  res->d2 += tmp;
  res->d3  = AS_UINT_V((tmp > res->d2)? 1 : 0);
  res->d2 += tmp;
  res->d3 += AS_UINT_V((tmp > res->d2)? 1 : 0);

  tmp      = a.d0 * TWOad2;  
  res->d2 += tmp;
  res->d3 += AS_UINT_V((tmp > res->d2)? 1 : 0);


  tmp      = mul_hi(a.d1, a.d1);
  res->d3 += tmp;
  res->d4  = AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = mul_hi(a.d0, TWOad2);
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);

  tmp      = a.d1 * TWOad2;
  res->d3 += tmp;
  res->d4 += AS_UINT_V((tmp > res->d3)? 1 : 0);


  res->d4 += a.d2 * a.d2; // no carry possible

  res->d4 += mul_hi(a.d1, TWOad2);
}


void shl_96(int96_t *a)
/* shiftleft a one bit */
{
  a->d2 = (a->d2 << 1) + (a->d1 >> 31);
  a->d1 = (a->d1 << 1) + (a->d0 >> 31);
  a->d0 = a->d0 << 1;
}


#undef DIV_160_96
#ifndef CHECKS_MODBASECASE
void div_192_96(int96_t *res, int192_t q, int96_t n, float_v nf)
#else
void div_192_96(int96_t *res, int192_t q, int96_t n, float_v nf, __global uint *modbasecase_debug)
#endif
/* res = q / n (integer division) */
{
  float_v qf;
  uint_v qi, tmp, carry;
  int192_t nn;
  int96_t tmp96;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d4);
#else
  #ifdef CHECKS_MODBASECASE
    q.d5 = 0;	// later checks in debug code will test if q.d5 is 0 or not but 160bit variant ignores q.d5
  #endif
  qf= CONVERT_FLOAT_V(q.d4);
#endif  
  qf*= 2097152.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 11;

// nn = n * qi
  nn.d2  = n.d0 * qi;
  nn.d3  = mul_hi(n.d0, qi);
  tmp    = n.d1 * qi;
  nn.d3 += tmp;
  nn.d4  = AS_UINT_V((tmp > nn.d3)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d4 += tmp;
#ifndef DIV_160_96
  nn.d5  = AS_UINT_V((tmp > nn.d4)? 1 : 0);
  tmp    = n.d2 * qi;
  nn.d4 += tmp;
  nn.d5 += AS_UINT_V((tmp > nn.d4)? 1 : 0);
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
  carry= AS_UINT_V((nn.d2 > q.d2)? 1 : 0);
  q.d2 = q.d2 - nn.d2;

  tmp  = q.d3 - nn.d3 - carry ;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

#ifndef DIV_160_96
  tmp  = q.d4 - nn.d4 - carry;
  carry= AS_UINT_V(((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)))? 1 : 0);
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d5 - carry;
#else
  q.d4 = q.d4 - nn.d4 - carry;
#endif
/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef DIV_160_96
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d4);
#else
  qf= CONVERT_FLOAT_V(q.d4);
#endif
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf*= 512.0f;

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);

  res->d1 =  qi << 23;
  res->d2 += qi >>  9;

// nn = n * qi
  nn.d1 = n.d0 * qi;
  nn.d2  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d2 += tmp;
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d3 += tmp;
  nn.d4  = AS_UINT_V((tmp > nn.d3)? 1 : 0);
  tmp    = n.d2* qi;
  nn.d3 += tmp;
  nn.d4 += AS_UINT_V((tmp > nn.d3)? 1 : 0);
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
  carry= AS_UINT_V((nn.d1 > q.d1) ? 1 : 0);
  q.d1 = q.d1 - nn.d1;

  tmp  = q.d2 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)))? 1 : 0);
  q.d2 = tmp;

  tmp  = q.d3 - nn.d3 - carry;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

#ifdef CHECKS_MODBASECASE
  tmp  = q.d4 - nn.d4 - carry;
  carry= AS_UINT_V(((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)))? 1 : 0);
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d5 - carry;
#else
  q.d4 = q.d4 - nn.d4 - carry;
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

  tmp     = (qi << 3);
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + (qi >> 29) + AS_UINT_V((tmp > res->d1)? 1 : 0);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;

// nn = n * qi
  
  nn.d1 = n.d0 * qi;
  nn.d2  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d2 += tmp;
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d3 += tmp;
  nn.d4  = AS_UINT_V((tmp > nn.d3)? 1 : 0);
  tmp    = n.d2* qi;
  nn.d3 += tmp;
  nn.d4 += AS_UINT_V((tmp > nn.d3)? 1 : 0);
  nn.d4 += mul_hi(n.d2, qi);

//  q = q - nn
  carry= AS_UINT_V((nn.d1 > q.d1) ? 1 : 0);
  q.d1 = q.d1 - nn.d1;

  tmp  = q.d2 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)))? 1 : 0);
  q.d2 = tmp;

  tmp  = q.d3 - nn.d3 - carry;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 - carry;

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 4);

  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d2);
  qf*= 131072.0f;
  
  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  tmp     = qi >> 17;
  res->d0 = qi << 15;
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + AS_UINT_V((tmp > res->d1)? 1 : 0);
  
// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V((tmp > nn.d1)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V((tmp > nn.d2)? 1 : 0);
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
  carry= AS_UINT_V((nn.d0 > q.d0) ? 1 : 0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - carry;
  carry= AS_UINT_V(((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1))) ? 1 : 0);
  q.d1 = tmp;

  tmp  = q.d2 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2))) ? 1 : 0);
  q.d2 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d3 = q.d3 - nn.d3 - carry;
#else
  tmp  = q.d3 - nn.d3 - carry;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3))) ? 1 : 0);
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 - carry;
#endif

/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 6);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= CONVERT_FLOAT_V(q.d3);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d2);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d1);
  
  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 += qi;
  carry    = AS_UINT_V((qi > res->d0)? 1 : 0);
  res->d1 += carry;
  res->d2 += AS_UINT_V((carry > res->d1)? 1 : 0);
  
// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V((tmp > nn.d1)? 1 : 0);
#ifndef CHECKS_MODBASECASE
  nn.d2 += mul_hi(n.d1, qi) + n.d2* qi;
#else
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V((tmp > nn.d2)? 1 : 0);
  nn.d3 += mul_hi(n.d2, qi);
#endif  

//  q = q - nn
  carry= AS_UINT_V((nn.d0 > q.d0) ? 1 : 0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - carry;
  carry= AS_UINT_V(((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1))) ? 1 : 0);
  q.d1 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d2 = q.d2 - nn.d2 - carry;
#else
  tmp  = q.d2 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2))) ? 1 : 0);
  q.d2 = tmp;

  q.d3 = q.d3 - nn.d3 - carry;
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
  inc_if_ge_96(res, tmp96, n);
}


#define DIV_160_96
#ifndef CHECKS_MODBASECASE
void div_160_96(int96_t *res, int192_t q, int96_t n, float_v nf)
#else
void div_160_96(int96_t *res, int192_t q, int96_t n, float_v nf, __global uint *modbasecase_debug)
#endif
/* res = q / n (integer division) */
/* the code of div_160_96() is an EXACT COPY of div_192_96(), the only
difference is that the 160bit version ignores the most significant
word of q (q.d5) because it assumes it is 0. This is controlled by defining
DIV_160_96 here. */
{
  float_v qf;
  uint_v qi, tmp, carry;
  int192_t nn;
  int96_t tmp96;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d4);
#else
  #ifdef CHECKS_MODBASECASE
    q.d5 = 0;	// later checks in debug code will test if q.d5 is 0 or not but 160bit variant ignores q.d5
  #endif
  qf= CONVERT_FLOAT_V(q.d4);
#endif  
  qf*= 2097152.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 11;

// nn = n * qi
  nn.d2  = n.d0 * qi;
  nn.d3  = mul_hi(n.d0, qi);
  tmp    = n.d1 * qi;
  nn.d3 += tmp;
  nn.d4  = AS_UINT_V((tmp > nn.d3)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d4 += tmp;
#ifndef DIV_160_96
  nn.d5  = AS_UINT_V((tmp > nn.d4)? 1 : 0);
  tmp    = n.d2 * qi;
  nn.d4 += tmp;
  nn.d5 += AS_UINT_V((tmp > nn.d4)? 1 : 0);
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
  carry= AS_UINT_V((nn.d2 > q.d2)? 1 : 0);
  q.d2 = q.d2 - nn.d2;

  tmp  = q.d3 - nn.d3 - carry ;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

#ifndef DIV_160_96
  tmp  = q.d4 - nn.d4 - carry;
  carry= AS_UINT_V(((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)))? 1 : 0);
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d5 - carry;
#else
  q.d4 = q.d4 - nn.d4 - carry;
#endif
/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef DIV_160_96
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d4);
#else
  qf= CONVERT_FLOAT_V(q.d4);
#endif
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf*= 512.0f;

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);

  res->d1 =  qi << 23;
  res->d2 += qi >>  9;

// nn = n * qi
  nn.d1 = n.d0 * qi;
  nn.d2  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d2 += tmp;
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d3 += tmp;
  nn.d4  = AS_UINT_V((tmp > nn.d3)? 1 : 0);
  tmp    = n.d2* qi;
  nn.d3 += tmp;
  nn.d4 += AS_UINT_V((tmp > nn.d3)? 1 : 0);
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
  carry= AS_UINT_V((nn.d1 > q.d1) ? 1 : 0);
  q.d1 = q.d1 - nn.d1;

  tmp  = q.d2 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)))? 1 : 0);
  q.d2 = tmp;

  tmp  = q.d3 - nn.d3 - carry;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

#ifdef CHECKS_MODBASECASE
  tmp  = q.d4 - nn.d4 - carry;
  carry= AS_UINT_V(((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)))? 1 : 0);
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d5 - carry;
#else
  q.d4 = q.d4 - nn.d4 - carry;
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

  tmp     = (qi << 3);
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + (qi >> 29) + AS_UINT_V((tmp > res->d1)? 1 : 0);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;

// nn = n * qi
  
  nn.d1 = n.d0 * qi;
  nn.d2  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d2 += tmp;
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d3 += tmp;
  nn.d4  = AS_UINT_V((tmp > nn.d3)? 1 : 0);
  tmp    = n.d2* qi;
  nn.d3 += tmp;
  nn.d4 += AS_UINT_V((tmp > nn.d3)? 1 : 0);
  nn.d4 += mul_hi(n.d2, qi);

//  q = q - nn
  carry= AS_UINT_V((nn.d1 > q.d1) ? 1 : 0);
  q.d1 = q.d1 - nn.d1;

  tmp  = q.d2 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)))? 1 : 0);
  q.d2 = tmp;

  tmp  = q.d3 - nn.d3 - carry;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 - carry;

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 4);

  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d2);
  qf*= 131072.0f;
  
  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  tmp     = qi >> 17;
  res->d0 = qi << 15;
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + AS_UINT_V((tmp > res->d1)? 1 : 0);
  
// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V((tmp > nn.d1)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V((tmp > nn.d2)? 1 : 0);
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
  carry= AS_UINT_V((nn.d0 > q.d0) ? 1 : 0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - carry;
  carry= AS_UINT_V(((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1))) ? 1 : 0);
  q.d1 = tmp;

  tmp  = q.d2 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2))) ? 1 : 0);
  q.d2 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d3 = q.d3 - nn.d3 - carry;
#else
  tmp  = q.d3 - nn.d3 - carry;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3))) ? 1 : 0);
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 - carry;
#endif

/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 6);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= CONVERT_FLOAT_V(q.d3);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d2);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d1);
  
  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 += qi;
  carry    = AS_UINT_V((qi > res->d0)? 1 : 0);
  res->d1 += carry;
  res->d2 += AS_UINT_V((carry > res->d1)? 1 : 0);
  
// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V((tmp > nn.d1)? 1 : 0);
#ifndef CHECKS_MODBASECASE
  nn.d2 += mul_hi(n.d1, qi) + n.d2* qi;
#else
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V((tmp > nn.d2)? 1 : 0);
  nn.d3 += mul_hi(n.d2, qi);
#endif  

//  q = q - nn
  carry= AS_UINT_V((nn.d0 > q.d0) ? 1 : 0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - carry;
  carry= AS_UINT_V(((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1))) ? 1 : 0);
  q.d1 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d2 = q.d2 - nn.d2 - carry;
#else
  tmp  = q.d2 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2))) ? 1 : 0);
  q.d2 = tmp;

  q.d3 = q.d3 - nn.d3 - carry;
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
  inc_if_ge_96(res, tmp96, n);
}

#undef DIV_160_96




void mod_simple_96(int96_t *res, int96_t q, int96_t n, float_v nf
#if (TRACE_KERNEL > 1)
                  , __private uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , int bit_max64, unsigned int limit, __global uint *modbasecase_debug
#endif
)
/*
res = q mod n
used for refinement in barrett modular multiplication
assumes q < 6n (6n includes "optional mul 2")
*/
{
  float_v qf;
  uint_v qi, tmp, carry;
  int96_t nn;

  qf = CONVERT_FLOAT_V(q.d2);
  qf = qf * 4294967296.0f + CONVERT_FLOAT_V(q.d1);
  
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
    if (tid==TRACE_TID) printf("mod_simple_96: q=%x:%x:%x, n=%x:%x:%x, nf=%.9G, qf=%f, qi=%x\n",
        q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, nf, qf, qi);
#endif

  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V((tmp > nn.d1)? 1 : 0);
  nn.d2 += mul_hi(n.d1, qi) + n.d2* qi;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mod_simple_96: nn=%x:%x:%x\n",
        nn.d2, nn.d1, nn.d0);
#endif

  carry= AS_UINT_V((nn.d0 > q.d0) ? 1 : 0);
  res->d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 - carry;
  carry= AS_UINT_V(((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1))) ? 1 : 0);
  res->d1 = tmp;

  res->d2 = q.d2 - nn.d2 - carry;
}

#ifndef CHECKS_MODBASECASE
__kernel void mfakto_cl_barrett92(__private uint exp, __private int96_1t k_base, __global uint *k_tab,
         __private int shiftcount, __private int192_1t bb, __global uint *RES, __private int bit_max64)
#else
__kernel void mfakto_cl_barrett92(__private uint exp, __private int96_1t k_base, __global uint *k_tab,
         __private int shiftcount, __private int192_1t bb, __global uint *RES, __private int bit_max64, __global uint *modbasecase_debug)
#endif
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int96_1t exp96;
  __private int96_t a, u, f, k;
  __private int192_t b, tmp192;
  __private int96_t tmp96;
  __private float_v ff;
  __private int bit_max64_32 = 32 - bit_max64; /* used for bit shifting... */
  __private uint tid;
  __private uint_v t, tmp, carry;

	tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * BARRETT_VECTOR_SIZE;

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp<<1;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_barrett92: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
#endif

#if (BARRETT_VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (BARRETT_VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
#elif (BARRETT_VECTOR_SIZE == 4)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
  t.w  = k_tab[tid+3];
#elif (BARRETT_VECTOR_SIZE == 8)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
#elif (BARRETT_VECTOR_SIZE == 16)
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
  tmp  = t * 4620; // NUM_CLASSES
  k.d0 = k_base.d0 + tmp;
  k.d1 = k_base.d1 + mul_hi(t, 4620) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
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
  f.d2 +=  AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mul_hi(k.d1, exp96.d0) + AS_UINT_V((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we always underestimate the quotient
        
        
  // OpenCL shifts 32-bit values by 31 at most
  tmp192.d5 = (0x40000000 >> (31 - bit_max64)) >> (31 - bit_max64);	// tmp192 = 2^(2*bit_max)
  tmp192.d4 = (1 << bit_max64) << bit_max64;   // 1 << (b << 1) = (1 << b) << b
  tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);						// u = floor(tmp192 / f)
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(tmp192 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: u=%x:%x:%x, ff=%G\n",
        u.d2, u.d1, u.d0, ff);
#endif

  a.d0 = (bb.d2 >> bit_max64) + (bb.d3 << bit_max64_32);			// a = b / (2^bit_max)
  a.d1 = (bb.d3 >> bit_max64) + (bb.d4 << bit_max64_32);
  a.d2 = (bb.d4 >> bit_max64) + (bb.d5 << bit_max64_32);

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
  carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = bb.d0 - tmp96.d0;

  tmp  = bb.d1 - tmp96.d1 - carry;
  carry= AS_UINT_V(((tmp > bb.d1) || (carry && AS_UINT_V(tmp == bb.d1))) ? 1 : 0);
  tmp96.d1 = tmp;

  tmp96.d2 = bb.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett92: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp96.d2, tmp96.d1, tmp96.d0);
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
  mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max64, limit, modbasecase_debug);
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
    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
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
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max64, limit, modbasecase_debug);
#endif

    exp<<=1;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("loopend: exp=%d, tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        exp, tmp96.d2, tmp96.d1, tmp96.d0, f.d2, f.d1, f.d0, a.d2, a.d1, a.d0 );
#endif
  }


#ifndef CHECKS_MODBASECASE
  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#else
  tmp96 = sub_if_gte_96(a,f);
  a = sub_if_gte_96(tmp96,f);
  if( (tmp96.d2 != a.d2) || (tmp96.d1 != a.d1) || (tmp96.d0 != a.d0))
  {
    printf("EEEEEK, final a was >= f\n");
  }
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x \n",
         a.d2, a.d1, a.d0 );
#endif
  

/* finally check if we found a factor and write the factor to RES[] */
#if (BARRETT_VECTOR_SIZE == 1)
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
#elif (BARRETT_VECTOR_SIZE == 2)
  EVAL_RES_b(x)
  EVAL_RES_b(y)
#elif (BARRETT_VECTOR_SIZE == 4)
  EVAL_RES_b(x)
  EVAL_RES_b(y)
  EVAL_RES_b(z)
  EVAL_RES_b(w)
#elif (BARRETT_VECTOR_SIZE == 8)
  EVAL_RES_b(s0)
  EVAL_RES_b(s1)
  EVAL_RES_b(s2)
  EVAL_RES_b(s3)
  EVAL_RES_b(s4)
  EVAL_RES_b(s5)
  EVAL_RES_b(s6)
  EVAL_RES_b(s7)
#elif (BARRETT_VECTOR_SIZE == 16)
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

#ifndef CHECKS_MODBASECASE
__kernel void mfakto_cl_barrett79(__private uint exp, __private int96_1t k_base, __global uint *k_tab,
         __private int shiftcount, __private int192_1t bb, __global uint *RES, __private int bit_max64)
#else
__kernel void mfakto_cl_barrett79(__private uint exp, __private int96_1t k_base, __global uint *k_tab,
         __private int shiftcount, __private int192_1t bb, __global uint *RES, __private int bit_max64, __global uint *modbasecase_debug)
#endif
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private int96_1t exp96;
  __private int96_t a, u, f, k;
  __private int192_t tmp192, b;
  __private int96_t tmp96;
  __private float_v ff;
  __private uint tid;
  __private uint_v t, tmp, carry;

	tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * BARRETT_VECTOR_SIZE;

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp<<1;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_barrett79: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0, k.d2, k.d1, k.d0);
#endif

#if (BARRETT_VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (BARRETT_VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
#elif (BARRETT_VECTOR_SIZE == 4)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
  t.w  = k_tab[tid+3];
#elif (BARRETT_VECTOR_SIZE == 8)
  t.s0 = k_tab[tid];
  t.s1 = k_tab[tid+1];
  t.s2 = k_tab[tid+2];
  t.s3 = k_tab[tid+3];
  t.s4 = k_tab[tid+4];
  t.s5 = k_tab[tid+5];
  t.s6 = k_tab[tid+6];
  t.s7 = k_tab[tid+7];
#elif (BARRETT_VECTOR_SIZE == 16)
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
  tmp  = t * 4620; // NUM_CLASSES
  k.d0 = k_base.d0 + tmp;
  k.d1 = k_base.d1 + mul_hi(t, 4620) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */

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
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mul_hi(k.d1, exp96.d0) + AS_UINT_V((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett79: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount);
#endif

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

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

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

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

  carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = bb.d0 - tmp96.d0;

  tmp  = bb.d1 - tmp96.d1 - carry;
  carry= AS_UINT_V(((tmp > bb.d1) || (carry && AS_UINT_V(tmp == bb.d1))) ? 1 : 0);
  tmp96.d1 = tmp;

  tmp96.d2 = bb.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

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
  mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
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

    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
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
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2, tmp96.d1, tmp96.d0, f.d2, f.d1, f.d0, a.d2, a.d1, a.d0 );
#endif

    exp<<=1;
  }

  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x \n",
         a.d2, a.d1, a.d0 );
#endif

  
/* finally check if we found a factor and write the factor to RES[] */
#if (BARRETT_VECTOR_SIZE == 1)
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
#elif (BARRETT_VECTOR_SIZE == 2)
  EVAL_RES_b(x)
  EVAL_RES_b(y)
#elif (BARRETT_VECTOR_SIZE == 4)
  EVAL_RES_b(x)
  EVAL_RES_b(y)
  EVAL_RES_b(z)
  EVAL_RES_b(w)
#elif (BARRETT_VECTOR_SIZE == 8)
  EVAL_RES_b(s0)
  EVAL_RES_b(s1)
  EVAL_RES_b(s2)
  EVAL_RES_b(s3)
  EVAL_RES_b(s4)
  EVAL_RES_b(s5)
  EVAL_RES_b(s6)
  EVAL_RES_b(s7)
#elif (BARRETT_VECTOR_SIZE == 16)
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

