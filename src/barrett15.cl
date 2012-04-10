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

Version 0.11
*/

/****************************************
 ****************************************
 * 15-bit-stuff for the 90/75/60-bit-barrett-kernel
 * included by main kernel file
 ****************************************
 ****************************************/


// 60-bit
#define EVAL_RES_c(comp) \
  if((a.d3.comp|a.d2.comp|a.d1.comp)==0 && a.d0.comp==1) \
  { \
      tid=ATOMIC_INC(RES[0]); \
      if(tid<10) \
      { \
    /*    RES[tid*3 + 1]=f.d2.comp;  unused */  \
        RES[tid*3 + 2]=mad24(f.d3.comp,0x8000u, f.d2.comp); \
        RES[tid*3 + 3]=mad24(f.d1.comp,0x8000u, f.d0.comp); \
      } \
  }

// 75-bit
#define EVAL_RES_d(comp) \
  if((a.d4.comp|a.d3.comp|a.d2.comp|a.d1.comp)==0 && a.d0.comp==1) \
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
#define EVAL_RES_e(comp) \
  if((a.d5.comp|a.d4.comp|a.d3.comp|a.d2.comp|a.d1.comp)==0 && a.d0.comp==1) \
  { \
      tid=ATOMIC_INC(RES[0]); \
      if(tid<10) \
      { \
        RES[tid*3 + 1]=mad24(f.d5.comp,0x8000u, f.d4.comp); \
        RES[tid*3 + 2]=mad24(f.d3.comp,0x8000u, f.d2.comp); \
        RES[tid*3 + 3]=mad24(f.d1.comp,0x8000u, f.d0.comp); \
      } \
  }

typedef struct _int60_t
{
  uint d0,d1,d2,d3;
}int60_t;

typedef struct _int120_t
{
  uint d0,d1,d2,d3,d4,d5,d6,d7;
}int120_t;

typedef struct _int75_t
{
  uint d0,d1,d2,d3,d4;
}int75_t;

typedef struct _int150_t
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_t;

typedef struct _int90_t
{
  uint d0,d1,d2,d3,d4,d5;
}int90_t;

typedef struct _int180_t
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_t;

#if (BARRETT_VECTOR_SIZE == 1)

typedef struct _int60_v
{
  uint d0,d1,d2,d3;
}int60_v;

typedef struct _int120_v
{
  uint d0,d1,d2,d3,d4,d5,d6,d7;
}int120_v;

typedef struct _int75_v
{
  uint d0,d1,d2,d3,d4;
}int75_v;

typedef struct _int150_v
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_v;

typedef struct _int90_v
{
  uint d0,d1,d2,d3,d4,d5;
}int90_v;

typedef struct _int180_v
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_v;

#define int_v int
#define uint_v uint
#define float_v float
#define CONVERT_FLOAT_V convert_float
#define CONVERT_UINT_V convert_uint
#define AS_UINT_V as_uint

#elif (BARRETT_VECTOR_SIZE == 2)
typedef struct _int60_v
{
  uint2 d0,d1,d2,d3;
}int60_v;

typedef struct _int120_v
{
  uint2 d0,d1,d2,d3,d4,d5,d6,d7;
}int120_v;

typedef struct _int75_v
{
  uint2 d0,d1,d2,d3,d4;
}int75_v;

typedef struct _int150_v
{
  uint2 d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_v;

typedef struct _int90_v
{
  uint2 d0,d1,d2,d3,d5,d6;
}int90_v;

typedef struct _int180_v
{
  uint2 d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_v;

#define int_v int2
#define uint_v uint2
#define float_v float2
#define CONVERT_FLOAT_V convert_float2
#define CONVERT_UINT_V convert_uint2
#define AS_UINT_V as_uint2

#elif (BARRETT_VECTOR_SIZE == 4)
typedef struct _int60_v
{
  uint4 d0,d1,d2,d3;
}int60_v;

typedef struct _int120_v
{
  uint4 d0,d1,d2,d3,d4,d5,d6,d7;
}int120_v;

typedef struct _int75_v
{
  uint4 d0,d1,d2,d3,d4;
}int75_v;

typedef struct _int150_v
{
  uint4 d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_v;

typedef struct _int90_v
{
  uint4 d0,d1,d2,d3,d4,d5;
}int90_v;

typedef struct _int180_v
{
  uint4 d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_v;

#define int_v int4
#define uint_v uint4
#define float_v float4
#define CONVERT_FLOAT_V convert_float4
#define CONVERT_UINT_V convert_uint4
#define AS_UINT_V as_uint4

#elif (BARRETT_VECTOR_SIZE == 8)
typedef struct _int60_v
{
  uint8 d0,d1,d2,d3;
}int60_v;

typedef struct _int120_v
{
  uint8 d0,d1,d2,d3,d4,d5,d6,d7;
}int120_v;

typedef struct _int75_v
{
  uint8 d0,d1,d2,d3,d4;
}int75_v;

typedef struct _int150_v
{
  uint8 d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_v;

typedef struct _int90_v
{
  uint8 d0,d1,d2,d3,d4,d5;
}int90_v;

typedef struct _int180_v
{
  uint8 d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_v;

#define int_v int8
#define uint_v uint8
#define float_v float8
#define CONVERT_FLOAT_V convert_float8
#define CONVERT_UINT_V convert_uint8
#define AS_UINT_V as_uint8

#elif (BARRETT_VECTOR_SIZE == 16)
//# error "Vector size 16 is so slow, don't use it. If you really want to, remove this pragma."
typedef struct _int60_v
{
  uint16 d0,d1,d2,d3;
}int60_v;

typedef struct _int120_v
{
  uint16 d0,d1,d2,d3,d4,d5,d6,d7;
}int120_v;

typedef struct _int75_v
{
  uint16 d0,d1,d2,d3,d4;
}int75_v;

typedef struct _int150_v
{
  uint16 d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_v;

typedef struct _int90_v
{
  uint16 d0,d1,d2,d3,d4,d5;
}int90_v;

typedef struct _int180_v
{
  uint16 d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_v;

#define int_v int16
#define uint_v uint16
#define float_v float16
#define CONVERT_FLOAT_V convert_float16
#define CONVERT_UINT_V convert_uint16
#define AS_UINT_V as_uint16

#else
# error "invalid BARRETT_VECTOR_SIZE"
#endif

void mul_60(int60_v * const res, const int60_v a, const int60_v b);
void mul_60_120_no_low2(int120_v *const res, const int60_v a, const int60_v b);
void mul_60_120_no_low3(int120_v *const res, const int60_v a, const int60_v b);

void mul_75(int75_v * const res, const int75_v a, const int75_v b);
void mul_75_150_no_low2(int150_v *const res, const int75_v a, const int75_v b);
void mul_75_150_no_low3(int150_v *const res, const int75_v a, const int75_v b);

void mul_90(int90_v * const res, const int90_v a, const int90_v b);
void mul_90_180_no_low2(int180_v *const res, const int90_v a, const int90_v b);
void mul_90_180_no_low3(int180_v *const res, const int90_v a, const int90_v b);



/****************************************
 ****************************************
 * 15-bit based 60-bit barrett-kernels
 *
 ****************************************
 ****************************************/

int60_v sub_if_gte_60(const int60_v a, const int60_v b)
/* return (a>b)?a-b:a */
{
  int60_v tmp;
  /* do the subtraction and use tmp.d2 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0) & 0x7FFF;
  tmp.d1 = (a.d1 - b.d1 - AS_UINT_V((b.d0 > a.d0) ? 1 : 0));
  tmp.d2 = (a.d2 - b.d2 - AS_UINT_V((tmp.d1 > a.d1) ? 1 : 0));
  tmp.d3 = (a.d3 - b.d3 - AS_UINT_V((tmp.d2 > a.d2) ? 1 : 0));
  tmp.d1&= 0x7FFF;
  tmp.d2&= 0x7FFF;

  tmp.d0 = (tmp.d3 > a.d3) ? a.d0 : tmp.d0;
  tmp.d1 = (tmp.d3 > a.d3) ? a.d1 : tmp.d1;
  tmp.d2 = (tmp.d3 > a.d3) ? a.d2 : tmp.d2;
  tmp.d3 = (tmp.d3 > a.d3) ? a.d3 : tmp.d3 & 0x7FFF;

  return tmp;
}

void inc_if_ge_60(int60_v * const res, const int60_v a, const int60_v b)
{ /* if (a >= b) res++ */
  __private uint_v ge;

  ge = AS_UINT_V(a.d3 == b.d3 ? 1 : 0);
  ge = AS_UINT_V(ge ? ((a.d2 == b.d2) ? ((a.d1 == b.d1) ? (a.d0 >= b.d0) : (a.d1 > b.d1)) : (a.d2 > b.d2)) : (a.d3 > b.d3));

  res->d0 += AS_UINT_V(ge ? 1 : 0);
  res->d1 += AS_UINT_V((res->d0 > 0x7FFF) ? 1 : 0);
  res->d2 += AS_UINT_V((res->d1 > 0x7FFF) ? 1 : 0);
  res->d3 += AS_UINT_V((res->d2 > 0x7FFF) ? 1 : 0);
  res->d0 &= 0x7FFF;
  res->d1 &= 0x7FFF;
  res->d2 &= 0x7FFF;
  res->d3 &= 0x7FFF;
}

void mul_60(int60_v * const res, const int60_v a, const int60_v b)
/* res = a * b */
{
  res->d0 = mul24(a.d0, b.d0);

  res->d1 = mad24(a.d1, b.d0, res->d0 >> 15);
  res->d1 = mad24(a.d0, b.d1, res->d1);
  res->d0 &= 0x7FFF;

  res->d2 = mad24(a.d2, b.d0, res->d1 >> 15);
  res->d2 = mad24(a.d1, b.d1, res->d2);
  res->d2 = mad24(a.d0, b.d2, res->d2);
  res->d1 &= 0x7FFF;

  res->d3 = mad24(a.d3, b.d0, res->d2 >> 15);
  res->d3 = mad24(a.d2, b.d1, res->d3);
  res->d3 = mad24(a.d1, b.d2, res->d3);
  res->d3 = mad24(a.d0, b.d3, res->d3);
  res->d2 &= 0x7FFF;
  res->d3 &= 0x7FFF;

  return;

}


void mul_60_120_no_low2(int120_v * const res, const int60_v a, const int60_v b)
/*
res ~= a * b
res.d0 and res.d1 are NOT computed. Carry from res.d1 to res.d2 is ignored,
too. So the digits res.d{2-5} might differ from mul_96_192(). In
mul_96_192() are two carries from res.d1 to res.d2. So ignoring the digits
res.d0 and res.d1 the result of mul_96_192_no_low() is 0 to 2 lower than
of mul_96_192().
 */
{
  /* this is the exact multiplication including d0-d2 - optimize later */
  // carry is always immediately evaluated: quite slow but can run in shorts

  __private uint_v tmp;
  
  // d0 * d0
  tmp = mul24(a.d0, b.d0);
  res->d0  = tmp & 0x7FFF;
  res->d1  = (tmp >> 15);
  // d0 * d1
  tmp = mul24(a.d0, b.d1); // PERF:? mad24(a.d0, b.d1, res->d1);
  res->d1 += tmp & 0x7FFF;
  res->d2  = (tmp >> 15) + AS_UINT_V((res->d1 > 0x7FFF) ? 1 : 0);
  res->d1 &= 0x7FFF;
  // d0 * d2
  tmp = mul24(a.d0, b.d2); // PERF:? mad24(a.d0, b.d2, res->d2);
  res->d2 += tmp & 0x7FFF;
  res->d3  = (tmp >> 15) + AS_UINT_V((res->d2 > 0x7FFF) ? 1 : 0);
  res->d2 &= 0x7FFF;
  // d0 * d3
  tmp = mul24(a.d0, b.d3); // PERF:? mad24(a.d0, b.d3, res->d3);
  res->d3 += tmp & 0x7FFF;
  res->d4  = (tmp >> 15) + AS_UINT_V((res->d3 > 0x7FFF) ? 1 : 0);
  res->d3 &= 0x7FFF;
  // d3 * d1
  tmp = mul24(a.d3, b.d1); // PERF:? mad24(a.d3, b.d1, res->d4);
  res->d4 += tmp & 0x7FFF;
  res->d5  = (tmp >> 15) + AS_UINT_V((res->d4 > 0x7FFF) ? 1 : 0);
  res->d4 &= 0x7FFF;
  // d3 * d2
  tmp = mul24(a.d2, b.d3); // PERF:? mad24(a.d2, b.d3, res->d5);
  res->d5 += tmp & 0x7FFF;
  res->d6  = (tmp >> 15) + AS_UINT_V((res->d5 > 0x7FFF) ? 1 : 0);
  res->d5 &= 0x7FFF;
  // d3 * d3
  tmp = mul24(a.d3, b.d3); // PERF:? mad24(a.d3, b.d3, res->d6);
  res->d6 += tmp & 0x7FFF;
  res->d7  = (tmp >> 15) + AS_UINT_V((res->d6 > 0x7FFF) ? 1 : 0);
  res->d6 &= 0x7FFF;
  // d1 * d0
  tmp = mul24(a.d1, b.d0); // PERF:? mad24(a.d1, b.d0, res->d1);
  res->d1 += tmp & 0x7FFF;
  res->d2 += (tmp >> 15) + AS_UINT_V((res->d1 > 0x7FFF) ? 1 : 0);
  res->d1 &= 0x7FFF;
  // d1 * d1
  tmp = mul24(a.d1, b.d1); // PERF:? mad24(a.d1, b.d1, res->d2);
  // d2 already has 2 15-bit values and a carry: delay the addition until after
  // d2's carry is processed
  res->d3 += (tmp >> 15) + AS_UINT_V((res->d2 > 0x7FFF) ? 1 : 0);
  res->d2 &= 0x7FFF;
  res->d2 += tmp & 0x7FFF;
  // d1 * d2
  tmp = mul24(a.d1, b.d2); // PERF:? mad24(a.d1, b.d2, res->d3);
  res->d4 += (tmp >> 15) + AS_UINT_V((res->d3 > 0x7FFF) ? 1 : 0);
  res->d3 &= 0x7FFF;
  res->d3 += (tmp & 0x7FFF) + AS_UINT_V((res->d2 > 0x7FFF) ? 1 : 0);
  res->d2 &= 0x7FFF;
  // d1 * d3
  tmp = mul24(a.d1, b.d3); // PERF:? mad24(a.d1, b.d3, res->d3);
  res->d5 += (tmp >> 15) + AS_UINT_V((res->d4 > 0x7FFF) ? 1 : 0);
  res->d4 &= 0x7FFF;
  res->d4 += (tmp & 0x7FFF) + AS_UINT_V((res->d3 > 0x7FFF) ? 1 : 0);
  res->d3 &= 0x7FFF;
  // d2 * d3
  tmp = mul24(a.d2, b.d3); // PERF:? mad24(a.d2, b.d3, res->d5);
  res->d6 += (tmp >> 15) + AS_UINT_V((res->d5 > 0x7FFF) ? 1 : 0);
  res->d5 &= 0x7FFF;
  res->d5 += (tmp & 0x7FFF) + AS_UINT_V((res->d4 > 0x7FFF) ? 1 : 0);
  res->d4 &= 0x7FFF;
  // carries to make room in d5
  res->d7 += AS_UINT_V((res->d6 > 0x7FFF) ? 1 : 0);
  res->d6 &= 0x7FFF;
  res->d6 += AS_UINT_V((res->d5 > 0x7FFF) ? 1 : 0);
  res->d5 &= 0x7FFF;
  // d2 * d0
  tmp = mul24(a.d2, b.d0); // PERF:? mad24(a.d2, b.d0, res->d2);
  res->d2 += tmp & 0x7FFF;
  res->d3 += (tmp >> 15) + AS_UINT_V((res->d2 > 0x7FFF) ? 1 : 0);
  res->d2 &= 0x7FFF;
  // d2 * d1
  tmp = mul24(a.d2, b.d1); // PERF:? mad24(a.d2, b.d1, res->d3);
  res->d4 += (tmp >> 15) + AS_UINT_V((res->d3 > 0x7FFF) ? 1 : 0);
  res->d3 &= 0x7FFF;
  res->d3 += tmp & 0x7FFF;
  // d2 * d2
  tmp = mul24(a.d2, b.d2); // PERF:? mad24(a.d2, b.d2, res->d4);
  res->d5 += (tmp >> 15) + AS_UINT_V((res->d4 > 0x7FFF) ? 1 : 0);
  res->d4 &= 0x7FFF;
  res->d4 += (tmp & 0x7FFF) + AS_UINT_V((res->d3 > 0x7FFF) ? 1 : 0);
  res->d3 &= 0x7FFF;
  res->d6 += AS_UINT_V((res->d5 > 0x7FFF) ? 1 : 0);
  res->d5 &= 0x7FFF;
  res->d5 += AS_UINT_V((res->d4 > 0x7FFF) ? 1 : 0);
  res->d4 &= 0x7FFF;
  // d3 * d0
  tmp = mul24(a.d3, b.d0); // PERF:? mad24(a.d3, b.d0, res->d3);
  res->d3 += tmp & 0x7FFF;
  res->d4 += (tmp >> 15) + AS_UINT_V((res->d3 > 0x7FFF) ? 1 : 0);
  res->d3 &= 0x7FFF;
  //  final carries
  res->d5 += AS_UINT_V((res->d4 > 0x7FFF) ? 1 : 0);
  res->d4 &= 0x7FFF;
  res->d6 += AS_UINT_V((res->d5 > 0x7FFF) ? 1 : 0);
  res->d5 &= 0x7FFF;
  res->d7 += AS_UINT_V((res->d6 > 0x7FFF) ? 1 : 0);
  res->d6 &= 0x7FFF;
}


void mul_60_120_no_low3(int120_v * const res, const int60_v a, const int60_v b)
/*
res ~= a * b
res.d0, res.d1 and res.d2 are NOT computed. Carry to res.d3 is ignored,
too. So the digits res.d{3-5} might differ from mul_96_192(). In
mul_96_192() are four carries from res.d2 to res.d3. So ignoring the digits
res.d0, res.d1 and res.d2 the result of mul_96_192_no_low() is 0 to 4 lower
than of mul_96_192().
 */
{
  /* for now, use the complete implementation, optimize later */
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // this should allow for the fastest mul60: 16x mul24/mad24, 7x shift, 7x and

  res->d0 = mul24(a.d0, b.d0);

  res->d1 = mad24(a.d1, b.d0, res->d0 >> 15);
  res->d1 = mad24(a.d0, b.d1, res->d1);
  res->d0 &= 0x7FFF;

  res->d2 = mad24(a.d2, b.d0, res->d1 >> 15);
  res->d2 = mad24(a.d1, b.d1, res->d2);
  res->d2 = mad24(a.d0, b.d2, res->d2);
  res->d1 &= 0x7FFF;

  res->d3 = mad24(a.d3, b.d0, res->d2 >> 15);
  res->d3 = mad24(a.d2, b.d1, res->d3);
  res->d3 = mad24(a.d1, b.d2, res->d3);
  res->d3 = mad24(a.d0, b.d3, res->d3);
  res->d2 &= 0x7FFF;

  res->d4 = mad24(a.d3, b.d1, res->d3 >> 15);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
  res->d3 &= 0x7FFF;

  res->d5 = mad24(a.d3, b.d2, res->d4 >> 15);
  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d4 &= 0x7FFF;

  res->d6 = mad24(a.d3, b.d3, res->d5 >> 15);
  res->d5 &= 0x7FFF;

  res->d7 = res->d6 >> 15;
  res->d6 &= 0x7FFF;
}


void square_60_120(int120_v * const res, const int60_v a)
/* res = a^2 = a.d0^2 + a.d1^2 + a.d2^2 + a.d3^2 +
         2(a.d0*a.d1 + a.d0*a.d2 + a.d0*a.d3 + a.d1*a.d2 + a.d1*a.d3 + a.d2*a.d3)
       = a.d0^2 + a.d1^2 + a.d2^2 + a.d3^2 +
         2(a.d0(a.d1+a.d2) + a.d1(a.d2+a.d3) + a.d3(a.d0+a.d2))
   */
{
/*
highest possible value for x * x is 0xFFFFFFF9
this occurs for x = {479772853, 1667710795, 2627256501, 3815194443}
Adding x*x to a few carries will not cascade the carry
*/


  /* for now, use the complete implementation, optimize later */

  mul_60_120_no_low3(res, a, a);
}


void shl_60(int60_v * const a)
/* shiftleft a one bit */
{
  a->d3 = mad24(a->d3, 2u, a->d2 >> 14); // leave the extra top bit
  a->d2 = mad24(a->d2, 2u, a->d1 >> 14) & 0x7FFF;
  a->d1 = mad24(a->d1, 2u, a->d0 >> 14) & 0x7FFF;
  a->d0 = (a->d0 << 1) & 0x7FFF;
}


void div_120_60(int60_v * const res, __private int120_v q, const int60_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , __global uint * restrict modbasecase_debug
#endif
)/* res = q / n (integer division) */
{
  __private float_v qf;
  __private uint_v qi, qil, qih;
  __private int120_v nn;
  __private int60_v tmp60;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("div_120_60#0: q=%x:%x:%x:%x:%x:%x:%x:%x, n=%x:%x:%x:%x, nf=%#G\n",
        q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d3.s0, n.d2.s0, n.d1.s0, n.d0.s0, nf.s0);
#endif

/********** Step 1, Offset 2^40 (2*15 + 10) **********/
  qf= CONVERT_FLOAT_V(mad24(q.d7, 32768u, q.d6));
  qf= qf * 32768.0f * 32768.0f + CONVERT_FLOAT_V(mad24(q.d5, 32768u, q.d4));
  qf*= 32.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d3 = (qi >> 5) & 0x7FFF;
  res->d2 = (qi << 10) & 0x7FFF;
  qil = qi & 0x7FFF;
  qih = (qi >> 15) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("div_120_60#1: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:..:..\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d3.s0, res->d2.s0);
#endif

  /*******************************************************/

// nn = n * qi
  nn.d2  = mul24(n.d0, qil);
  nn.d3  = mad24(n.d0, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#1.1: nn=..:..:..:..:%x:%x:..:..\n",
        nn.d3.s0, nn.d2.s0);
#endif

  nn.d3  = mad24(n.d1, qil, nn.d3);
  nn.d4  = mad24(n.d1, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#1.2: nn=..:..:..:%x:%x:%x:..:..\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

  nn.d4  = mad24(n.d2, qil, nn.d4);
  nn.d5  = mad24(n.d2, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#1.3: nn=..:..:%x:%x:%x:%x:..:..\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

  nn.d5  = mad24(n.d3, qil, nn.d5);
  nn.d6  = mad24(n.d3, qih, nn.d5 >> 15);
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_120_60#1.4: nn=..:%x:%x:%x:%x:%x:..:..\n",
        nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

// now shift-left 10 bits
  nn.d7  = nn.d6 >> 5;  // PERF: not needed as it will be gone anyway after sub
  nn.d6  = mad24(nn.d6 & 0x1F, 1024u, nn.d5 >> 5);
  nn.d5  = mad24(nn.d5 & 0x1F, 1024u, nn.d4 >> 5);
  nn.d4  = mad24(nn.d4 & 0x1F, 1024u, nn.d3 >> 5);
  nn.d3  = mad24(nn.d3 & 0x1F, 1024u, nn.d2 >> 5);
  nn.d2  = (nn.d2 & 0x1F) << 10;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_120_60#1.5: nn=%x:%x:%x:%x:%x:%x:..:..\n",
        nn.d7.s0, nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

//  q = q - nn
  q.d2 = q.d2 - nn.d2;
  q.d3 = q.d3 - nn.d3 - AS_UINT_V((q.d2 > 0x7FFF)?1:0);
  q.d4 = q.d4 - nn.d4 - AS_UINT_V((q.d3 > 0x7FFF)?1:0);
  q.d5 = q.d5 - nn.d5 - AS_UINT_V((q.d4 > 0x7FFF)?1:0);
  q.d6 = q.d6 - nn.d6 - AS_UINT_V((q.d5 > 0x7FFF)?1:0);
  q.d7 = q.d7 - nn.d7 - AS_UINT_V((q.d6 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("div_120_60#1.6: q=%x:%x:%x:%x:%x:%x:..:..\n",
        q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0);
#endif

/********** Step 2, Offset 2^20 (1*15 + 5) **********/
  MODBASECASE_NONZERO_ERROR(q.d7, 2, 7, 1);

  qf= CONVERT_FLOAT_V(mad24(q.d6, 32768u, q.d5));
  qf= qf * 32768.0f * 32768.0f + CONVERT_FLOAT_V(mad24(q.d4, 32768u, q.d3));
  qf*= 1024.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 2);

  res->d2 += (qi >> 10);
  res->d1  = (qi << 5) & 0x7FFF;
  res->d3 += (res->d2) >> 15;   // PERF: do carry for res at the very end
  res->d2 &= 0x7FFF;

  qil = qi & 0x7FFF;
  qih = (qi >> 15) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("div_120_60#2: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:..\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d3.s0, res->d2.s0, res->d1.s0);
#endif

// nn = n * qi
  nn.d1  = mul24(n.d0, qil);
  nn.d2  = mad24(n.d0, qih, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#2.1: nn=..:..:..:..:..:%x:%x:..\n",
        nn.d2.s0, nn.d1.s0);
#endif

  nn.d2  = mad24(n.d1, qil, nn.d2);
  nn.d3  = mad24(n.d1, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#2.2: nn=..:..:..:..:%x:%x:%x:..\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif

  nn.d3  = mad24(n.d2, qil, nn.d3);
  nn.d4  = mad24(n.d2, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#2.3: nn=..:..:..:%x:%x:%x:%x:..\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif

  nn.d4  = mad24(n.d3, qil, nn.d4);
  nn.d5  = mad24(n.d3, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_120_60#2.4: nn=..:..:%x:%x:%x:%x:%x:..\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif

// now shift-left 5 bits
  nn.d6  = nn.d5 >> 10;  // PERF: not needed as it will be gone anyway after sub
  nn.d5  = mad24(nn.d5 & 0x3FF, 32u, nn.d4 >> 10);
  nn.d4  = mad24(nn.d4 & 0x3FF, 32u, nn.d3 >> 10);
  nn.d3  = mad24(nn.d3 & 0x3FF, 32u, nn.d2 >> 10);
  nn.d2  = mad24(nn.d2 & 0x3FF, 32u, nn.d1 >> 10);
  nn.d1  = (nn.d1 & 0x3FF) << 5;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_120_60#2.5: nn=..:%x:%x:%x:%x:%x:%x:..\n",
        nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif

//  q = q - nn
  q.d1 = q.d1 - nn.d1;
  q.d2 = q.d2 - nn.d2 - AS_UINT_V((q.d1 > 0x7FFF)?1:0);
  q.d3 = q.d3 - nn.d3 - AS_UINT_V((q.d2 > 0x7FFF)?1:0);
  q.d4 = q.d4 - nn.d4 - AS_UINT_V((q.d3 > 0x7FFF)?1:0);
  q.d5 = q.d5 - nn.d5 - AS_UINT_V((q.d4 > 0x7FFF)?1:0);
  q.d6 = q.d6 - nn.d6 - AS_UINT_V((q.d5 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
  q.d1 &= 0x7FFF;
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("div_120_60#2.6: q=%x:%x:%x:%x:%x:%x:%x:..\n",
        q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0);
#endif

/********** Step 3, Offset 2^0 (0*15 +0) **********/
  MODBASECASE_NONZERO_ERROR(q.d6, 3, 6, 3);

  qf= CONVERT_FLOAT_V(mad24(q.d5, 32768u, q.d4));   // PERF: combine 2 into 1 before CONVERT (mad(d5d4))
  qf= qf * 32768.0f * 32768.0f + CONVERT_FLOAT_V(mad24(q.d3, 32768u, q.d2));

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 4);

  res->d1 += (qi >> 15);
  res->d0  = (qi ) & 0x7FFF;
  res->d2 += (res->d1) >> 15;   // PERF: do carry for res at the very end
  res->d3 += (res->d2) >> 15;   // PERF: do carry for res at the very end
  res->d2 &= 0x7FFF;

  qil = qi & 0x7FFF;
  qih = (qi >> 15) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("div_120_60#3: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d3.s0, res->d2.s0, res->d1.s0, res->d0.s0);
#endif

// nn = n * qi
  nn.d0  = mul24(n.d0, qil);
  nn.d1  = mad24(n.d0, qih, nn.d0 >> 15);
  nn.d0 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#3.1: nn=..:..:..:..:..:..:%x:%x\n",
        nn.d1.s0, nn.d0.s0);
#endif

  nn.d1  = mad24(n.d1, qil, nn.d1);
  nn.d2  = mad24(n.d1, qih, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#3.2: nn=..:..:..:..:..:%x:%x:%x\n",
        nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d2  = mad24(n.d2, qil, nn.d2);
  nn.d3  = mad24(n.d2, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_120_60#3.3: nn=..:..:..:..:%x:%x:%x:%x\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d3  = mad24(n.d3, qil, nn.d3);
  nn.d4  = mad24(n.d3, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_120_60#3.4: nn=..:..:..:%x:%x:%x:%x:%x\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

// no shift-left 

//  q = q - nn
  q.d0 = q.d0 - nn.d0;
  q.d1 = q.d1 - nn.d1 - AS_UINT_V((q.d0 > 0x7FFF)?1:0);
  q.d2 = q.d2 - nn.d2 - AS_UINT_V((q.d1 > 0x7FFF)?1:0);
  q.d3 = q.d3 - nn.d3 - AS_UINT_V((q.d2 > 0x7FFF)?1:0);
  q.d4 = q.d4 - nn.d4 - AS_UINT_V((q.d3 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
  q.d5 = q.d5 - nn.d5 - AS_UINT_V((q.d4 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
  q.d0 &= 0x7FFF;
  q.d1 &= 0x7FFF;
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("div_120_60#3.5: q=%x:%x:%x:%x:%x:%x:%x:%x\n",
        q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0);
#endif

/********** Step 4, final compare **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 5);
  MODBASECASE_NONZERO_ERROR(q.d4, 4, 4, 6);


//  res->d0=q.d0;
//  res->d1=q.d1;
//  res->d2=q.d2;
  tmp60.d0=q.d0;
  tmp60.d1=q.d1;
  tmp60.d2=q.d2;
  tmp60.d3=q.d3;
  
/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
  inc_if_ge_60(res, tmp60, n);
}


void mod_simple_60(int60_v * const res, const int60_v q, const int60_v n, const float_v nf
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
  __private uint_v qi, qil, qih;
  __private int60_v nn;

  qf = CONVERT_FLOAT_V(mad24(q.d3, 32768u, q.d2));
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
  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 100, qi, 12);
  }
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_simple_60: q=%x:%x:%x:%x, n=%x:%x:%x:%x, nf=%.7G, qf=%f, qi=%x\n",
        q.d3, q.d2, q.d1, q.d0, n.d3, n.d2, n.d1, n.d0, nf, qf, qi);
#endif
  qil = qi & 0x7FFF;
  qih = (qi >> 15) & 0x7FFF;  // PERF: really needed? qi should not be that big ...

// nn = n * qi
  nn.d0  = mul24(n.d0, qil);
  nn.d1  = mad24(n.d0, qih, nn.d0 >> 15);
  nn.d0 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_simple_60#1: nn=..:..:%x:%x\n",
        nn.d1.s0, nn.d0.s0);
#endif

  nn.d1  = mad24(n.d1, qil, nn.d1);
  nn.d2  = mad24(n.d1, qih, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_simple_60#2: nn=..:%x:%x:%x\n",
        nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d2  = mad24(n.d2, qil, nn.d2);
  nn.d3  = mad24(n.d2, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_simple_60#3: nn=%x:%x:%x:%x\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d3  = mad24(n.d3, qil, nn.d3);
  nn.d3 &= 0x7FFF;  // PERF: needed?
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_simple_60#4: nn=%x:%x:%x:%x\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif


  res->d0 = q.d0 - nn.d0;
  res->d1 = q.d1 - nn.d1 - AS_UINT_V((res->d0 > 0x7FFF)?1:0);
  res->d2 = q.d2 - nn.d2 - AS_UINT_V((res->d1 > 0x7FFF)?1:0);
  res->d3 = q.d3 - nn.d3 - AS_UINT_V((res->d2 > 0x7FFF)?1:0);
  res->d0 &= 0x7FFF;
  res->d1 &= 0x7FFF;
  res->d2 &= 0x7FFF;
  res->d3 &= 0x7FFF;  // PERF: needed?

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("mod_simple_60#5: res=%x:%x:%x:%x\n",
        res->d3.s0, res->d2.s0, res->d1.s0, res->d0.s0);
#endif

}

__kernel void barrett15_60(__private uint exp, const int60_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int120_t bb,
#endif
                           __global uint * restrict RES, const int bit_max
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

*/
{
  __private int60_t exp60;
  __private int60_v a, u, f, k;
  __private int120_v b, tmp120;
  __private int60_v tmp60;
  __private float_v ff;
  __private uint tid, bit_max_60=60-bit_max, bit_max_45=bit_max-45; //bit_max is 45 .. 58
  __private uint bit_max45_mult = 1 << bit_max_45; /* used for bit shifting... */
  __private uint_v t;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int120_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * BARRETT_VECTOR_SIZE;

  exp60.d3=0;exp60.d2=exp>>29;exp60.d1=(exp>>14)&0x7FFF;exp60.d0=(exp+exp)&0x7FFF;	// exp60 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("barrett15_60: exp=%d, x2=%x:%x:%x, b=%x:%x:%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x:%x\n",
        exp, exp60.d2, exp60.d1, exp60.d0, bb.d7, bb.d6, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d3, k_base.d2, k_base.d1, k_base.d0);
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
  a.d0 = t & 0x7FFF;
  a.d1 = t >> 15;  // t is 24 bits at most

  k.d0  = mad24(a.d0, 4620u, k_base.d0);
  k.d1  = mad24(a.d1, 4620u, k_base.d1) + (k.d0 >> 15);
  k.d0 &= 0x7FFF;
  k.d2  = (k.d1 >> 15) + k_base.d2;
  k.d1 &= 0x7FFF;
  k.d3  = (k.d2 >> 15) + k_base.d3;
  k.d2 &= 0x7FFF;
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("barrett15_60: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x:%x\n",
        tid, t, k.d3, k.d2, k.d1, k.d0);
#endif
		// f = 2 * k * exp + 1
  f.d0 = mad24(k.d0, exp60.d0, 1u);

  f.d1 = mad24(k.d1, exp60.d0, f.d0 >> 15);
  f.d1 = mad24(k.d0, exp60.d1, f.d1);
  f.d0 &= 0x7FFF;

  f.d2 = mad24(k.d2, exp60.d0, f.d1 >> 15);
  f.d2 = mad24(k.d1, exp60.d1, f.d2);
  f.d2 = mad24(k.d0, exp60.d2, f.d2);
  f.d1 &= 0x7FFF;

  f.d3 = mad24(k.d3, exp60.d0, f.d2 >> 15);
  f.d3 = mad24(k.d2, exp60.d1, f.d3);
  f.d3 = mad24(k.d1, exp60.d2, f.d3);
//  f.d3 = mad24(k.d0, exp60.d3, f.d3);    // exp60.d3 = 0
  f.d2 &= 0x7FFF;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("barrett15_60: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_V(mad24(f.d3, 32768u, f.d2));
  ff= ff * 32768.0f + CONVERT_FLOAT_V(f.d1);

  ff= as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we always underestimate the quotient
        
        
  // OpenCL shifts 32-bit values by 31 at most
  tmp120.d7 = (0x4000 >> (bit_max_60)) >> (bit_max_60);	// tmp120 = 2^(2*bit_max)
  tmp120.d6 = ((1 << (bit_max_45)) << (bit_max_45))&0x7FFF;   // 1 << (b << 1) = (1 << b) << b
  tmp120.d5 = 0; tmp120.d4 = 0; tmp120.d3 = 0; tmp120.d2 = 0; tmp120.d1 = 0; tmp120.d0 = 0;

  div_120_60(&u,tmp120,f,ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
#ifdef CHECKS_MODBASECASE
                  ,modbasecase_debug
#endif
);						// u = floor(tmp120 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("barrett15_60: u=%x:%x:%x:%x, ff=%G\n",
        u.d3, u.d2, u.d1, u.d0, ff);
#endif
  a.d0 = mad24(bb.d4, bit_max45_mult, (bb.d3 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bb.d5, bit_max45_mult, (bb.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bb.d6, bit_max45_mult, (bb.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d3 = mad24(bb.d7, bit_max45_mult, (bb.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)

  mul_60_120_no_low2(&tmp120, a, u);					// tmp120 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("barrett15_60: a=%x:%x:%x:%x * u = %x:%x:%x:%x:%x...\n",
        a.d3, a.d2, a.d1, a.d0, tmp120.d7, tmp120.d6, tmp120.d5, tmp120.d4, tmp120.d3);
#endif

  a.d0 = mad24(tmp120.d4, bit_max45_mult, (tmp120.d3 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = mad24(tmp120.d5, bit_max45_mult, (tmp120.d4 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d2 = mad24(tmp120.d6, bit_max45_mult, (tmp120.d5 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d3 = mad24(tmp120.d7, bit_max45_mult, (tmp120.d6 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)

  mul_60(&tmp60, a, f);							// tmp60 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("barrett15_60: a=%x:%x:%x:%x * f = %x:%x:%x:%x (tmp)\n",
        a.d3, a.d2, a.d1, a.d0, tmp60.d3, tmp60.d2, tmp60.d1, tmp60.d0);
#endif
  tmp60.d0 = (bb.d0 - tmp60.d0) & 0x7FFF;
  tmp60.d1 = (bb.d1 - tmp60.d1 - AS_UINT_V((tmp60.d0 > bb.d0) ? 1 : 0 ));
  tmp60.d2 = (bb.d2 - tmp60.d2 - AS_UINT_V((tmp60.d1 > bb.d1) ? 1 : 0 ));
  tmp60.d3 = (bb.d3 - tmp60.d3 - AS_UINT_V((tmp60.d2 > bb.d2) ? 1 : 0 ));
  tmp60.d1 &= 0x7FFF;
  tmp60.d2 &= 0x7FFF;
  tmp60.d3 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("barrett15_60: b=%x:%x:%x:%x - tmp = %x:%x:%x:%x (tmp)\n",
        bb.d3, bb.d2, bb.d1, bb.d0, tmp60.d3, tmp60.d2, tmp60.d1, tmp60.d0);
#endif
#ifndef CHECKS_MODBASECASE
  mod_simple_60(&a, tmp60, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 6;
  if(bit_max_60 == 1) limit = 8;						// bit_max == 65, due to decreased accuracy of mul_96_192_no_low2() above we need a higher threshold
  if(bit_max_60 == 2) limit = 7;						// bit_max == 66, ...
  mod_simple_60(&a, tmp60, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max_60, limit, modbasecase_debug);
#endif
  
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("barrett15_60: tmp=%x:%x:%x:%x mod f=%x:%x:%x:%x = %x:%x:%x:%x (a)\n",
        tmp60.d3, tmp60.d2, tmp60.d1, tmp60.d0, f.d3, f.d2, f.d1, f.d0, a.d3, a.d2, a.d1, a.d0 );
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_60_120(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d3, a.d2, a.d1, a.d0, b.d7, b.d6, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0 );
#endif
    a.d0 = mad24(b.d4, bit_max45_mult, (b.d3 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d0 = mad24(b.d5, bit_max45_mult, (b.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d0 = mad24(b.d6, bit_max45_mult, (b.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d0 = mad24(b.d7, bit_max45_mult, (b.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)

    mul_60_120_no_low2(&tmp120, a, u);					// tmp120 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x:%x * u = %x:%x:%x:%x:%x...\n",
        a.d3, a.d2, a.d1, a.d0, tmp120.d7, tmp120.d6, tmp120.d5, tmp120.d4, tmp120.d3);
#endif
    a.d0 = mad24(tmp120.d4, bit_max45_mult, (tmp120.d3 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d0 = mad24(tmp120.d5, bit_max45_mult, (tmp120.d4 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d0 = mad24(tmp120.d6, bit_max45_mult, (tmp120.d5 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d0 = mad24(tmp120.d7, bit_max45_mult, (tmp120.d6 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)

    mul_60(&tmp60, a, f);						// tmp60 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x:%x * f = %x:%x:%x:%x (tmp)\n",
        a.d3, a.d2, a.d1, a.d0, tmp60.d3, tmp60.d2, tmp60.d1, tmp60.d0);
#endif
    tmp60.d0 = (b.d0 - tmp60.d0) & 0x7FFF;
    tmp60.d1 = (b.d1 - tmp60.d1 - AS_UINT_V((tmp60.d0 > bb.d0) ? 1 : 0 ));
    tmp60.d2 = (b.d2 - tmp60.d2 - AS_UINT_V((tmp60.d1 > bb.d1) ? 1 : 0 ));
    tmp60.d3 = (b.d3 - tmp60.d3 - AS_UINT_V((tmp60.d2 > bb.d2) ? 1 : 0 ));
    tmp60.d1 &= 0x7FFF;
    tmp60.d2 &= 0x7FFF;
    tmp60.d3 &= 0x7FFF;
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2, b.d1, b.d0, tmp60.d2, tmp60.d1, tmp60.d0);
#endif
    if(exp&0x80000000)shl_60(&tmp60);					// "optional multiply by 2" in Prime 95 documentation

#ifndef CHECKS_MODBASECASE
    mod_simple_60(&a, tmp60, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    if(bit_max_60 == 1) limit = 8;					// bit_max == 65, due to decreased accuracy of mul_96_192_no_low2() above we need a higher threshold
    if(bit_max_60 == 2) limit = 7;					// bit_max == 66, ...
    mod_simple_60(&a, tmp60, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max_60, limit, modbasecase_debug);
#endif

    exp+=exp;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("loopend: exp=%d, tmp=%x:%x:%x:%x mod f=%x:%x:%x:%x = %x:%x:%x:%x (a)\n",
        exp, tmp60.d3, tmp60.d2, tmp60.d1, tmp60.d0, f.d3, f.d2, f.d1, f.d0, a.d3, a.d2, a.d1, a.d0 );
#endif
  }


#ifndef CHECKS_MODBASECASE
  a = sub_if_gte_60(a,f);	// final adjustment in case a >= f
#else
  tmp60 = sub_if_gte_60(a,f);
  a = sub_if_gte_60(tmp60,f);
  if( (tmp60.d2 != a.d2) || (tmp60.d1 != a.d1) || (tmp60.d0 != a.d0))
  {
    printf("EEEEEK, final a was >= f\n");
  }
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x:%x \n",
         a.d3, a.d2, a.d1, a.d0 );
#endif
  

/* finally check if we found a factor and write the factor to RES[] */
#if (BARRETT_VECTOR_SIZE == 1)
  if( ((a.d3|a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("barrett15_60: tid=%ld found factor: q=%x:%x:%x:%x, k=%x:%x:%x:%x\n", tid, f.d3, f.d2, f.d1, f.d0, k.d3, k.d2, k.d1, k.d0);
#endif
/* in contrast to the other kernels the barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f != 1 */  
    tid=ATOMIC_INC(RES[0]);
    if(tid<10)				/* limit to 10 factors per class */
    {
//      RES[tid*3 + 1]=f.d2;
      RES[tid*3 + 2]=mad24(f.d3,0x8000u, f.d2);  // that's now 30 bits per int
      RES[tid*3 + 3]=mad24(f.d1,0x8000u, f.d0);  
    }
  }
#elif (BARRETT_VECTOR_SIZE == 2)
  EVAL_RES_c(x)
  EVAL_RES_c(y)
#elif (BARRETT_VECTOR_SIZE == 4)
  EVAL_RES_c(x)
  EVAL_RES_c(y)
  EVAL_RES_c(z)
  EVAL_RES_c(w)
#elif (BARRETT_VECTOR_SIZE == 8)
  EVAL_RES_c(s0)
  EVAL_RES_c(s1)
  EVAL_RES_c(s2)
  EVAL_RES_c(s3)
  EVAL_RES_c(s4)
  EVAL_RES_c(s5)
  EVAL_RES_c(s6)
  EVAL_RES_c(s7)
#elif (BARRETT_VECTOR_SIZE == 16)
  EVAL_RES_c(s0)
  EVAL_RES_c(s1)
  EVAL_RES_c(s2)
  EVAL_RES_c(s3)
  EVAL_RES_c(s4)
  EVAL_RES_c(s5)
  EVAL_RES_c(s6)
  EVAL_RES_c(s7)
  EVAL_RES_c(s8)
  EVAL_RES_c(s9)
  EVAL_RES_c(sa)
  EVAL_RES_c(sb)
  EVAL_RES_c(sc)
  EVAL_RES_c(sd)
  EVAL_RES_c(se)
  EVAL_RES_c(sf)
#endif
 
}

/****************************************
 ****************************************
 * 15-bit based 75-bit barrett-kernels
 *
 ****************************************
 ****************************************/

int75_v sub_if_gte_75(const int75_v a, const int75_v b)
/* return (a>b)?a-b:a */
{
  int75_v tmp;
  /* do the subtraction and use tmp.d4 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0) & 0x7FFF;
  tmp.d1 = (a.d1 - b.d1 - AS_UINT_V((b.d0 > a.d0) ? 1 : 0));
  tmp.d2 = (a.d2 - b.d2 - AS_UINT_V((tmp.d1 > a.d1) ? 1 : 0));
  tmp.d3 = (a.d3 - b.d3 - AS_UINT_V((tmp.d2 > a.d2) ? 1 : 0));
  tmp.d4 = (a.d4 - b.d4 - AS_UINT_V((tmp.d3 > a.d3) ? 1 : 0));
  tmp.d1&= 0x7FFF;
  tmp.d2&= 0x7FFF;
  tmp.d3&= 0x7FFF;

  tmp.d0 = (tmp.d4 > a.d4) ? a.d0 : tmp.d0;
  tmp.d1 = (tmp.d4 > a.d4) ? a.d1 : tmp.d1;
  tmp.d2 = (tmp.d4 > a.d4) ? a.d2 : tmp.d2;
  tmp.d3 = (tmp.d4 > a.d4) ? a.d3 : tmp.d3;
  tmp.d4 = (tmp.d4 > a.d4) ? a.d4 : tmp.d4; //  & 0x7FFF not necessary as tmp.d4 is <= a.d4

  return tmp;
}

void inc_if_ge_75(int75_v * const res, const int75_v a, const int75_v b)
{ /* if (a >= b) res++ */
  __private uint_v ge,tmpa0,tmpa1,tmpb0,tmpb1;
  // PERF: faster to combine them to 30-bits before all this?
  // Yes, a tiny bit: 9 operations each, plus 2 vs. 4 conditional loads with dependencies.
  // PERF: further improvement by upsampling to long int? No, that is slower.
  tmpa0=mad24(a.d2, 32768, a.d1);
  tmpa1=mad24(a.d4, 32768, a.d3);
  tmpb0=mad24(b.d2, 32768, b.d1);
  tmpb1=mad24(b.d4, 32768, b.d3);
  ge = AS_UINT_V((tmpa1 == tmpb1) ? ((tmpa0 == tmpb0) ? (a.d0 >= b.d0)
                                                      : (tmpa0 > tmpb0))
                                  : (tmpa1 > tmpb1));
  /*
  ge = AS_UINT_V((a.d4 == b.d4) ? ((a.d3 == b.d3) ? ((a.d2 == b.d2) ? ((a.d1 == b.d1) ? (a.d0 >= b.d0)
                                                                                      : (a.d1 > b.d1))
                                                                    : (a.d2 > b.d2))
                                                  : (a.d3 > b.d3))
                                :(a.d4 > b.d4));

                                */
  res->d0 += AS_UINT_V(ge ? 1 : 0);
  res->d1 += res->d0 >> 15;
  res->d2 += res->d1 >> 15;
  res->d3 += res->d2 >> 15;
  res->d4 += res->d3 >> 15;
  res->d0 &= 0x7FFF;
  res->d1 &= 0x7FFF;
  res->d2 &= 0x7FFF;
  res->d3 &= 0x7FFF;
}

void mul_75(int75_v * const res, const int75_v a, const int75_v b)
/* res = a * b */
{
  res->d0 = mul24(a.d0, b.d0);

  res->d1 = mad24(a.d1, b.d0, res->d0 >> 15);
  res->d1 = mad24(a.d0, b.d1, res->d1);
  res->d0 &= 0x7FFF;

  res->d2 = mad24(a.d2, b.d0, res->d1 >> 15);
  res->d2 = mad24(a.d1, b.d1, res->d2);
  res->d2 = mad24(a.d0, b.d2, res->d2);
  res->d1 &= 0x7FFF;

  res->d3 = mad24(a.d3, b.d0, res->d2 >> 15);
  res->d3 = mad24(a.d2, b.d1, res->d3);
  res->d3 = mad24(a.d1, b.d2, res->d3);
  res->d3 = mad24(a.d0, b.d3, res->d3);
  res->d2 &= 0x7FFF;

  res->d4 = mad24(a.d4, b.d0, res->d3 >> 15);
  res->d4 = mad24(a.d3, b.d1, res->d4);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
  res->d4 = mad24(a.d0, b.d4, res->d4);  // the 5th mad can overflow d4, but that's ok for this function.
  res->d3 &= 0x7FFF;
  res->d4 &= 0x7FFF;

}


void mul_75_150_no_low3(int150_v * const res, const int75_v a, const int75_v b)
/*
res ~= a * b
res.d0 to res.d2 are NOT computed. Carries to res.d2 are ignored,
too. So the digits res.d{4-9} might differ from mul_75_150(). 
 */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // this optimized mul 5x5 requires: 19 mul/mad24, 7 shift, 6 and, 1 add

  res->d3 = mul24(a.d3, b.d0);
  res->d3 = mad24(a.d2, b.d1, res->d3);
  res->d3 = mad24(a.d1, b.d2, res->d3);
  res->d3 = mad24(a.d0, b.d3, res->d3);

  res->d4 = mad24(a.d4, b.d0, res->d3 >> 15);
  // res->d3 &= 0x7FFF;  // d3 itself is not used, only its carry to d4 is required
  res->d4 = mad24(a.d3, b.d1, res->d4);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
   // 5th mad24 can overflow d4, need to handle carry before: pull in the first d5 line
  res->d5 = mad24(a.d4, b.d1, res->d4 >> 15);
  res->d4 &= 0x7FFF;
  res->d4 = mad24(a.d0, b.d4, res->d4);  // 31-bit at most

  res->d5 = mad24(a.d3, b.d2, res->d4 >> 15) + res->d5;
  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d5 = mad24(a.d1, b.d4, res->d5);
  res->d4 &= 0x7FFF;
  // now we have in d5: 4x mad24() + 1x 17-bit carry + 1x 16-bit carry: still fits into 32 bits

  res->d6 = mad24(a.d2, b.d4, res->d5 >> 15);
  res->d6 = mad24(a.d3, b.d3, res->d6);
  res->d6 = mad24(a.d4, b.d2, res->d6);
  res->d5 &= 0x7FFF;

  res->d7 = mad24(a.d3, b.d4, res->d6 >> 15);
  res->d7 = mad24(a.d4, b.d3, res->d7);
  res->d6 &= 0x7FFF;

  res->d8 = mad24(a.d4, b.d4, res->d7 >> 15);
  res->d7 &= 0x7FFF;

  res->d9 = res->d8 >> 15;
  res->d8 &= 0x7FFF;
}


void mul_75_150(int150_v * const res, const int75_v a, const int75_v b)
/*
res = a * b
 */
{
  /* this is the complete implementation, optimize later */
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // mul 5x5 requires: 25 mul/mad24, 10 shift, 10 and, 1 add

  res->d0 = mul24(a.d0, b.d0);

  res->d1 = mad24(a.d1, b.d0, res->d0 >> 15);
  res->d1 = mad24(a.d0, b.d1, res->d1);
  res->d0 &= 0x7FFF;

  res->d2 = mad24(a.d2, b.d0, res->d1 >> 15);
  res->d2 = mad24(a.d1, b.d1, res->d2);
  res->d2 = mad24(a.d0, b.d2, res->d2);
  res->d1 &= 0x7FFF;

  res->d3 = mad24(a.d3, b.d0, res->d2 >> 15);
  res->d3 = mad24(a.d2, b.d1, res->d3);
  res->d3 = mad24(a.d1, b.d2, res->d3);
  res->d3 = mad24(a.d0, b.d3, res->d3);
  res->d2 &= 0x7FFF;

  res->d4 = mad24(a.d4, b.d0, res->d3 >> 15);
  res->d3 &= 0x7FFF;
  res->d4 = mad24(a.d3, b.d1, res->d4);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
   // 5th mad24 can overflow d4, need to handle carry before: pull in the first d5 line
  res->d5 = mad24(a.d4, b.d1, res->d4 >> 15);
  res->d4 &= 0x7FFF;
  res->d4 = mad24(a.d0, b.d4, res->d4);  // 31-bit at most

  res->d5 = mad24(a.d3, b.d2, res->d4 >> 15) + res->d5;
  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d5 = mad24(a.d1, b.d4, res->d5);
  res->d4 &= 0x7FFF;
  // now we have in d5: 4x mad24() + 1x 17-bit carry + 1x 16-bit carry: still fits into 32 bits

  res->d6 = mad24(a.d2, b.d4, res->d5 >> 15);
  res->d6 = mad24(a.d3, b.d3, res->d6);
  res->d6 = mad24(a.d4, b.d2, res->d6);
  res->d5 &= 0x7FFF;

  res->d7 = mad24(a.d3, b.d4, res->d6 >> 15);
  res->d7 = mad24(a.d4, b.d3, res->d7);
  res->d6 &= 0x7FFF;

  res->d8 = mad24(a.d4, b.d4, res->d7 >> 15);
  res->d7 &= 0x7FFF;

  res->d9 = res->d8 >> 15;
  res->d8 &= 0x7FFF;
}


void square_75_150(int150_v * const res, const int75_v a)
/* res = a^2 = a.d0^2 + a.d1^2 + a.d2^2 + a.d3^2 + ...
         2(a.d0*a.d1 + a.d0*a.d2 + a.d0*a.d3 + a.d1*a.d2 + a.d1*a.d3 + a.d2*a.d3 ...)
       = a.d0^2 + a.d1^2 + a.d2^2 + a.d3^2 +
         2(a.d0(a.d1+a.d2) + a.d1(a.d2+a.d3) + a.d3(a.d0+a.d2)) ...
   */
{


  /* for now, use the complete implementation, optimize later */

  mul_75_150(res, a, a);
}


void shl_75(int75_v * const a)
/* shiftleft a one bit */
{
  a->d4 = mad24(a->d4, 2u, a->d3 >> 14); // keep the extra top bit
  a->d3 = mad24(a->d3, 2u, a->d2 >> 14) & 0x7FFF;
  a->d2 = mad24(a->d2, 2u, a->d1 >> 14) & 0x7FFF;
  a->d1 = mad24(a->d1, 2u, a->d0 >> 14) & 0x7FFF;
  a->d0 = (a->d0 << 1u) & 0x7FFF;
}


void div_150_75(int75_v * const res, __private int150_v q, const int75_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
#ifdef CHECKS_MODBASECASE
                  , __global uint * restrict modbasecase_debug
#endif
)/* res = q / n (integer division) */
{
  __private float_v qf;
  __private uint_v qi, qil, qih;
  __private int150_v nn;
  __private int75_v tmp75;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("div_150_75#0: q=%x:%x:%x:%x:%x:%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, nf=%#G\n",
        q.d9.s0, q.d8.s0, q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0,
        n.d4.s0, n.d3.s0, n.d2.s0, n.d1.s0, n.d0.s0, nf.s0);
#endif

/********** Step 1, Offset 2^53 (3*15 + 8) **********/
  qf= CONVERT_FLOAT_V(mad24(q.d9, 32768u, q.d8));
  qf= qf * 32768.0f * 32768.0f + CONVERT_FLOAT_V(mad24(q.d7, 32768u, q.d6));
  qf*= 256.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<23, 1, qi, 0);  // first step is 1 bit bigger

  res->d4 = (qi >> 8);
  res->d3 = (qi << 7) & 0x7FFF;
  qil = qi & 0x7FFF;
  qih = (qi >> 15) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("div_150_75#1: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:..:..:..\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d4.s0, res->d3.s0);
#endif

  /*******************************************************/

// nn = n * qi
  nn.d3  = mul24(n.d0, qil);
  nn.d4  = mad24(n.d0, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#1.1: nn=..:..:..:..:..:%x:%x:..:..:..\n",
        nn.d4.s0, nn.d3.s0);
#endif

  nn.d4  = mad24(n.d1, qil, nn.d4);
  nn.d5  = mad24(n.d1, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#1.2: nn=..:..:..:..:%x:%x:%x:...\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0);
#endif

  nn.d5  = mad24(n.d2, qil, nn.d5);
  nn.d6  = mad24(n.d2, qih, nn.d5 >> 15);
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#1.3: nn=..:..:..:%x:%x:%x:%x:...\n",
        nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0);
#endif

  nn.d6  = mad24(n.d3, qil, nn.d6);
  nn.d7  = mad24(n.d3, qih, nn.d6 >> 15);
  nn.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#1.4: nn=..:..:%x:%x:%x:%x:%x:...\n",
        nn.d7.s0, nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0);
#endif
  nn.d7  = mad24(n.d4, qil, nn.d7);
  nn.d8  = mad24(n.d4, qih, nn.d7 >> 15);
  nn.d7 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#1.5: nn=..:%x:%x:%x:%x:%x:%x:...\n",
        nn.d8.s0, nn.d7.s0, nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0);
#endif

// now shift-left 7 bits
#ifdef CHECKS_MODBASECASE
  nn.d9  = nn.d8 >> 8;  // PERF: not needed as it will be gone anyway after sub
#endif
  nn.d8  = mad24(nn.d8 & 0xFF, 128u, nn.d7 >> 8);
  nn.d7  = mad24(nn.d7 & 0xFF, 128u, nn.d6 >> 8);
  nn.d6  = mad24(nn.d6 & 0xFF, 128u, nn.d5 >> 8);
  nn.d5  = mad24(nn.d5 & 0xFF, 128u, nn.d4 >> 8);
  nn.d4  = mad24(nn.d4 & 0xFF, 128u, nn.d3 >> 8);
  nn.d3  = (nn.d3 & 0xFF) << 7;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("div_150_75#1.6: nn=%x:%x:%x:%x:%x:%x:%x:..:..:..\n",
        nn.d9.s0, nn.d8.s0, nn.d7.s0, nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0);
#endif

//  q = q - nn
  q.d3 = q.d3 - nn.d3;
  q.d4 = q.d4 - nn.d4 - AS_UINT_V((q.d3 > 0x7FFF)?1:0);
  q.d5 = q.d5 - nn.d5 - AS_UINT_V((q.d4 > 0x7FFF)?1:0);
  q.d6 = q.d6 - nn.d6 - AS_UINT_V((q.d5 > 0x7FFF)?1:0);
  q.d7 = q.d7 - nn.d7 - AS_UINT_V((q.d6 > 0x7FFF)?1:0);
  q.d8 = q.d8 - nn.d8 - AS_UINT_V((q.d7 > 0x7FFF)?1:0); 
#ifdef CHECKS_MODBASECASE
  q.d9 = q.d9 - nn.d9 - AS_UINT_V((q.d8 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
#endif
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
  q.d7 &= 0x7FFF;
  q.d8 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("div_150_75#1.7: q=%x!%x:%x:%x:%x:%x:%x:..:..:..\n",
        q.d9.s0, q.d8.s0, q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0);
#endif
  MODBASECASE_NONZERO_ERROR(q.d9, 1, 9, 1);

  /********** Step 2, Offset 2^38 (2*15 + 8) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d8, 32768u, q.d7));
  qf= qf * 32768.0f * 32768.0f + CONVERT_FLOAT_V(mad24(q.d6, 32768u, q.d5));
  qf*= 16384.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<23, 2, qi, 2); // here, we need 2^23 ...

  res->d3 += (qi >> 14);
  res->d2 = (qi << 1) & 0x7FFF;
  qil = qi & 0x7FFF;
  qih = (qi >> 15) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("div_150_75#2: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:..:..\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d4.s0, res->d3.s0, res->d2.s0);
#endif

  /*******************************************************/

// nn = n * qi
  nn.d2  = mul24(n.d0, qil);
  nn.d3  = mad24(n.d0, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#2.1: nn=..:..:..:..:%x:%x:..:..\n",
        nn.d3.s0, nn.d2.s0);
#endif

  nn.d3  = mad24(n.d1, qil, nn.d3);
  nn.d4  = mad24(n.d1, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#2.2: nn=..:..:..:%x:%x:%x:..:..\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

  nn.d4  = mad24(n.d2, qil, nn.d4);
  nn.d5  = mad24(n.d2, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#2.3: nn=..:..:%x:%x:%x:%x:..:..\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

  nn.d5  = mad24(n.d3, qil, nn.d5);
  nn.d6  = mad24(n.d3, qih, nn.d5 >> 15);
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#2.4: nn=..:%x:%x:%x:%x:%x:..:..\n",
        nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

  nn.d6  = mad24(n.d4, qil, nn.d6);
#ifdef CHECKS_MODBASECASE
  nn.d7  = mad24(n.d4, qih, nn.d6 >> 15);
#endif
  nn.d6 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#2.5: nn=..:%x:%x:%x:%x:%x:%x:..:..\n",
        nn.d7.s0, nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

// now shift-left 1 bit
#ifdef CHECKS_MODBASECASE
  nn.d8  = nn.d7 >> 14;  // PERF: not needed as it will be gone anyway after sub
  nn.d7  = mad24(nn.d7 & 0x3FFF, 2u, nn.d6 >> 14);  // PERF: not needed as it will be gone anyway after sub
#endif
  nn.d6  = mad24(nn.d6 & 0x3FFF, 2u, nn.d5 >> 14);
  nn.d5  = mad24(nn.d5 & 0x3FFF, 2u, nn.d4 >> 14);
  nn.d4  = mad24(nn.d4 & 0x3FFF, 2u, nn.d3 >> 14);
  nn.d3  = mad24(nn.d3 & 0x3FFF, 2u, nn.d2 >> 14);
  nn.d2  = (nn.d2 & 0x3FFF) << 1;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#2.6: nn=..:%x:%x:%x:%x:%x:%x:%x:..:..\n",
        nn.d8.s0, nn.d7.s0, nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0);
#endif

//  q = q - nn
  q.d2 = q.d2 - nn.d2;
  q.d3 = q.d3 - nn.d3 - AS_UINT_V((q.d2 > 0x7FFF)?1:0);
  q.d4 = q.d4 - nn.d4 - AS_UINT_V((q.d3 > 0x7FFF)?1:0);
  q.d5 = q.d5 - nn.d5 - AS_UINT_V((q.d4 > 0x7FFF)?1:0);
  q.d6 = q.d6 - nn.d6 - AS_UINT_V((q.d5 > 0x7FFF)?1:0);
#ifdef CHECKS_MODBASECASE
  q.d7 = q.d7 - nn.d7 - AS_UINT_V((q.d6 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
  q.d8 = q.d8 - nn.d8 - AS_UINT_V((q.d7 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
  q.d7 &= 0x7FFF;
#endif
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("div_150_75#2.7: q=..:%x:%x!%x:%x:%x:%x:%x:..:..\n",
        q.d8.s0, q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0);
#endif

  MODBASECASE_NONZERO_ERROR(q.d8, 2, 8, 3);
  MODBASECASE_NONZERO_ERROR(q.d7, 2, 7, 4);

  /********** Step 3, Offset 2^20 (1*15 + 5) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d6, 32768u, q.d5)); 
  qf= qf * 32768.0f * 32768.0f + CONVERT_FLOAT_V(mad24(q.d4, 32768u, q.d3));
//  qf*= 512.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 5);

  qih = (qi >> 15) & 0x7FFF;
  qil = qi & 0x7FFF;
  res->d2 += qih;
  res->d1 = qil;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("div_150_75#3: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:..\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d4.s0, res->d3.s0, res->d2.s0, res->d1.s0);
#endif

  /*******************************************************/

// nn = n * qi
  nn.d1  = mul24(n.d0, qil);
  nn.d2  = mad24(n.d0, qih, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#3.1: nn=..:..:..:..:%x:%x:..\n",
        nn.d2.s0, nn.d1.s0);
#endif

  nn.d2  = mad24(n.d1, qil, nn.d2);
  nn.d3  = mad24(n.d1, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#3.2: nn=..:..:..:%x:%x:%x:..\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif

  nn.d3  = mad24(n.d2, qil, nn.d3);
  nn.d4  = mad24(n.d2, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#3.3: nn=..:..:%x:%x:%x:%x:..\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif

  nn.d4  = mad24(n.d3, qil, nn.d4);
  nn.d5  = mad24(n.d3, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#3.4: nn=..:%x:%x:%x:%x:%x:..\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif
  nn.d5  = mad24(n.d4, qil, nn.d5);
#ifdef CHECKS_MODBASECASE
  nn.d6  = mad24(n.d4, qih, nn.d5 >> 15);
#endif
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#3.5: nn=..:%x:%x:%x:%x:%x:%x:..\n",
        nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif

// now shift-left 10 bits
//  nn.d7  = nn.d6 >> 5;  // PERF: not needed as it will be gone anyway after sub
//  nn.d6  = mad24(nn.d6 & 0x1F, 1024u, nn.d5 >> 5);  // PERF: not needed as it will be gone anyway after sub
//  nn.d5  = mad24(nn.d5 & 0x1F, 1024u, nn.d4 >> 5);
//  nn.d4  = mad24(nn.d4 & 0x1F, 1024u, nn.d3 >> 5);
//  nn.d3  = mad24(nn.d3 & 0x1F, 1024u, nn.d2 >> 5);
//  nn.d2  = mad24(nn.d2 & 0x1F, 1024u, nn.d1 >> 5);
//  nn.d1  = (nn.d1 & 0x1F) << 10;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#3.6: nn=..:..:%x:%x:%x:%x:%x:%x:..\n",
        nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0);
#endif

//  q = q - nn
  q.d1 = q.d1 - nn.d1;
  q.d2 = q.d2 - nn.d2 - AS_UINT_V((q.d1 > 0x7FFF)?1:0);
  q.d3 = q.d3 - nn.d3 - AS_UINT_V((q.d2 > 0x7FFF)?1:0);
  q.d4 = q.d4 - nn.d4 - AS_UINT_V((q.d3 > 0x7FFF)?1:0);
  q.d5 = q.d5 - nn.d5 - AS_UINT_V((q.d4 > 0x7FFF)?1:0);
#ifdef CHECKS_MODBASECASE
  q.d6 = q.d6 - nn.d6 - AS_UINT_V((q.d5 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
  q.d6 &= 0x7FFF;
#endif
  q.d1 &= 0x7FFF;
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("div_150_75#3.7: q=..:%x:%x:%x!%x:%x:%x:%x:%x:..\n",
        q.d8.s0, q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0);
#endif

  MODBASECASE_NONZERO_ERROR(q.d6, 3, 6, 6);

  /********** Step 4, Offset 2^0 (0*15 + 0) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d5, 32768u, q.d4));
  qf= qf * 32768.0f * 32768.0f + CONVERT_FLOAT_V(mad24(q.d3, 32768u, q.d2));
//  qf*= 2048.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 7);

  qil = qi & 0x7FFF;
  qih = (qi >> 15) & 0x7FFF;
  res->d1 += qih;
  res->d0 = qil;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("div_150_75#4: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:%x\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d4.s0, res->d3.s0, res->d2.s0, res->d1.s0, res->d0.s0);
#endif

  /*******************************************************/

// nn = n * qi
  nn.d0  = mul24(n.d0, qil);
  nn.d1  = mad24(n.d0, qih, nn.d0 >> 15);
  nn.d0 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#4.1: nn=..:..:..:..:%x:%x\n",
        nn.d1.s0, nn.d0.s0);
#endif

  nn.d1  = mad24(n.d1, qil, nn.d1);
  nn.d2  = mad24(n.d1, qih, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#4.2: nn=..:..:..:%x:%x:%x\n",
        nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d2  = mad24(n.d2, qil, nn.d2);
  nn.d3  = mad24(n.d2, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("div_150_75#4.3: nn=..:..:%x:%x:%x:%x\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d3  = mad24(n.d3, qil, nn.d3);
  nn.d4  = mad24(n.d3, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#4.4: nn=..:%x:%x:%x:%x:%x\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif
  nn.d4  = mad24(n.d4, qil, nn.d4);
#ifdef CHECKS_MODBASECASE
  nn.d5  = mad24(n.d4, qih, nn.d4 >> 15);
#endif
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("div_150_75#4.5: nn=..:%x:%x:%x:%x:%x:%x\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

// no shift-left required

//  q = q - nn
  q.d0 = q.d0 - nn.d0;
  q.d1 = q.d1 - nn.d1 - AS_UINT_V((q.d0 > 0x7FFF)?1:0);
  q.d2 = q.d2 - nn.d2 - AS_UINT_V((q.d1 > 0x7FFF)?1:0);
  q.d3 = q.d3 - nn.d3 - AS_UINT_V((q.d2 > 0x7FFF)?1:0);
  q.d4 = q.d4 - nn.d4 - AS_UINT_V((q.d3 > 0x7FFF)?1:0);
#ifdef CHECKS_MODBASECASE
  q.d5 = q.d5 - nn.d5 - AS_UINT_V((q.d4 > 0x7FFF)?1:0); // PERF: not needed: should be zero anyway
  q.d5 &= 0x7FFF;// PERF: not needed: should be zero anyway
#endif
  q.d0 &= 0x7FFF;
  q.d1 &= 0x7FFF;
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("div_150_75#4.7: q=..:%x:%x:%x:%x!%x:%x:%x:%x:%x\n",
        q.d8.s0, q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0);
#endif
 MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 8);

/********** Step 5, final compare **********/


//  res->d0=q.d0;
//  res->d1=q.d1;
//  res->d2=q.d2;

  tmp75.d0=q.d0;
  tmp75.d1=q.d1;
  tmp75.d2=q.d2;
  tmp75.d3=q.d3;
  tmp75.d4=q.d4;
  
/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
This function also handles outstanding carries in res.
*/
  inc_if_ge_75(res, tmp75, n);
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
    MODBASECASE_QI_ERROR(limit, 100, qi, 12);
  }
#endif
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("mod_simple_75: q=%x:%x:%x:%x:%x, n=%x:%x:%x:%x:%x, nf=%.7G, qf=%#G, qi=%x\n",
        q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0, q.d0.s0, n.d4.s0, n.d3.s0, n.d2.s0, n.d1.s0, n.d0.s0, nf.s0, qf.s0, qi.s0);
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
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_simple_75#5: nn=%x:%x:%x:%x:%x\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif


  res->d0 = q.d0 - nn.d0;
  res->d1 = q.d1 - nn.d1 - AS_UINT_V((res->d0 > 0x7FFF)?1:0);
  res->d2 = q.d2 - nn.d2 - AS_UINT_V((res->d1 > 0x7FFF)?1:0);
  res->d3 = q.d3 - nn.d3 - AS_UINT_V((res->d2 > 0x7FFF)?1:0);
  res->d4 = q.d4 - nn.d4 - AS_UINT_V((res->d3 > 0x7FFF)?1:0);
  res->d0 &= 0x7FFF;
  res->d1 &= 0x7FFF;
  res->d2 &= 0x7FFF;
  res->d3 &= 0x7FFF;

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("mod_simple_75#6: res=%x:%x:%x:%x:%x\n",
        res->d4.s0, res->d3.s0, res->d2.s0, res->d1.s0, res->d0.s0);
#endif

}

__kernel void barrett15_75(__private uint exp, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                           const uint8 b_in, __global uint * restrict RES, const int bit_max
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

*/
{
  __private int75_t exp75;
  __private int75_v a, u, f, k;
  __private int150_v b, tmp150;
  __private int75_v tmp75;
  __private float_v ff;
  __private uint tid, bit_max_75=75-bit_max, bit_max_60=bit_max-60; //bit_max is 60 .. 73
  __private uint bit_max75_mult = 1 << bit_max_75; /* used for bit shifting... */
  __private uint_v t;

  // implicitely assume b > 2^30 and use the 8 fields of the uint8 for d2-d9
  __private int150_t bb={0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5, b_in.s6, b_in.s7};

	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * BARRETT_VECTOR_SIZE;

  // exp75.d4=0;exp75.d3=0;  // not used, PERF: we can skip d2 as well, if we limit exp to 2^29
  exp75.d2=exp>>29;exp75.d1=(exp>>14)&0x7FFF;exp75.d0=(exp<<1)&0x7FFF;	// exp75 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("barrett15_75: exp=%d, x2=%x:%x:%x, b=%x:%x:%x:%x:%x:%x:%x:%x:0:0, k_base=%x:%x:%x:%x:%x\n",
        exp, exp75.d2, exp75.d1, exp75.d0, bb.d9, bb.d8, bb.d7, bb.d6, bb.d5, bb.d4, bb.d3, bb.d2, k_base.d4, k_base.d3, k_base.d2, k_base.d1, k_base.d0);
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
  a.d0 = t & 0x7FFF;
  a.d1 = t >> 15;  // t is 24 bits at most

  k.d0  = mad24(a.d0, 4620u, k_base.d0);
  k.d1  = mad24(a.d1, 4620u, k_base.d1) + (k.d0 >> 15);
  k.d0 &= 0x7FFF;
  k.d2  = (k.d1 >> 15) + k_base.d2;
  k.d1 &= 0x7FFF;
  k.d3  = (k.d2 >> 15) + k_base.d3;
  k.d2 &= 0x7FFF;
  k.d4  = (k.d3 >> 15) + k_base.d4;  // PERF: k.d4 = 0, normally. Can we limit k to 2^60?
  k.d3 &= 0x7FFF;
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("barrett15_75: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x:%x:%x\n",
        tid, t.s0, k.d4.s0, k.d3.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif
		// f = 2 * k * exp + 1
  f.d0 = mad24(k.d0, exp75.d0, 1u);

  f.d1 = mad24(k.d1, exp75.d0, f.d0 >> 15);
  f.d1 = mad24(k.d0, exp75.d1, f.d1);
  f.d0 &= 0x7FFF;

  f.d2 = mad24(k.d2, exp75.d0, f.d1 >> 15);
  f.d2 = mad24(k.d1, exp75.d1, f.d2);
  f.d2 = mad24(k.d0, exp75.d2, f.d2);  // PERF: if we limit exp at kernel compile time to 2^29, then we can skip exp75.d2 here and above.
  f.d1 &= 0x7FFF;

  f.d3 = mad24(k.d3, exp75.d0, f.d2 >> 15);
  f.d3 = mad24(k.d2, exp75.d1, f.d3);
  f.d3 = mad24(k.d1, exp75.d2, f.d3);
//  f.d3 = mad24(k.d0, exp75.d3, f.d3);    // exp75.d3 = 0
  f.d2 &= 0x7FFF;

  f.d4 = mad24(k.d4, exp75.d0, f.d3 >> 15);  // PERF: see above
  f.d4 = mad24(k.d3, exp75.d1, f.d4);
  f.d4 = mad24(k.d2, exp75.d2, f.d4);
  f.d3 &= 0x7FFF;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("barrett15_75: k_tab[%d]=%x, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d4.s0, k.d3.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d4.s0, f.d3.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_V(mad24(f.d4, 32768u, f.d3));
  ff= ff * 32768.0f + CONVERT_FLOAT_V(f.d2);   // f.d1 needed?

  ff= as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we always underestimate the quotient
        
        
  // OpenCL shifts 32-bit values by 31 at most
  tmp150.d9 = (0x8000 >> (bit_max_75)) >> (bit_max_75);	// tmp150 = 2^(2*bit_max)
  tmp150.d8 = ((1 << (bit_max_60)) << (bit_max_60))&0x7FFF;   // 1 << (b << 1) = (1 << b) << b
  tmp150.d7 = 0; tmp150.d6 = 0; tmp150.d5 = 0; tmp150.d4 = 0; tmp150.d3 = 0; tmp150.d2 = 0; tmp150.d1 = 0; tmp150.d0 = 0;
  // PERF: as div is only used here, use all those zeros directly in there and evaluate only d9 and d8, or keep all in d8 (30 bits, omit d9)

  div_150_75(&u,tmp150,f,ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
#ifdef CHECKS_MODBASECASE
                  ,modbasecase_debug
#endif
);						// u = floor(tmp150 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("barrett15_75: u=%x:%x:%x:%x:%x, ff=%G\n",
        u.d4.s0, u.d3.s0, u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif
    //PERF: min limit of bb? skip lower eval's?
  a.d0 = mad24(bb.d5, bit_max75_mult, (bb.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bb.d6, bit_max75_mult, (bb.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bb.d7, bit_max75_mult, (bb.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d3 = mad24(bb.d8, bit_max75_mult, (bb.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d4 = mad24(bb.d9, bit_max75_mult, (bb.d8 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)

  mul_75_150_no_low3(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("barrett15_75: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:%x:...\n",
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        tmp150.d9.s0, tmp150.d8.s0, tmp150.d7.s0, tmp150.d6.s0, tmp150.d5.s0, tmp150.d4.s0, tmp150.d3.s0);
#endif

  a.d0 = mad24(tmp150.d5, bit_max75_mult, (tmp150.d4 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = mad24(tmp150.d6, bit_max75_mult, (tmp150.d5 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d2 = mad24(tmp150.d7, bit_max75_mult, (tmp150.d6 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d3 = mad24(tmp150.d8, bit_max75_mult, (tmp150.d7 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d4 = mad24(tmp150.d9, bit_max75_mult, (tmp150.d8 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)

  mul_75(&tmp75, a, f);							// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("barrett15_75: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0, tmp75.d4.s0, tmp75.d3.s0, tmp75.d2.s0, tmp75.d1.s0, tmp75.d0.s0);
#endif
    // PERF: shouldn't all those bb's be 0, thus always require a borrow?
  tmp75.d0 = (bb.d0 - tmp75.d0) & 0x7FFF;
  tmp75.d1 = (bb.d1 - tmp75.d1 - AS_UINT_V((tmp75.d0 > bb.d0) ? 1 : 0 ));
  tmp75.d2 = (bb.d2 - tmp75.d2 - AS_UINT_V((tmp75.d1 > bb.d1) ? 1 : 0 ));
  tmp75.d3 = (bb.d3 - tmp75.d3 - AS_UINT_V((tmp75.d2 > bb.d2) ? 1 : 0 ));
  tmp75.d4 = (bb.d4 - tmp75.d4 - AS_UINT_V((tmp75.d3 > bb.d3) ? 1 : 0 ));
  tmp75.d1 &= 0x7FFF;
  tmp75.d2 &= 0x7FFF;
  tmp75.d3 &= 0x7FFF;
  tmp75.d4 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("barrett15_75: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (tmp)\n",
        bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, tmp75.d4.s0, tmp75.d3.s0, tmp75.d2.s0, tmp75.d1.s0, tmp75.d0.s0);
#endif
#ifndef CHECKS_MODBASECASE
  mod_simple_75(&a, tmp75, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 6;
  if(bit_max_75 == 2) limit = 8;						// bit_max == 65, due to decreased accuracy of mul_96_192_no_low2() above we need a higher threshold
  if(bit_max_75 == 3) limit = 7;						// bit_max == 66, ...
  mod_simple_75(&a, tmp75, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max_75, limit, modbasecase_debug);
#endif
  
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("barrett15_75: tmp=%x:%x:%x:%x:%x mod f=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x (a)\n",
        tmp75.d4.s0, tmp75.d3.s0, tmp75.d2.s0, tmp75.d1.s0, tmp75.d0.s0,
        f.d4.s0, f.d3.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_75_150(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        b.d9.s0, b.d8.s0, b.d7.s0, b.d6.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
#if (TRACE_KERNEL > 14)
    // verify squaring by dividing again.
    __private float_v f1 = CONVERT_FLOAT_V(mad24(a.d4, 32768, a.d3));
    f1= f1 * 32768.0f + CONVERT_FLOAT_V(a.d2);   // f.d1 needed?

    f1= as_float(0x3f7ffffb) / f1;		// just a little bit below 1.0f so we always underestimate the quotient
    div_150_75(&tmp75, b, a, f1, tid
#ifdef CHECKS_MODBASECASE
                  ,modbasecase_debug
#endif
              );
    if (tid==TRACE_TID) printf("vrfy: b = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x / a=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x\n",
        b.d9.s0, b.d8.s0, b.d7.s0, b.d6.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0,
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0, tmp75.d4.s0, tmp75.d3.s0, tmp75.d2.s0, tmp75.d1.s0, tmp75.d0.s0);
#endif
    a.d0 = mad24(b.d5, bit_max75_mult, (b.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(b.d6, bit_max75_mult, (b.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(b.d7, bit_max75_mult, (b.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(b.d8, bit_max75_mult, (b.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(b.d9, bit_max75_mult, (b.d8 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)

    mul_75_150_no_low3(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:%x:...\n",
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        tmp150.d9.s0, tmp150.d8.s0, tmp150.d7.s0, tmp150.d6.s0, tmp150.d5.s0, tmp150.d4.s0, tmp150.d3.s0);
#endif
    a.d0 = mad24(tmp150.d5, bit_max75_mult, (tmp150.d4 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = mad24(tmp150.d6, bit_max75_mult, (tmp150.d5 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d2 = mad24(tmp150.d7, bit_max75_mult, (tmp150.d6 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d3 = mad24(tmp150.d8, bit_max75_mult, (tmp150.d7 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d4 = mad24(tmp150.d9, bit_max75_mult, (tmp150.d8 >> bit_max_60))&0x7FFF;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)

    mul_75(&tmp75, a, f);						// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0, tmp75.d4.s0, tmp75.d3.s0, tmp75.d2.s0, tmp75.d1.s0, tmp75.d0.s0);
#endif
    tmp75.d0 = (b.d0 - tmp75.d0) & 0x7FFF;
    tmp75.d1 = (b.d1 - tmp75.d1 - AS_UINT_V((tmp75.d0 > b.d0) ? 1 : 0 ));
    tmp75.d2 = (b.d2 - tmp75.d2 - AS_UINT_V((tmp75.d1 > b.d1) ? 1 : 0 ));
    tmp75.d3 = (b.d3 - tmp75.d3 - AS_UINT_V((tmp75.d2 > b.d2) ? 1 : 0 ));
    tmp75.d4 = (b.d4 - tmp75.d4 - AS_UINT_V((tmp75.d3 > b.d3) ? 1 : 0 ));
    tmp75.d1 &= 0x7FFF;
    tmp75.d2 &= 0x7FFF;
    tmp75.d3 &= 0x7FFF;
    tmp75.d4 &= 0x7FFF;
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (tmp)\n",
        b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0, tmp75.d4.s0, tmp75.d3.s0, tmp75.d2.s0, tmp75.d1.s0, tmp75.d0.s0);
#endif
    if(exp&0x80000000)shl_75(&tmp75);					// "optional multiply by 2" in Prime 95 documentation

#ifndef CHECKS_MODBASECASE
    mod_simple_75(&a, tmp75, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    if(bit_max_75 == 2) limit = 8;					// bit_max == 65, due to decreased accuracy of mul_96_192_no_low2() above we need a higher threshold
    if(bit_max_75 == 3) limit = 7;					// bit_max == 66, ...
    mod_simple_75(&a, tmp75, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max_75, limit, modbasecase_debug);
#endif

    exp+=exp;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("loopend: exp=%x, tmp=%x:%x:%x:%x:%x mod f=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x (a)\n",
        exp, tmp75.d4.s0, tmp75.d3.s0, tmp75.d2.s0, tmp75.d1.s0, tmp75.d0.s0,
        f.d4.s0, f.d3.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  }


#ifndef CHECKS_MODBASECASE
  a = sub_if_gte_75(a,f);	// final adjustment in case a >= f
#else
  tmp75 = sub_if_gte_75(a,f);
  a = sub_if_gte_75(tmp75,f);
  if( tmp75.d0 != a.d0 )  // f is odd, so it is sufficient to compare the last part
  {
    printf("EEEEEK, final a was >= f\n");
  }
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x:%x:%x \n",
         a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  

/* finally check if we found a factor and write the factor to RES[] */
#if (BARRETT_VECTOR_SIZE == 1)
  if( ((a.d4|a.d3|a.d2|a.d1)==0 && a.d0==1) )
  {
/* in contrast to the other kernels this barrett based kernel is only allowed for factors above 2^60 so there is no need to check for f != 1 */  
    tid=ATOMIC_INC(RES[0]);
    if(tid<10)				/* limit to 10 factors per class */
    {
      RES[tid*3 + 1]=f.d4;
      RES[tid*3 + 2]=mad24(f.d3,0x8000u, f.d2);  // that's now 30 bits per int
      RES[tid*3 + 3]=mad24(f.d1,0x8000u, f.d0);  
    }
  }
#elif (BARRETT_VECTOR_SIZE == 2)
  EVAL_RES_d(x)
  EVAL_RES_d(y)
#elif (BARRETT_VECTOR_SIZE == 4)
  EVAL_RES_d(x)
  EVAL_RES_d(y)
  EVAL_RES_d(z)
  EVAL_RES_d(w)
#elif (BARRETT_VECTOR_SIZE == 8)
  EVAL_RES_d(s0)
  EVAL_RES_d(s1)
  EVAL_RES_d(s2)
  EVAL_RES_d(s3)
  EVAL_RES_d(s4)
  EVAL_RES_d(s5)
  EVAL_RES_d(s6)
  EVAL_RES_d(s7)
#elif (BARRETT_VECTOR_SIZE == 16)
  EVAL_RES_d(s0)
  EVAL_RES_d(s1)
  EVAL_RES_d(s2)
  EVAL_RES_d(s3)
  EVAL_RES_d(s4)
  EVAL_RES_d(s5)
  EVAL_RES_d(s6)
  EVAL_RES_d(s7)
  EVAL_RES_d(s8)
  EVAL_RES_d(s9)
  EVAL_RES_d(sa)
  EVAL_RES_d(sb)
  EVAL_RES_d(sc)
  EVAL_RES_d(sd)
  EVAL_RES_d(se)
  EVAL_RES_d(sf)
#endif
 
}

