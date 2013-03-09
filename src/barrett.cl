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

/****************************************
 ****************************************
 * 32-bit-stuff for the 92/76-bit-barrett-kernel
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


#ifndef CHECKS_MODBASECASE
void div_192_96(int96_v * const res, __private int192_v q, const int96_v n, const float_v nf);
void div_160_96(int96_v * const res, __private int192_v q, const int96_v n, const float_v nf);
#else
void div_192_96(int96_v * const res, __private int192_v q, const int96_v n, const float_v nf, __global uint* modcasebase_debug);
void div_160_96(int96_v * const res, __private int192_v q, const int96_v n, const float_v nf, __global uint* modcasebase_debug);
#endif
void mul_96(int96_v * const res, const int96_v a, const int96_v b);
void mul_96_192_no_low2(int192_v *const res, const int96_v a, const int96_v b);
void mul_96_192_no_low3(int192_v *const res, const int96_v a, const int96_v b);


/****************************************
 ****************************************
 * 32-bit based 79- and 92-bit barrett-kernels
 *
 ****************************************
 ****************************************/

int96_v sub_if_gte_96(const int96_v a, const int96_v b)
/* return (a>b)?a-b:a */
{
  int96_v tmp;
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

void inc_if_ge_96(int96_v * const res, const int96_v a, const int96_v b)
{ /* if (a >= b) res++ */
  __private uint_v ge, carry;

  ge = AS_UINT_V(a.d2 == b.d2);
  ge = AS_UINT_V(ge ? ((a.d1 == b.d1) ? (a.d0 >= b.d0) : (a.d1 > b.d1)) : (a.d2 > b.d2));

  carry    = AS_UINT_V(ge ? 1 : 0);
  res->d0 += carry;
  carry    = AS_UINT_V((carry > res->d0)? 1 : 0);
  res->d1 += carry;
  res->d2 += AS_UINT_V((carry > res->d1)? 1 : 0);
}

void mul_96(int96_v * const res, const int96_v a, const int96_v b)
/* res = a * b */
{
  __private uint_v tmp;

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


/*
   not used anymore
void mul_96_192_no_low2(int192_v * const res, const int96_v a, const int96_v b)
res ~= a * b
res.d0 and res.d1 are NOT computed. Carry from res.d1 to res.d2 is ignored,
too. So the digits res.d{2-5} might differ from mul_96_192(). In
mul_96_192() are two carries from res.d1 to res.d2. So ignoring the digits
res.d0 and res.d1 the result of mul_96_192_no_low() is 0 to 2 lower than
of mul_96_192().
{
  
  __private uint_v tmp;
  
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


  res->d5  = mad_hi(a.d2, b.d2, res->d5);
}
 */


void mul_96_192_no_low3(int192_v * const res, const int96_v a, const int96_v b)
/*
res ~= a * b
res.d0, res.d1 and res.d2 are NOT computed. Carry to res.d3 is ignored,
too. So the digits res.d{3-5} might differ from mul_96_192(). In
mul_96_192() are four carries from res.d2 to res.d3. So ignoring the digits
res.d0, res.d1 and res.d2 the result of mul_96_192_no_low() is 0 to 4 lower
than of mul_96_192().
 */
{
  
  __private uint_v tmp;

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


void square_96_192(int192_v * const res, const int96_v a)
/* res = a^2 = a.d0^2 + a.d1^2 + a.d2^2 + 2(a.d0*a.d1 + a.d0*a.d2 + a.d1*a.d2) */
{
/*
highest possible value for x * x is 0xFFFFFFF9
this occurs for x = {479772853, 1667710795, 2627256501, 3815194443}
Adding x*x to a few carries will not cascade the carry
*/
  __private uint_v tmp;

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

void square_96_160(int192_v * const res, const int96_v a)
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
  __private uint_v tmp, TWOad2 = a.d2 << 1; // a.d2 < 2^16 so this always fits

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

void shl_96(int96_v * const a)
/* shiftleft a one bit */
{ /* here, bitalign improves the 92-bit kernel, and slows down 76-bit */
  a->d2 = amd_bitalign(a->d2, a->d1, 31);
  a->d1 = amd_bitalign(a->d1, a->d0, 31);
//  a->d2 = (a->d2 << 1) | (a->d1 >> 31);
//  a->d1 = (a->d1 << 1) | (a->d0 >> 31);
  a->d0 = a->d0 << 1;
}

void shl_192(int192_v * const a)
/* shiftleft a one bit */
{ /* in this function, bitalign slows down all kernels */
//  a->d5 = amd_bitalign(a->d5, a->d4, 31);
//  a->d4 = amd_bitalign(a->d4, a->d3, 31);
//  a->d3 = amd_bitalign(a->d3, a->d2, 31);
//  a->d2 = amd_bitalign(a->d2, a->d1, 31);
//  a->d1 = amd_bitalign(a->d1, a->d0, 31);
  a->d5 = (a->d5 << 1) | (a->d4 >> 31);
  a->d4 = (a->d4 << 1) | (a->d3 >> 31);
  a->d3 = (a->d3 << 1) | (a->d2 >> 31);
  a->d2 = (a->d2 << 1) | (a->d1 >> 31);
  a->d1 = (a->d1 << 1) | (a->d0 >> 31);
  a->d0 = a->d0 << 1;
}


#undef DIV_160_96
#ifndef CHECKS_MODBASECASE
void div_192_96(int96_v * const res, __private int192_v q, const int96_v n, const float_v nf)
#else
void div_192_96(int96_v * const res, __private int192_v q, const int96_v n, const float_v nf, __global uint * restrict modbasecase_debug)
#endif
/* res = q / n (integer division) */
{
  __private float_v qf;
  __private uint_v qi, tmp, carry;
  __private int192_v nn;
  __private int96_v tmp96;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf= CONVERT_FLOAT_V(q.d5);
  qf= qf * 4294967296.0f;  // combining this and the the below 2M multiplier makes it slower!
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
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1 * qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V((tmp > nn.d1)? 1 : 0);
  tmp    = mul_hi(n.d1, qi);
  nn.d2 += tmp;
#ifndef DIV_160_96
  nn.d3  = AS_UINT_V((tmp > nn.d2)? 1 : 0);
  tmp    = n.d2 * qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V((tmp > nn.d2)? 1 : 0);
  nn.d3 += mul_hi(n.d2, qi);
#else
  nn.d2 += n.d2 * qi;
#endif

// shiftleft nn 11 bits
#ifndef DIV_160_96
  nn.d3 = (nn.d3 << 11) + (nn.d2 >> 21);
#endif
  nn.d2 = amd_bitalign(nn.d2, nn.d1, 21);
  nn.d1 = amd_bitalign(nn.d1, nn.d0, 21);
//  nn.d2 = (nn.d2 << 11) + (nn.d1 >> 21);
//  nn.d1 = (nn.d1 << 11) + (nn.d0 >> 21);
  nn.d0 =  nn.d0 << 11;

//  q = q - nn
  carry= AS_UINT_V((nn.d0 > q.d2)? 1 : 0);
  q.d2 = q.d2 - nn.d0;

  tmp  = q.d3 - nn.d1 - carry ;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

#ifndef DIV_160_96
  tmp  = q.d4 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)))? 1 : 0);
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d3 - carry;
#else
  q.d4 = q.d4 - nn.d2 - carry;
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
  nn.d0 = n.d0 * qi;
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

  // shiftleft nn 23 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                  nn.d3 >> 9;
#endif  
//  nn.d3 = amd_bitalign(nn.d3, nn.d2, 9);
  nn.d3 = (nn.d3 << 23) + (nn.d2 >> 9);
  nn.d2 = amd_bitalign(nn.d2, nn.d1, 9);
//  nn.d2 = (nn.d2 << 23) + (nn.d1 >> 9);
//  nn.d1 = amd_bitalign(nn.d1, nn.d0, 9);
  nn.d1 = (nn.d1 << 23) + (nn.d0 >> 9);
  nn.d0 =  nn.d0 << 23;

// q = q - nn
  carry= AS_UINT_V((nn.d0 > q.d1) ? 1 : 0);
  q.d1 = q.d1 - nn.d0;

  tmp  = q.d2 - nn.d1 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)))? 1 : 0);
  q.d2 = tmp;

  tmp  = q.d3 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

#ifdef CHECKS_MODBASECASE
  tmp  = q.d4 - nn.d3 - carry;
  carry= AS_UINT_V(((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)))? 1 : 0);
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d4 - carry;
#else
  q.d4 = q.d4 - nn.d3 - carry;
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
  
  nn.d0 = n.d0 * qi;
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

//  q = q - nn
  carry= AS_UINT_V((nn.d0 > q.d1) ? 1 : 0);
  q.d1 = q.d1 - nn.d0;

  tmp  = q.d2 - nn.d1 - carry;
  carry= AS_UINT_V(((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)))? 1 : 0);
  q.d2 = tmp;

  tmp  = q.d3 - nn.d2 - carry;
  carry= AS_UINT_V(((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)))? 1 : 0);
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d3 - carry;

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
  
  return;

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
void div_160_96(int96_v * const res, __private int192_v q, const int96_v n, const float_v nf)
#else
void div_160_96(int96_v * const res, __private int192_v q, const int96_v n, const float_v nf, __global uint * restrict modbasecase_debug)
#endif
/* res = q / n (integer division) */
/* the code of div_160_96() is an EXACT COPY of div_192_96(), the only
difference is that the 160bit version ignores the most significant
word of q (q.d5) because it assumes it is 0. This is controlled by defining
DIV_160_96 here. */
{
  __private float_v qf;
  __private uint_v qi, tmp, carry;
  __private int192_v nn;
  __private int96_v tmp96;

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

  return;

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


__kernel void cl_barrett32_92(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v b, tmp192;
  __private int96_v tmp96;
  __private float_v ff;
  __private int bit_max65 = bit_max64 - 1; /* used for bit shifting... */
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_92: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, bit_max=%d\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, bit_max64+64);
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
  //tmp  = t * 4620u; // NUM_CLASSES
  //k.d0 = k_base.d0 + tmp;
  //k.d1 = k_base.d1 + mul_hi(t, 4620u) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  // this mad_hi actually improves performance!
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mad_hi(t, 4620u, k_base.d1) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_92: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp); // this mad_hi is good
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  // here, a mad_hi would be bad
  f.d2 += mul_hi(k.d1, exp96.d0) + AS_UINT_V((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_92: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 
//  ff= 1.0f / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 
        
        
  tmp192.d5 = 1 << bit_max65;			  // tmp192 = 2^(95 + bits_in_f)
  tmp192.d4 = 0; tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);				// u = floor(2^(95 + bits_in_f) / f), giving 96 bits of precision
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);
#endif
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("cl_barrett32_92: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_92: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);					// tmp96 = quotient * f, we only compute the low 96-bits here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_92: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  tmp96.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_92: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );                         // Adjustment.  The code above may produce an a that is too large by up to 6 times f.

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
    if (tid==TRACE_TID) printf("cl_barrett32_92: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {                                 // On input a is at most 93 bits (see end of this loop)
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32); // this here is the reason bit_max needs to be 66 at least:
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32); // OpenCL does not shift by 32 bits

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
							// The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
							// into tmp192.d2 which could have generated 4 carries into tmp192.d3.
							// Also, since u was generated with the floor function, it could be low by up to
							// almost 1.  If we account for this a value up to a.d2 could have been added into
							// tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
							// function, and could thus be low by almost 1.  If we account for this a value up
							// to u.d2 could have been added into tmp192.d2 possibly generating a carry.
							// A grand total of up to 6 carries lost.

    mul_96(&tmp96, a, f);				// tmp96 = quotient * f, we only compute the low 96-bits here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp96.d1 = b.d1 - tmp96.d1 - carry;
    carry    = AS_UINT_V(((tmp96.d1 > b.d1) || (carry && AS_UINT_V(tmp96.d1 == b.d1))) ? 1 : 0);

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
							// At this point tmp96 is 92 bits + log2 (6) bits to account for the fact that
							// the quotient was up to 6 too small.  This is 94.585 bits.

    if(exp&0x80000000)shl_96(&tmp96);			// Optional multiply by 2.  At this point tmp96 can be 95.585 bits.

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 );
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

    exp+=exp;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("loopend: exp=%d, tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        exp, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
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

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_92: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_92: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
__kernel void cl_barrett32_87(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v b, tmp192;
  __private int96_v tmp96;
  __private float_v ff;
  __private int bit_max65 = bit_max64 - 1; /* used for bit shifting... */
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_87: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, bit_max=%d\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, bit_max64+64);
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
  //tmp  = t * 4620u; // NUM_CLASSES
  //k.d0 = k_base.d0 + tmp;
  //k.d1 = k_base.d1 + mul_hi(t, 4620u) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mul_hi(t, 4620u)+ k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */  // bad mad_hi
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_87: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);  // good mad_hi
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mad_hi(k.d1, exp96.d0, AS_UINT_V((tmp > f.d1)? 1 : 0)); 	// f = 2 * k * exp + 1  // good mad_hi

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_87: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

//  ff= as_float(0x3f7ffffd) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.
  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 
        
  tmp192.d5 = 1 << bit_max65;			  // tmp192 = 2^(95 + bits_in_f)
  tmp192.d4 = 0; tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);						// u = floor(tmp192 / f)
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(tmp192 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("cl_barrett32_87: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_87: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_87: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  a.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  a.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_87: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        bb.d2, bb.d1, bb.d0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {                                 // On input a is at most 93 bits (see end of this loop)
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32);
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32);

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif
    a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
							// The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
							// into tmp192.d2 which could have generated 4 carries into tmp192.d3.
							// Also, since u was generated with the floor function, it could be low by up to
							// almost 1.  If we account for this a value up to a.d2 could have been added into
							// tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
							// function, and could thus be low by almost 1.  If we account for this a value up
							// to u.d2 could have been added into tmp192.d2 possibly generating a carry.
							// A grand total of up to 6 carries lost.

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    a.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    a.d1 = tmp;

    a.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
    if(exp&0x80000000)
    {
      shl_96(&a);					// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 1)
      if (tid==TRACE_TID) printf("loop shl: %x:%x:%x (a)\n",
        a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
    }
    exp+=exp;
  }

  tmp96.d0 = a.d0;
  tmp96.d1 = a.d1;
  tmp96.d2 = a.d2;

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

#ifndef CHECKS_MODBASECASE
  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#else
  tmp96 = sub_if_gte_96(a,f);
  a = sub_if_gte_96(tmp96,f);
  if( tmp96.d0 != a.d0 )
  {
    printf("EEEEEK, final a was >= f\n");
  }
#endif

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_87: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_87: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
__kernel void cl_barrett32_88(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v b, tmp192;
  __private int96_v tmp96;
  __private float_v ff;
  __private int bit_max65 = bit_max64 - 1; /* used for bit shifting... */
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_88: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, bit_max=%d\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, bit_max64+64);
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
  //tmp  = t * 4620u; // NUM_CLASSES
  //k.d0 = k_base.d0 + tmp;
  //k.d1 = k_base.d1 + mul_hi(t, 4620u) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mul_hi(t, 4620u) + k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */  // bad mad_hi
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_88: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);  // good mad_hi
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mad_hi(k.d1, exp96.d0, AS_UINT_V((tmp > f.d1)? 1 : 0)); 	// f = 2 * k * exp + 1  // good mad_hi

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_88: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

//  ff= as_float(0x3f7ffffd) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.
  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 
        
  tmp192.d5 = 1 << bit_max65;			  // tmp192 = 2^(95 + bits_in_f)
  tmp192.d4 = 0; tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);						// u = floor(tmp192 / f)
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(tmp192 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("cl_barrett32_88: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_88: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_88: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  a.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  a.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_88: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        bb.d2, bb.d1, bb.d0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {                                 // On input a is at most 93 bits (see end of this loop)
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits
    if(exp&0x80000000)shl_192(&b);	// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32);
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32);

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif
    a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
							// The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
							// into tmp192.d2 which could have generated 4 carries into tmp192.d3.
							// Also, since u was generated with the floor function, it could be low by up to
							// almost 1.  If we account for this a value up to a.d2 could have been added into
							// tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
							// function, and could thus be low by almost 1.  If we account for this a value up
							// to u.d2 could have been added into tmp192.d2 possibly generating a carry.
							// A grand total of up to 6 carries lost.

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    a.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    a.d1 = tmp;

    a.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif

    exp+=exp;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("loopend: exp=%d, tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        exp, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  }

  tmp96.d0 = a.d0;
  tmp96.d1 = a.d1;
  tmp96.d2 = a.d2;

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    if(bit_max64 == 15) limit = 9;
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

#ifndef CHECKS_MODBASECASE
  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#else
  tmp96 = sub_if_gte_96(a,f);
  a = sub_if_gte_96(tmp96,f);
  if (tmp96.d0 != a.d0)
  {
    printf("EEEEEK, final a was >= f\n");
  }
#endif

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_88: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_88: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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

__kernel void cl_barrett32_79(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v tmp192, b;
  __private int96_v tmp96;
  __private float_v ff;
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_79: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
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
//MAD only available for float
  k.d0 = mad24(t, 4620u, k_base.d0);
//  k.d1 = mad_hi(t, 4620u, k_base.d1) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  k.d1 = mul_hi(t, 4620u) + k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */  // bad mad_hi
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);   // good mad_hi
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mul_hi(k.d1, exp96.d0) + AS_UINT_V((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1   // bad mad_hi

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_79: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 

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
    if (tid==TRACE_TID) printf("cl_barrett32_79: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  tmp96.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("cl_barrett32_79: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
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
    if (tid==TRACE_TID) printf("cl_barrett32_79: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
 
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
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
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

    exp+=exp;
  }

  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_79: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  
/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_79: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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

__kernel void cl_barrett32_76(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v tmp192, b;
  __private int96_v tmp96;
  __private float_v ff;
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={0, 0, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_barrett76: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
#endif

#if (VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
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
  k.d1 = mul_hi(t, 4620u) + k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);  // good mad_hi
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;
  f.d0  = k.d0 * exp96.d0 + 1;

  f.d2 += mul_hi(k.d1, exp96.d0) + AS_UINT_V((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1  // good mad_hi

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
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
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

  carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = bb.d0 - tmp96.d0;

  tmp  = bb.d1 - tmp96.d1 - carry;
  carry= AS_UINT_V(((tmp > bb.d1) || (carry && AS_UINT_V(tmp == bb.d1))) ? 1 : 0);
  tmp96.d1 = tmp;

  tmp96.d2 = bb.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mfakto_cl_barrett76: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

#if 0
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
#else
  a.d0=tmp96.d0;
  a.d1=tmp96.d1;
  a.d2=tmp96.d2;
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
 
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#if 0
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
#else
  a.d0=tmp96.d0;
  a.d1=tmp96.d1;
  a.d2=tmp96.d2;
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

    exp+=exp;
  }

#if 1
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
#endif

  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x \n",
         a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

  
/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("mfakto_cl_barrett76: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
__kernel void cl_barrett32_77(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v tmp192, b;
  __private int96_v tmp96;
  __private float_v ff;
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_77: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
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
//MAD only available for float
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mul_hi(t, 4620u) + k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_77: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mad_hi(k.d1, exp96.d0, AS_UINT_V((tmp > f.d1)? 1 : 0)); 	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_77: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.

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
    if (tid==TRACE_TID) printf("cl_barrett32_77: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_77: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_77: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  a.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  a.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("cl_barrett32_77: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
 
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2
    if(exp&0x80000000)shl_192(&b);	// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    a.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    a.d1 = tmp;

    a.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    exp+=exp;
  }

  tmp96.d0 = a.d0;
  tmp96.d1 = a.d1;
  tmp96.d2 = a.d2;

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

  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_77: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_77: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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

// a copy of the 79-bit barrett for testing the effect of (not) sieving
__kernel void cl_barrett32_79_ns(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v tmp192, b;
  __private int96_v tmp96;
  __private float_v ff;
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("cl_barrett32_79-ns: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
#endif

#if (VECTOR_SIZE == 1)
  t    = tid;
#elif (VECTOR_SIZE == 2)
  t.x  = tid;
  t.y  = tid+1;
#elif (VECTOR_SIZE == 3)
  t.x  = tid;
  t.y  = tid+1;
  t.z  = tid+2;
#elif (VECTOR_SIZE == 4)
  t.x  = tid;
  t.y  = tid+1;
  t.z  = tid+2;
  t.w  = tid+3;
#elif (VECTOR_SIZE == 8)
  t.s0 = tid;
  t.s1 = tid+1;
  t.s2 = tid+2;
  t.s3 = tid+3;
  t.s4 = tid+4;
  t.s5 = tid+5;
  t.s6 = tid+6;
  t.s7 = tid+7;
#elif (VECTOR_SIZE == 16)
  t.s0 = tid;
  t.s1 = tid+1;
  t.s2 = tid+2;
  t.s3 = tid+3;
  t.s4 = tid+4;
  t.s5 = tid+5;
  t.s6 = tid+6;
  t.s7 = tid+7;
  t.s8 = tid+8;
  t.s9 = tid+9;
  t.sa = tid+10;
  t.sb = tid+11;
  t.sc = tid+12;
  t.sd = tid+13;
  t.se = tid+14;
  t.sf = tid+15;
#endif
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mad_hi(t, 4620u, k_base.d1) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79_ns: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mad_hi(k.d1, exp96.d0, AS_UINT_V((tmp > f.d1)? 1 : 0)); 	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_79_ns: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.

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
    if (tid==TRACE_TID) printf("cl_barrett32_79_ns: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79_ns: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79_ns: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  tmp96.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("cl_barrett32_79_ns: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
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
    if (tid==TRACE_TID) printf("cl_barrett32_79_ns: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
 
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
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
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

    exp+=exp;
  }

  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x \n",
         a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

  
/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_88: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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

#ifdef CL_GPU_SIEVE
/****************************************
 ****************************************
 * 32-bit-kernel consuming the GPU sieve
 * included by main kernel file
 ****************************************
 ****************************************/


__kernel void cl_barrett32_92_gs(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v b, tmp192;
  __private int96_v tmp96;
  __private float_v ff;
  __private int bit_max65 = bit_max64 - 1; /* used for bit shifting... */
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_92: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, bit_max=%d\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, bit_max64+64);
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
  //tmp  = t * 4620u; // NUM_CLASSES
  //k.d0 = k_base.d0 + tmp;
  //k.d1 = k_base.d1 + mul_hi(t, 4620u) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  // this mad_hi actually improves performance!
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mad_hi(t, 4620u, k_base.d1) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_92: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp); // this mad_hi is good
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  // here, a mad_hi would be bad
  f.d2 += mul_hi(k.d1, exp96.d0) + AS_UINT_V((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_92: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 
//  ff= 1.0f / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 
        
        
  tmp192.d5 = 1 << bit_max65;			  // tmp192 = 2^(95 + bits_in_f)
  tmp192.d4 = 0; tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);				// u = floor(2^(95 + bits_in_f) / f), giving 96 bits of precision
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);
#endif
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("cl_barrett32_92: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_92: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);					// tmp96 = quotient * f, we only compute the low 96-bits here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_92: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  tmp96.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_92: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );                         // Adjustment.  The code above may produce an a that is too large by up to 6 times f.

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
    if (tid==TRACE_TID) printf("cl_barrett32_92: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {                                 // On input a is at most 93 bits (see end of this loop)
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32); // this here is the reason bit_max needs to be 66 at least:
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32); // OpenCL does not shift by 32 bits

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
							// The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
							// into tmp192.d2 which could have generated 4 carries into tmp192.d3.
							// Also, since u was generated with the floor function, it could be low by up to
							// almost 1.  If we account for this a value up to a.d2 could have been added into
							// tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
							// function, and could thus be low by almost 1.  If we account for this a value up
							// to u.d2 could have been added into tmp192.d2 possibly generating a carry.
							// A grand total of up to 6 carries lost.

    mul_96(&tmp96, a, f);				// tmp96 = quotient * f, we only compute the low 96-bits here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp96.d1 = b.d1 - tmp96.d1 - carry;
    carry    = AS_UINT_V(((tmp96.d1 > b.d1) || (carry && AS_UINT_V(tmp96.d1 == b.d1))) ? 1 : 0);

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
							// At this point tmp96 is 92 bits + log2 (6) bits to account for the fact that
							// the quotient was up to 6 too small.  This is 94.585 bits.

    if(exp&0x80000000)shl_96(&tmp96);			// Optional multiply by 2.  At this point tmp96 can be 95.585 bits.

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 );
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

    exp+=exp;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("loopend: exp=%d, tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        exp, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
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

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_92: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_92: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
__kernel void cl_barrett32_87_gs(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v b, tmp192;
  __private int96_v tmp96;
  __private float_v ff;
  __private int bit_max65 = bit_max64 - 1; /* used for bit shifting... */
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_87: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, bit_max=%d\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, bit_max64+64);
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
  //tmp  = t * 4620u; // NUM_CLASSES
  //k.d0 = k_base.d0 + tmp;
  //k.d1 = k_base.d1 + mul_hi(t, 4620u) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mul_hi(t, 4620u)+ k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */  // bad mad_hi
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_87: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);  // good mad_hi
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mad_hi(k.d1, exp96.d0, AS_UINT_V((tmp > f.d1)? 1 : 0)); 	// f = 2 * k * exp + 1  // good mad_hi

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_87: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

//  ff= as_float(0x3f7ffffd) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.
  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 
        
  tmp192.d5 = 1 << bit_max65;			  // tmp192 = 2^(95 + bits_in_f)
  tmp192.d4 = 0; tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);						// u = floor(tmp192 / f)
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(tmp192 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("cl_barrett32_87: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_87: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_87: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  a.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  a.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_87: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        bb.d2, bb.d1, bb.d0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {                                 // On input a is at most 93 bits (see end of this loop)
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32);
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32);

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif
    a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
							// The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
							// into tmp192.d2 which could have generated 4 carries into tmp192.d3.
							// Also, since u was generated with the floor function, it could be low by up to
							// almost 1.  If we account for this a value up to a.d2 could have been added into
							// tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
							// function, and could thus be low by almost 1.  If we account for this a value up
							// to u.d2 could have been added into tmp192.d2 possibly generating a carry.
							// A grand total of up to 6 carries lost.

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    a.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    a.d1 = tmp;

    a.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
    if(exp&0x80000000)
    {
      shl_96(&a);					// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 1)
      if (tid==TRACE_TID) printf("loop shl: %x:%x:%x (a)\n",
        a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
    }
    exp+=exp;
  }

  tmp96.d0 = a.d0;
  tmp96.d1 = a.d1;
  tmp96.d2 = a.d2;

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

#ifndef CHECKS_MODBASECASE
  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#else
  tmp96 = sub_if_gte_96(a,f);
  a = sub_if_gte_96(tmp96,f);
  if( tmp96.d0 != a.d0 )
  {
    printf("EEEEEK, final a was >= f\n");
  }
#endif

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_87: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_87: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
__kernel void cl_barrett32_88_gs(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v b, tmp192;
  __private int96_v tmp96;
  __private float_v ff;
  __private int bit_max65 = bit_max64 - 1; /* used for bit shifting... */
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_88: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, bit_max=%d\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, bit_max64+64);
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
  //tmp  = t * 4620u; // NUM_CLASSES
  //k.d0 = k_base.d0 + tmp;
  //k.d1 = k_base.d1 + mul_hi(t, 4620u) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mul_hi(t, 4620u) + k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */  // bad mad_hi
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_88: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);  // good mad_hi
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mad_hi(k.d1, exp96.d0, AS_UINT_V((tmp > f.d1)? 1 : 0)); 	// f = 2 * k * exp + 1  // good mad_hi

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_88: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif
/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

//  ff= as_float(0x3f7ffffd) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.
  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 
        
  tmp192.d5 = 1 << bit_max65;			  // tmp192 = 2^(95 + bits_in_f)
  tmp192.d4 = 0; tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);						// u = floor(tmp192 / f)
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(tmp192 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("cl_barrett32_88: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_88: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_88: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  a.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  a.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_88: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        bb.d2, bb.d1, bb.d0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {                                 // On input a is at most 93 bits (see end of this loop)
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits
    if(exp&0x80000000)shl_192(&b);	// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32);
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32);

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0, tmp192.d2.s0);
#endif
    a.d0 = tmp192.d3;					// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
							// The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
							// into tmp192.d2 which could have generated 4 carries into tmp192.d3.
							// Also, since u was generated with the floor function, it could be low by up to
							// almost 1.  If we account for this a value up to a.d2 could have been added into
							// tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
							// function, and could thus be low by almost 1.  If we account for this a value up
							// to u.d2 could have been added into tmp192.d2 possibly generating a carry.
							// A grand total of up to 6 carries lost.

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    a.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    a.d1 = tmp;

    a.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif

    exp+=exp;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("loopend: exp=%d, tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        exp, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  }

  tmp96.d0 = a.d0;
  tmp96.d1 = a.d1;
  tmp96.d2 = a.d2;

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    if(bit_max64 == 15) limit = 9;
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

#ifndef CHECKS_MODBASECASE
  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#else
  tmp96 = sub_if_gte_96(a,f);
  a = sub_if_gte_96(tmp96,f);
  if (tmp96.d0 != a.d0)
  {
    printf("EEEEEK, final a was >= f\n");
  }
#endif

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_88: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_88: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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

__kernel void cl_barrett32_79_gs(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v tmp192, b;
  __private int96_v tmp96;
  __private float_v ff;
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_79: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
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
//MAD only available for float
  k.d0 = mad24(t, 4620u, k_base.d0);
//  k.d1 = mad_hi(t, 4620u, k_base.d1) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  k.d1 = mul_hi(t, 4620u) + k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */  // bad mad_hi
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);   // good mad_hi
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;

  f.d2 += mul_hi(k.d1, exp96.d0) + AS_UINT_V((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1   // bad mad_hi

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_79: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero. 

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
    if (tid==TRACE_TID) printf("cl_barrett32_79: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_79: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  tmp96.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("cl_barrett32_79: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
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
    if (tid==TRACE_TID) printf("cl_barrett32_79: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
 
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
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
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

    exp+=exp;
  }

  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_79: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
  
/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_79: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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

__kernel void cl_barrett32_76_gs(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private int96_t exp96;
  __private int96_v a, u, f, k;
  __private int192_v tmp192, b;
  __private int96_v tmp96;
  __private float_v ff;
  __private uint tid;
  __private uint_v t, tmp, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={0, 0, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_barrett76: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
#endif

#if (VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
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
  k.d1 = mul_hi(t, 4620u) + k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp   = exp96.d1 ? k.d0 : 0;  /* exp96.d1 is 0 or 1 */
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp);  // good mad_hi
  f.d2 += AS_UINT_V((tmp > f.d1)? 1 : 0);

  tmp   = k.d1 * exp96.d0;
  f.d1 += tmp;
  f.d0  = k.d0 * exp96.d0 + 1;

  f.d2 += mul_hi(k.d1, exp96.d0) + AS_UINT_V((tmp > f.d1)? 1 : 0); 	// f = 2 * k * exp + 1  // good mad_hi

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
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
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

  carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = bb.d0 - tmp96.d0;

  tmp  = bb.d1 - tmp96.d1 - carry;
  carry= AS_UINT_V(((tmp > bb.d1) || (carry && AS_UINT_V(tmp == bb.d1))) ? 1 : 0);
  tmp96.d1 = tmp;

  tmp96.d2 = bb.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mfakto_cl_barrett76: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

#if 0
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
#else
  a.d0=tmp96.d0;
  a.d1=tmp96.d1;
  a.d2=tmp96.d2;
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mfakto_cl_barrett76: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif
 
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1))) ? 1 : 0);
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#if 0
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
#else
  a.d0=tmp96.d0;
  a.d1=tmp96.d1;
  a.d2=tmp96.d2;
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

    exp+=exp;
  }

#if 1
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
#endif

  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("after sub: a = %x:%x:%x \n",
         a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

  
/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("mfakto_cl_barrett76: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
__kernel void cl_barrett32_77_gs(__private uint exp, const int96_t k_base, const __global uint * restrict bit_array, const uint bits_to_process, __local ushort *smem, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  int i, words_per_thread, initial_shifter_value, sieve_word, k_bit_base, total_bit_count;
  __local volatile ushort bitcount[256];	// Each thread of our block puts bit-counts here
  __private int96_t exp96, my_k_base, f_base;
  __private int96_v a, u, f, k;
  __private int192_v tmp192, b;
  __private int96_v tmp96;
  __private float_v ff;
  __private uint tid, tmp;
  __private uint_v tmp_v, carry;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif
  // Get pointer to section of the bit_array this thread is processing.

  words_per_thread = bits_to_process / 8192;
  bit_array += get_group_id(0) * bits_to_process / 32 + get_local_id(0) * words_per_thread;

// Count number of bits set in this thread's word(s) from the bit_array

  bitcount[get_local_id(0)] = 0;
  for (i = 0; i < words_per_thread; i++)
    bitcount[get_local_id(0)] +=  popcount(bit_array[i]);

// Create total count of bits set in block up to and including this threads popcnt.
// Kudos to Rocke Verser for the population counting code.
// CAUTION:  Following requires 256 threads per block

  // First five tallies remain within one warp.  Should be in lock-step.
  // AMD devs always run 16 threads at once => just 4 tallies
  if (get_local_id(0) & 1)        // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
    bitcount[get_local_id(0)] += bitcount[get_local_id(0) - 1];

  if (get_local_id(0) & 2)        // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
    bitcount[get_local_id(0)] += bitcount[(get_local_id(0) - 2) | 1];

  if (get_local_id(0) & 4)        // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
    bitcount[get_local_id(0)] += bitcount[(get_local_id(0) - 4) | 3];

  if (get_local_id(0) & 8)        // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
    bitcount[get_local_id(0)] += bitcount[(get_local_id(0) - 8) | 7];

  if (get_local_id(0) & 16)       // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
    bitcount[get_local_id(0)] += bitcount[(get_local_id(0) - 16) | 15];

  // Further tallies are across warps.  Must synchronize
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0)  & 32)      // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
    bitcount[get_local_id(0)] += bitcount[(get_local_id(0) - 32) | 31];

  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) & 64)       // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
    bitcount[get_local_id(0)] += bitcount[(get_local_id(0) - 64) | 63];

  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) & 128)       // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
    bitcount[get_local_id(0)] += bitcount[127];

  // At this point, bitcount[...] contains the total number of bits for the indexed
  // thread plus all lower-numbered threads.  I.e., bitcount[255] is the total count.

  barrier(CLK_LOCAL_MEM_FENCE);
  total_bit_count = bitcount[255];

//POSSIBLE OPTIMIZATION - bitcounts and smem could use the same memory space if we'd read bitcount into a register
// and sync threads before doing any writes to smem.

//POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

// Loop til this thread's section of the bit array is finished.

  sieve_word = *bit_array;
  k_bit_base = get_local_id(0) * words_per_thread * 32;
  for (i = total_bit_count - bitcount[get_local_id(0)]; ; i++) {
    int bit_to_test;

// Make sure we have a non-zero sieve word

    while (sieve_word == 0) {
      if (--words_per_thread == 0) break;
      sieve_word = *++bit_array;
      k_bit_base += 32;
    }

// Check if this thread has processed all its set bits

    if (sieve_word == 0) break;

// Find a bit to test in the sieve word

    bit_to_test = 31 - clz(sieve_word);
    sieve_word &= ~(1 << bit_to_test);

// Copy the k value to the shared memory array

    smem[i] = k_bit_base + bit_to_test;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested

  initial_shifter_value = exp << (32 - shiftcount);	// Initial shifter value

// Compute factor corresponding to first sieve bit in this block.

  // Compute base k value
  my_k_base.d0 = mad24(NUM_CLASSES, mul24(bits_to_process, get_group_id(0)), k_base.d0);
  my_k_base.d1 = k_base.d1 + mul_hi(NUM_CLASSES, mul24(bits_to_process, get_group_id(0))) + (k_base.d0 > my_k_base.d0)? 1u : 0u;	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  // k_base.d0 = __add_cc (k_base.d0, __umul32  (blockIdx.x * bits_to_process, NUM_CLASSES));
  // k_base.d1 = __addc   (k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

  // Compute k * exp
  f_base.d0 = my_k_base.d0 * exp;
  tmp       = mul_hi(my_k_base.d0, exp);
  f_base.d1 = tmp + my_k_base.d1 * exp;
  f_base.d2 = mul_hi(my_k_base.d1, exp) + (tmp > f_base.d1)? 1 : 0;

  // f_base.d0 =                                      __umul32(k_base.d0, exp);
  // f_base.d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
  // f_base.d2 = __addc  (__umul32hi(k_base.d1, exp),                       0);

  // Compute f_base = 2 * k * exp + 1
  f_base.d2 = amd_bitalign(f_base.d2, f_base.d1, 31);
  f_base.d1 = amd_bitalign(f_base.d1, f_base.d0, 31);
//  f_base.d2 = (f_base.d2 << 1) | (f_base.d1 >> 31);
//  f_base.d1 = (f_base.d1 << 1) | (f_base.d0 >> 31);
  f_base.d0 = f_base.d0 << 1 + 1;

// Loop til the k values written to shared memory are exhausted

  for (i = get_local_id(0); i < total_bit_count; i += 256) // THREADS_PER_BLOCK
  {
    int96_v f;
    uint k_delta;

// Get the (k - k_base) value to test

    k_delta = smem[i];  // PERF: we should read VECTORSIZE elements per thread

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

    f.d0 = f_base.d0 + mul24(k_delta, 2u * NUM_CLASSES) * exp;
    f.d1 = f_base.d1 + mul_hi(mul24(k_delta, 2u * NUM_CLASSES), exp) + (f_base.d0 > f.d0)? 1u : 0u;
    f.d2 = f_base.d2 + (f_base.d1 > f.d1 || (f_base.d1 == f.d1 && f_base.d0 > f.d0))? 1u : 0u;

//    f.d0 = __add_cc (f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
//    f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
//    f.d2 = __addc   (f_base.d2, 0);

/*    test_FC96_barrett79(f, b_preinit, initial_shifter_value, RES
#ifdef CHECKS_MODBASECASE
                        , bit_max64, modbasecase_debug
#endif
                        );
  }
}
	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

  exp96.d1=exp>>31;exp96.d0=exp+exp;	// exp96 = 2 * exp

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_77: exp=%d, x2=%x:%x, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, exp96.d1, exp96.d0, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
#endif

#if (VECTOR_SIZE == 1)
  tmp_v    = k_tab[tid];
#elif (VECTOR_SIZE == 2)
  tmp_v.x  = k_tab[tid];
  tmp_v.y  = k_tab[tid+1];
#elif (VECTOR_SIZE == 3)
  tmp_v.x  = k_tab[tid];
  tmp_v.y  = k_tab[tid+1];
  tmp_v.z  = k_tab[tid+2];
#elif (VECTOR_SIZE == 4)
  tmp_v.x  = k_tab[tid];
  tmp_v.y  = k_tab[tid+1];
  tmp_v.z  = k_tab[tid+2];
  tmp_v.w  = k_tab[tid+3];
#elif (VECTOR_SIZE == 8)
  tmp_v.s0 = k_tab[tid];
  tmp_v.s1 = k_tab[tid+1];
  tmp_v.s2 = k_tab[tid+2];
  tmp_v.s3 = k_tab[tid+3];
  tmp_v.s4 = k_tab[tid+4];
  tmp_v.s5 = k_tab[tid+5];
  tmp_v.s6 = k_tab[tid+6];
  tmp_v.s7 = k_tab[tid+7];
#elif (VECTOR_SIZE == 16)
  tmp_v.s0 = k_tab[tid];
  tmp_v.s1 = k_tab[tid+1];
  tmp_v.s2 = k_tab[tid+2];
  tmp_v.s3 = k_tab[tid+3];
  tmp_v.s4 = k_tab[tid+4];
  tmp_v.s5 = k_tab[tid+5];
  tmp_v.s6 = k_tab[tid+6];
  tmp_v.s7 = k_tab[tid+7];
  tmp_v.s8 = k_tab[tid+8];
  tmp_v.s9 = k_tab[tid+9];
  tmp_v.sa = k_tab[tid+10];
  tmp_v.sb = k_tab[tid+11];
  tmp_v.sc = k_tab[tid+12];
  tmp_v.sd = k_tab[tid+13];
  tmp_v.se = k_tab[tid+14];
  tmp_v.sf = k_tab[tid+15];
#endif
//MAD only available for float
  k.d0 = mad24(tmp_v, 4620u, k_base.d0);
  k.d1 = mul_hi(tmp_v, 4620u) + k_base.d1 + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	// k is limited to 2^64 -1 so there is no need for k.d2

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_77: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x\n",
        tid, tmp_v.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif

  f.d0  = k.d0 * exp96.d0 + 1;

  tmp_v   = exp96.d1 ? k.d0 : 0;  // exp96.d1 is 0 or 1
  f.d2  = exp96.d1 ? k.d1 : 0;

  f.d1  = mad_hi(k.d0, exp96.d0, tmp_v);
  f.d2 += AS_UINT_V((tmp_v > f.d1)? 1 : 0);

  tmp_v   = k.d1 * exp96.d0;
  f.d1 += tmp_v;

  f.d2 += mad_hi(k.d1, exp96.d0, AS_UINT_V((tmp_v > f.d1)? 1 : 0)); 	// f = 2 * k * exp + 1
  */
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("cl_barrett32_77: k_tab[%d]=%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        tid, tmp_v.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffb) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.

  tmp192.d4 = 0xFFFFFFFF;						// tmp192 is nearly 2^(81)
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
    if (tid==TRACE_TID) printf("cl_barrett32_77: u=%x:%x:%x, ff=%G\n",
        u.d2.s0, u.d1.s0, u.d0.s0, ff.s0);
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_77: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("cl_barrett32_77: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  a.d1 = -tmp96.d1 - AS_UINT_V((tmp96.d0 > 0) ? 1 : 0 );
  a.d2 = bb.d2-tmp96.d2 - AS_UINT_V(((tmp96.d1 | tmp96.d0) > 0) ? 1 : 0 );	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("cl_barrett32_77: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
 
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2
    if(exp&0x80000000)shl_192(&b);	// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        exp, a.d2.s0, a.d1.s0, a.d0.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp192.d5.s0, tmp192.d4.s0, tmp192.d3.s0);
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        a.d2.s0, a.d1.s0, a.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    carry= AS_UINT_V((tmp96.d0 > b.d0) ? 1 : 0);
    a.d0 = b.d0 - tmp96.d0;

    tmp_v  = b.d1 - tmp96.d1 - carry;
    carry= AS_UINT_V(((tmp_v > b.d1) || (carry && AS_UINT_V(tmp_v == b.d1))) ? 1 : 0);
    a.d1 = tmp_v;

    a.d2 = b.d2 - tmp96.d2 - carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf("loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        b.d2.s0, b.d1.s0, b.d0.s0, tmp96.d2.s0, tmp96.d1.s0, tmp96.d0.s0);
#endif

    exp+=exp;
  }

  tmp96.d0 = a.d0;
  tmp96.d1 = a.d1;
  tmp96.d2 = a.d2;

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

  a = sub_if_gte_96(a,f);	// final adjustment in case a >= f
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("cl_barrett32_77: f=%x:%x:%x, final a = %x:%x:%x \n",
         f.d2.s0, f.d1.s0, f.d0.s0, a.d2.s0, a.d1.s0, a.d0.s0 );
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_barrett32_77: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2.s0, f.d1.s0, f.d0.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
}
#endif