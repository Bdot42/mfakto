/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2014  Oliver Weihe (o.weihe@t-online.de)
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

/****************************************
 ****************************************
 * 32-bit-stuff for the 92/76-bit-barrett-kernel
 * included by main kernel file
 ****************************************
 ****************************************/

void div_192_96(int96_v * const res, __private uint qd5, const int96_v n, const float_v nf   MODBASECASE_PAR_DEF);
void div_160_96(int96_v * const res, __private uint qd5, const int96_v n, const float_v nf   MODBASECASE_PAR_DEF);
void mul_96(int96_v * const res, const int96_v a, const int96_v b);
void mul_96_192_no_low2(int192_v *const res, const int96_v a, const int96_v b);
void mul_96_192_no_low3(int192_v *const res, const int96_v a, const int96_v b);
void square_96_192(int192_v * const res, const int96_v a);
void square_96_160(int192_v * const res, const int96_v a);
void shl_96(int96_v * const a);
void shl_192(int192_v * const a);
void check_barrett32_76(uint shifter, const int96_v f, const uint tid, const int192_t bb, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);
void check_barrett32_77(uint shifter, const int96_v f, const uint tid, const int192_t bb, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);
void check_barrett32_78(uint shifter, const int96_v f, const uint tid, const int192_t bb, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);
void check_barrett32_79(uint shifter, const int96_v f, const uint tid, const int192_t bb, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);
void check_barrett32_87(uint shifter, const int96_v f, const uint tid, const int192_t bb, const uint bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);
void check_barrett32_88(uint shifter, const int96_v f, const uint tid, const int192_t bb, const uint bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);
void check_barrett32_89(uint shifter, const int96_v f, const uint tid, const int192_t bb, const uint bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);
void check_barrett32_92(uint shifter, const int96_v f, const uint tid, const int192_t bb, const uint bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);

/****************************************
 ****************************************
 * 32-bit based 79- and 92-bit barrett-kernels
 *
 ****************************************
 ****************************************/

void mul_96(int96_v * const res, const int96_v a, const int96_v b)
/* res = a * b */
{
  __private uint_v tmp;

  res->d0  = a.d0 * b.d0;
  res->d1  = mul_hi(a.d0, b.d0);

  res->d2  = mul_hi(a.d1, b.d0);

  tmp = a.d1 * b.d0;
  res->d1 += tmp;
  res->d2 -= AS_UINT_V(tmp > res->d1);

  res->d2 += mul_hi(a.d0, b.d1);

  tmp = a.d0 * b.d1;
  res->d1 += tmp;
  res->d2 -= AS_UINT_V(tmp > res->d1);

  res->d2 += a.d0 * b.d2 + a.d1 * b.d1 + a.d2 * b.d0;
}


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
  res->d4  = AS_UINT_V(tmp > res->d3);

  tmp      = mul_hi(a.d0, b.d2);
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);

  tmp      = a.d2 * b.d1;
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);

  tmp      = a.d1 * b.d2;
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);


  tmp      = mul_hi(a.d2, b.d1);
  res->d4  = tmp - res->d4;
  res->d5  = AS_UINT_V(tmp > res->d4);

  tmp      = mul_hi(a.d1, b.d2);
  res->d4 += tmp;
  res->d5 += AS_UINT_V(tmp > res->d4);

  tmp      = a.d2 * b.d2;
  res->d4 += tmp;
  res->d5 += AS_UINT_V(tmp > res->d4);

  res->d5  = mul_hi(a.d2, b.d2) - res->d5;
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
  res->d2  = AS_UINT_V(tmp > res->d1);
  res->d1 += tmp;
  res->d2 += AS_UINT_V(tmp > res->d1);


  res->d2  = a.d1 * a.d1 - res->d2;  // no carry possible

  tmp      = mul_hi(a.d0, a.d1);
  res->d2 += tmp;
  res->d3  = AS_UINT_V(tmp > res->d2);
  res->d2 += tmp;
  res->d3 += AS_UINT_V(tmp > res->d2);

  tmp      = a.d0 * a.d2;
  res->d2 += tmp;
  res->d3 += AS_UINT_V(tmp > res->d2);
  res->d2 += tmp;
  res->d3 += AS_UINT_V(tmp > res->d2);


  tmp      = mul_hi(a.d1, a.d1);
  res->d3  = tmp - res->d3;
  res->d4  = AS_UINT_V(tmp > res->d3);

  tmp      = mul_hi(a.d0, a.d2);
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);

  tmp      = a.d1 * a.d2;
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);


  res->d4  = a.d2 * a.d2 - res->d4; // no carry possible

  tmp      = mul_hi(a.d1, a.d2);
  res->d4 += tmp;
  res->d5  = AS_UINT_V(tmp > res->d4);
  res->d4 += tmp;
  res->d5 += AS_UINT_V(tmp > res->d4);


  res->d5 = mul_hi(a.d2, a.d2) - res->d5;
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
  res->d2  = AS_UINT_V(tmp > res->d1);
  res->d1 += tmp;
  res->d2 += AS_UINT_V(tmp > res->d1);


  res->d2  = a.d1 * a.d1 - res->d2;  // no carry possible

  tmp      = mul_hi(a.d0, a.d1);
  res->d2 += tmp;
  res->d3  = AS_UINT_V(tmp > res->d2);
  res->d2 += tmp;
  res->d3 += AS_UINT_V(tmp > res->d2);

  tmp      = a.d0 * TWOad2;
  res->d2 += tmp;
  res->d3 += AS_UINT_V(tmp > res->d2);


  tmp      = mul_hi(a.d1, a.d1);
  res->d3  = tmp - res->d3;
  res->d4  = AS_UINT_V(tmp > res->d3);

  tmp      = mul_hi(a.d0, TWOad2);
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);

  tmp      = a.d1 * TWOad2;
  res->d3 += tmp;
  res->d4 += AS_UINT_V(tmp > res->d3);


  res->d4  = a.d2 * a.d2 - res->d4; // no carry possible

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

#if defined USE_DP
void div_192_96_d(int96_v * const res, __private uint qd5, const int96_v n, const double_v nf   MODBASECASE_PAR_DEF)
/* res = q / n (integer division) */
{
  __private double_v qf;
  __private double qf_1; // the first conversion does not need a vector yet
  __private ulong_v qi;
  __private uint_v qi_l, qi_h, tmp;
  __private int192_v nn, q;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf_1 = CONVERT_DOUBLE(qd5);
  qf_1 = qf_1 * 4294967296.0 * 4294967296.0;
#else
  // div_160_96 will always be called with qd5 = 1 ==> q = 2^160 
  qf_1 = 4294967296.0 * 4294967296.0;
#endif
  qf_1 = qf_1 * 4294967296.0 * 1048576.0;

  qi = CONVERT_ULONG_V(qf_1*nf);

  MODBASECASE_QI_ERROR(1UL<<52, 1, qi, 0);

  res->d2 = qi_h = CONVERT_UINT_V(qi >> 20);
  res->d1 = qi_l = CONVERT_UINT_V(qi << 12);
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1.1: q=%x:0:0:0:0:0, n=%x:%x:%x, qi=%llx, nf=%G\n",
        qd5, V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

// nn = n * qi
// 3 x 2 components => 6x mul + 6x mul_hi
  nn.d0  = n.d0 * qi_l; // 1

  nn.d1  = mul_hi(n.d0, qi_l); // 1
  tmp    = n.d1 * qi_l; // 2
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = n.d0 * qi_h; // 3
  nn.d1 += tmp;
  nn.d2 += AS_UINT_V(tmp > nn.d1);

  tmp    = mul_hi(n.d1, qi_l); // 2
  nn.d2  = tmp - nn.d2; // carries are negative in OpenCL vectors; in fact this is an addition
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2 * qi_l; // 4
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  tmp    = mul_hi(n.d0, qi_h); // 3
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  tmp    = n.d1 * qi_h; // 5
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);

  tmp    = mul_hi(n.d2, qi_l); // 4
  nn.d3  = tmp - nn.d3; // add negative carries
  nn.d4  = AS_UINT_V(tmp > nn.d3);
  tmp    = mul_hi(n.d1, qi_h); // 5
  nn.d3 += tmp;
  nn.d4 += AS_UINT_V(tmp > nn.d3);
  tmp    = n.d2 * qi_h; // 6
  nn.d3 += tmp;
  nn.d4 += AS_UINT_V(tmp > nn.d3);

  nn.d4  = mul_hi(n.d2, qi_h) - nn.d4; // 6

#if (TRACE_KERNEL > 2)
  if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1.2: nn=%x:%x:%x:%x:%x, qi=%x:%x\n",
        V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(qi_h), V(qi_l));
#endif

// q = q - nn; q.d0..q.d4 are all zero, q.d5 was passed in as qd5

  q.d1 = -nn.d0;
  q.d2 = AS_UINT_V(nn.d0 > 0) - nn.d1;
  q.d3 = AS_UINT_V((nn.d0 > 0) || (nn.d1 > 0)) - nn.d2;
#if !defined DIV_160_96 || defined CHECKS_MODBASECASE || TRACE_KERNEL > 2
  q.d4 = AS_UINT_V((nn.d0 > 0) || (nn.d1 > 0) || (nn.d2 > 0)) - nn.d3;
#endif
#if defined CHECKS_MODBASECASE || TRACE_KERNEL > 2
  q.d5 = qd5 + 0xFFFFFFFF - nn.d4; // assume we have a carry from one of the prev 128 bits
#endif

#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1.3: q=%x:%x:%x:%x:%x:0\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1));
#endif

  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);
#ifdef DIV_160_96
  MODBASECASE_NONZERO_ERROR(q.d4, 3, 5, 2);

  qf = CONVERT_DOUBLE_V(q.d3);
#else
  qf = CONVERT_DOUBLE_V(q.d4) * 4294967296.0 + CONVERT_DOUBLE_V(q.d3);
#endif
  qf = qf * 4294967296.0 + CONVERT_DOUBLE_V(q.d2);
  qf = qf * 4294967296.0; //  + CONVERT_DOUBLE_V(q.d1)PERF: q.d1 needed?
  qf = qf * 4294967296.0; // q.d0 not needed

  qi = CONVERT_ULONG_V(qf*nf);

  MODBASECASE_QI_ERROR(1UL<<46, 1, qi, 0);

  qi_h = CONVERT_UINT_V(qi >> 32);
  res->d1 += qi_h;
  res->d0 = qi_l = CONVERT_UINT_V(qi);
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div2.1: qi=%llx=%x:%x,  res=%x:%x:%x\n",
        V(qi), V(qi_h), V(qi_l), V(res->d2), V(res->d1), V(res->d0));
#endif

}
#else
void div_192_96(int96_v * const res, __private uint qd5, const int96_v n, const float_v nf   MODBASECASE_PAR_DEF)
/* res = q / n (integer division) */
{
  __private float_v qf;
  __private float qf_1; // to do the first few calculations once per vector, as long as all vector elements would do the same.
  __private uint_v qi, tmp, carry;
  __private int192_v nn, q = {0,0,0,0,0,qd5};

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf_1 = CONVERT_FLOAT(qd5);
  qf_1 = qf_1 * 9007199254740992.0f;
#else
  // qf_1 = 1.0f; // DIV_160_96 will always be called with qd5=1
  qf_1 = 9007199254740992.0f;
#endif

  qi=CONVERT_UINT_V(qf_1*nf); // the multiplication implicitly promotes qf_1 to a vector of the right size

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 11;
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1.1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, nf=%G\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1 * qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = mul_hi(n.d1, qi);
  nn.d2  = tmp - nn.d2;
#ifndef DIV_160_96
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2 * qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  nn.d3  = mul_hi(n.d2, qi) - nn.d3;
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
  carry= AS_UINT_V(nn.d0 > q.d2);
  q.d2 = q.d2 - nn.d0;

  tmp  = q.d3 - nn.d1 + carry ;
  carry= AS_UINT_V((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)));
  q.d3 = tmp;

#ifndef DIV_160_96
  tmp  = q.d4 - nn.d2 + carry;
  carry= AS_UINT_V((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)));
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d3 + carry;
#else
  q.d4 = q.d4 - nn.d2 + carry;
#endif
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1: q=%x:%x:%x:%x, nn=%x:%x:%x:%x, res=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d2));
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
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div2.1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, nf=%G\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

// nn = n * qi
  nn.d0 = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = mul_hi(n.d1, qi);
  nn.d2  = tmp - nn.d2;
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  nn.d3  = mul_hi(n.d2, qi) - nn.d3;

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
  carry= AS_UINT_V(nn.d0 > q.d1);
  q.d1 = q.d1 - nn.d0;

  tmp  = q.d2 - nn.d1 + carry;
  carry= AS_UINT_V((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)));
  q.d2 = tmp;

  tmp  = q.d3 - nn.d2 + carry;
  carry= AS_UINT_V((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)));
  q.d3 = tmp;
#if (TRACE_KERNEL > 4)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div2.6: q=%x:%x:%x:%x:%x, nn=%x:%x:%x:%x, tmp=%x, c=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(tmp), V(carry));
#endif

#ifdef CHECKS_MODBASECASE
  tmp  = q.d4 - nn.d3 + carry;
  carry= AS_UINT_V((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)));
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d4 + carry;
#else
  q.d4 = q.d4 - nn.d3 + carry;
#endif
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div2: q=%x:%x:%x:%x:%x(c=%x), nn=%x:%x:%x:%x, res=%x:%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(carry), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d2), V(res->d1));
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div3.1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, nf=%G\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

  tmp     = (qi << 3);
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + (qi >> 29) - AS_UINT_V(tmp > res->d1);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;


// nn = n * qi

  nn.d0 = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = mul_hi(n.d1, qi);
  nn.d2  = tmp - nn.d2;
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  nn.d3  = mul_hi(n.d2, qi) - nn.d3;

//  q = q - nn
  carry= AS_UINT_V(nn.d0 > q.d1);
  q.d1 = q.d1 - nn.d0;

  tmp  = q.d2 - nn.d1 + carry;
  carry= AS_UINT_V((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)));
  q.d2 = tmp;

  tmp  = q.d3 - nn.d2 + carry;
  carry= AS_UINT_V((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)));
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d3 + carry;
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div3: q=%x:%x:%x:%x:%x:(c=%x), nn=%x:%x:%x:%x, res=%x:%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(carry), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d2), V(res->d1));
#endif

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d2);
  qf*= 131072.0f;

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  tmp     = qi >> 17;
  res->d0 = qi << 15;
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 - AS_UINT_V(tmp > res->d1);

#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div4.1: q=%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, nf=%G\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = mul_hi(n.d1, qi);
  nn.d2  = tmp - nn.d2;
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  nn.d3  = mul_hi(n.d2, qi) - nn.d3;

// shiftleft nn 15 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                  nn.d3 >> 17;
#endif
  nn.d3 = (nn.d3 << 15) + (nn.d2 >> 17);
  nn.d2 = (nn.d2 << 15) + (nn.d1 >> 17);
  nn.d1 = (nn.d1 << 15) + (nn.d0 >> 17);
  nn.d0 =  nn.d0 << 15;

//  q = q - nn
  carry= AS_UINT_V(nn.d0 > q.d0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 + carry;
  carry= AS_UINT_V((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1)));
  q.d1 = tmp;

  tmp  = q.d2 - nn.d2 + carry;
  carry= AS_UINT_V((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)));
  q.d2 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d3 = q.d3 - nn.d3 + carry;
#else
  tmp  = q.d3 - nn.d3 + carry;
  carry= AS_UINT_V((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)));
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 + carry;
#endif
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div4: q=%x:%x:%x:%x:%x:%x(c=%x), nn=%x:%x:%x:%x, res=%x:%x:%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(carry), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d2), V(res->d1), V(res->d0));
#endif


/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= CONVERT_FLOAT_V(q.d3);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d2);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d1);

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 += qi;
  carry    = AS_UINT_V(qi > res->d0);
  tmp      = res->d1 - carry;
  res->d2 -= AS_UINT_V(tmp < res->d1);
  res->d1  = tmp;

#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div5: qi=%x, nf=%G, res=%x:%x:%x\n",
        V(qi), V(nf), V(res->d2), V(res->d1), V(res->d0));
#endif


  return;

// not finishing the final multiplication/subtraction/comparison leaves the result off by 1 at most.
}
#endif


#define DIV_160_96
#if defined USE_DP
void div_160_96_d(int96_v * const res, __private uint qd5, const int96_v n, const double_v nf   MODBASECASE_PAR_DEF)
/* res = q / n (integer division) */
/* the code of div_160_96() is an EXACT COPY of div_192_96(), the only
difference is that the 160bit version ignores the most significant
word of q (q.d5) because it assumes it is 0. This is controlled by defining
DIV_160_96 here. */
{
  __private double_v qf;
  __private double qf_1; // the first conversion does not need a vector yet
  __private ulong_v qi;
  __private uint_v qi_l, qi_h, tmp;
  __private int192_v nn, q;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf_1 = CONVERT_DOUBLE(qd5);
  qf_1 = qf_1 * 4294967296.0 * 4294967296.0;
#else
  // div_160_96 will always be called with qd5 = 1 ==> q = 2^160 
  qf_1 = 4294967296.0 * 4294967296.0;
#endif
  qf_1 = qf_1 * 4294967296.0 * 1048576.0;

  qi = CONVERT_ULONG_V(qf_1*nf);

  MODBASECASE_QI_ERROR(1UL<<52, 1, qi, 0);

  res->d2 = qi_h = CONVERT_UINT_V(qi >> 20);
  res->d1 = qi_l = CONVERT_UINT_V(qi << 12);
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1.1: q=%x:0:0:0:0:0, n=%x:%x:%x, qi=%llx, nf=%G\n",
        qd5, V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

// nn = n * qi
// 3 x 2 components => 6x mul + 6x mul_hi
  nn.d0  = n.d0 * qi_l; // 1

  nn.d1  = mul_hi(n.d0, qi_l); // 1
  tmp    = n.d1 * qi_l; // 2
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = n.d0 * qi_h; // 3
  nn.d1 += tmp;
  nn.d2 += AS_UINT_V(tmp > nn.d1);

  tmp    = mul_hi(n.d1, qi_l); // 2
  nn.d2  = tmp - nn.d2; // carries are negative in OpenCL vectors; in fact this is an addition
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2 * qi_l; // 4
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  tmp    = mul_hi(n.d0, qi_h); // 3
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  tmp    = n.d1 * qi_h; // 5
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);

  tmp    = mul_hi(n.d2, qi_l); // 4
  nn.d3  = tmp - nn.d3; // add negative carries
  nn.d4  = AS_UINT_V(tmp > nn.d3);
  tmp    = mul_hi(n.d1, qi_h); // 5
  nn.d3 += tmp;
  nn.d4 += AS_UINT_V(tmp > nn.d3);
  tmp    = n.d2 * qi_h; // 6
  nn.d3 += tmp;
  nn.d4 += AS_UINT_V(tmp > nn.d3);

  nn.d4  = mul_hi(n.d2, qi_h) - nn.d4; // 6

#if (TRACE_KERNEL > 2)
  if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1.2: nn=%x:%x:%x:%x:%x, qi=%x:%x\n",
        V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(qi_h), V(qi_l));
#endif

// q = q - nn; q.d0..q.d4 are all zero, q.d5 was passed in as qd5

  q.d1 = -nn.d0;
  q.d2 = AS_UINT_V(nn.d0 > 0) - nn.d1;
  q.d3 = AS_UINT_V((nn.d0 > 0) || (nn.d1 > 0)) - nn.d2;
#if !defined DIV_160_96 || defined CHECKS_MODBASECASE || TRACE_KERNEL > 2
  q.d4 = AS_UINT_V((nn.d0 > 0) || (nn.d1 > 0) || (nn.d2 > 0)) - nn.d3;
#endif
#if defined CHECKS_MODBASECASE || TRACE_KERNEL > 2
  q.d5 = qd5 + 0xFFFFFFFF - nn.d4; // assume we have a carry from one of the prev 128 bits
#endif

#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1.3: q=%x:%x:%x:%x:%x:0\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1));
#endif

  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);
#ifdef DIV_160_96
  MODBASECASE_NONZERO_ERROR(q.d4, 3, 5, 2);

  qf = CONVERT_DOUBLE_V(q.d3);
#else
  qf = CONVERT_DOUBLE_V(q.d4) * 4294967296.0 + CONVERT_DOUBLE_V(q.d3);
#endif
  qf = qf * 4294967296.0 + CONVERT_DOUBLE_V(q.d2);
  qf = qf * 4294967296.0; //  + CONVERT_DOUBLE_V(q.d1)PERF: q.d1 needed?
  qf = qf * 4294967296.0; // q.d0 not needed

  qi = CONVERT_ULONG_V(qf*nf);

  MODBASECASE_QI_ERROR(1UL<<46, 1, qi, 0);

  qi_h = CONVERT_UINT_V(qi >> 32);
  res->d1 += qi_h;
  res->d0 = qi_l = CONVERT_UINT_V(qi);
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div2.1: qi=%llx=%x:%x,  res=%x:%x:%x\n",
        V(qi), V(qi_h), V(qi_l), V(res->d2), V(res->d1), V(res->d0));
#endif

}
#endif

void div_160_96(int96_v * const res, __private uint qd5, const int96_v n, const float_v nf   MODBASECASE_PAR_DEF)
/* res = q / n (integer division) */
/* the code of div_160_96() is an EXACT COPY of div_192_96(), the only
difference is that the 160bit version ignores the most significant
word of q (q.d5) because it assumes it is 0. This is controlled by defining
DIV_160_96 here. */
{
  __private float_v qf;
  __private float qf_1; // to do the first few calculations once per vector, as long as all vector elements would do the same.
  __private uint_v qi, tmp, carry;
  __private int192_v nn, q = {0,0,0,0,0,qd5};

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf_1 = CONVERT_FLOAT(qd5);
  qf_1 = qf_1 * 9007199254740992.0f;
#else
  // qf_1 = 1.0f; // DIV_160_96 will always be called with qd5=1
  qf_1 = 9007199254740992.0f;
#endif

  qi=CONVERT_UINT_V(qf_1*nf); // the multiplication implicitly promotes qf_1 to a vector of the right size

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

  res->d2 = qi << 11;
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1.1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, nf=%G\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1 * qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = mul_hi(n.d1, qi);
  nn.d2  = tmp - nn.d2;
#ifndef DIV_160_96
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2 * qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  nn.d3  = mul_hi(n.d2, qi) - nn.d3;
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
  carry= AS_UINT_V(nn.d0 > q.d2);
  q.d2 = q.d2 - nn.d0;

  tmp  = q.d3 - nn.d1 + carry ;
  carry= AS_UINT_V((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)));
  q.d3 = tmp;

#ifndef DIV_160_96
  tmp  = q.d4 - nn.d2 + carry;
  carry= AS_UINT_V((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)));
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d3 + carry;
#else
  q.d4 = q.d4 - nn.d2 + carry;
#endif
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div1: q=%x:%x:%x:%x, nn=%x:%x:%x:%x, res=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d2));
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
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div2.1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, nf=%G\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

// nn = n * qi
  nn.d0 = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = mul_hi(n.d1, qi);
  nn.d2  = tmp - nn.d2;
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  nn.d3  = mul_hi(n.d2, qi) - nn.d3;

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
  carry= AS_UINT_V(nn.d0 > q.d1);
  q.d1 = q.d1 - nn.d0;

  tmp  = q.d2 - nn.d1 + carry;
  carry= AS_UINT_V((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)));
  q.d2 = tmp;

  tmp  = q.d3 - nn.d2 + carry;
  carry= AS_UINT_V((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)));
  q.d3 = tmp;
#if (TRACE_KERNEL > 4)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div2.6: q=%x:%x:%x:%x:%x, nn=%x:%x:%x:%x, tmp=%x, c=%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(tmp), V(carry));
#endif

#ifdef CHECKS_MODBASECASE
  tmp  = q.d4 - nn.d3 + carry;
  carry= AS_UINT_V((tmp > q.d4) || (carry && AS_UINT_V(tmp == q.d4)));
  q.d4 = tmp;
  q.d5 = q.d5 - nn.d4 + carry;
#else
  q.d4 = q.d4 - nn.d3 + carry;
#endif
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div2: q=%x:%x:%x:%x:%x(c=%x), nn=%x:%x:%x:%x, res=%x:%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(carry), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d2), V(res->d1));
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div3.1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, nf=%G\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

  tmp     = (qi << 3);
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 + (qi >> 29) - AS_UINT_V(tmp > res->d1);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;


// nn = n * qi

  nn.d0 = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = mul_hi(n.d1, qi);
  nn.d2  = tmp - nn.d2;
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  nn.d3  = mul_hi(n.d2, qi) - nn.d3;

//  q = q - nn
  carry= AS_UINT_V(nn.d0 > q.d1);
  q.d1 = q.d1 - nn.d0;

  tmp  = q.d2 - nn.d1 + carry;
  carry= AS_UINT_V((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)));
  q.d2 = tmp;

  tmp  = q.d3 - nn.d2 + carry;
  carry= AS_UINT_V((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)));
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d3 + carry;
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div3: q=%x:%x:%x:%x:%x:(c=%x), nn=%x:%x:%x:%x, res=%x:%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(carry), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d2), V(res->d1));
#endif

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  qf= CONVERT_FLOAT_V(q.d4);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d3);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d2);
  qf*= 131072.0f;

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

  tmp     = qi >> 17;
  res->d0 = qi << 15;
  res->d1 = res->d1 + tmp;
  res->d2 = res->d2 - AS_UINT_V(tmp > res->d1);

#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div4.1: q=%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x, nf=%G\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(n.d2), V(n.d1), V(n.d0), V(qi), V(nf));
#endif

// nn = n * qi
  nn.d0  = n.d0 * qi;
  nn.d1  = mul_hi(n.d0, qi);
  tmp    = n.d1* qi;
  nn.d1 += tmp;
  nn.d2  = AS_UINT_V(tmp > nn.d1);
  tmp    = mul_hi(n.d1, qi);
  nn.d2  = tmp - nn.d2;
  nn.d3  = AS_UINT_V(tmp > nn.d2);
  tmp    = n.d2* qi;
  nn.d2 += tmp;
  nn.d3 += AS_UINT_V(tmp > nn.d2);
  nn.d3  = mul_hi(n.d2, qi) - nn.d3;

// shiftleft nn 15 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                  nn.d3 >> 17;
#endif
  nn.d3 = (nn.d3 << 15) + (nn.d2 >> 17);
  nn.d2 = (nn.d2 << 15) + (nn.d1 >> 17);
  nn.d1 = (nn.d1 << 15) + (nn.d0 >> 17);
  nn.d0 =  nn.d0 << 15;

//  q = q - nn
  carry= AS_UINT_V(nn.d0 > q.d0);
  q.d0 = q.d0 - nn.d0;

  tmp  = q.d1 - nn.d1 + carry;
  carry= AS_UINT_V((tmp > q.d1) || (carry && AS_UINT_V(tmp == q.d1)));
  q.d1 = tmp;

  tmp  = q.d2 - nn.d2 + carry;
  carry= AS_UINT_V((tmp > q.d2) || (carry && AS_UINT_V(tmp == q.d2)));
  q.d2 = tmp;

#ifndef CHECKS_MODBASECASE
  q.d3 = q.d3 - nn.d3 + carry;
#else
  tmp  = q.d3 - nn.d3 + carry;
  carry= AS_UINT_V((tmp > q.d3) || (carry && AS_UINT_V(tmp == q.d3)));
  q.d3 = tmp;

  q.d4 = q.d4 - nn.d4 + carry;
#endif
#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div4: q=%x:%x:%x:%x:%x:%x(c=%x), nn=%x:%x:%x:%x, res=%x:%x:%x\n",
        V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1), V(q.d0), V(carry), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0), V(res->d2), V(res->d1), V(res->d0));
#endif


/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= CONVERT_FLOAT_V(q.d3);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d2);
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(q.d1);

  qi= CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 += qi;
  carry    = AS_UINT_V(qi > res->d0);
  tmp      = res->d1 - carry;
  res->d2 -= AS_UINT_V(tmp < res->d1);
  res->d1  = tmp;

#if (TRACE_KERNEL > 2)
    if (get_global_id(0)==TRACE_TID) printf((__constant char *)"div5: qi=%x, nf=%G, res=%x:%x:%x\n",
        V(qi), V(nf), V(res->d2), V(res->d1), V(res->d0));
#endif


  return;

// not finishing the final multiplication/subtraction/comparison leaves the result off by 1 at most.
}

#undef DIV_160_96


/*
 * TF 64-76 bits using 32-bit barrett:
 * square - reduce - shift
 */
void check_barrett32_76(uint shifter, const int96_v f, const uint tid, const int192_t bb, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int96_v  a, u, tmp96;
  __private int192_v b, tmp192;
#if defined USE_DP
  __private double_v  ffd;
#endif
  __private float_v  ff;
  __private uint_v   carry;

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffc) / ff;		// just a little bit below 1.0f so we allways underestimate the quotient

#if defined USE_DP
  // use double when available, but not for the low and mid level GCN cards as they have 1:16 DP rate which is too slow
  ffd= CONVERT_DOUBLE_RTP_V(f.d2);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d1);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d0);

  ffd= as_double(0x3feffffffffffffdL) / ffd;		// just a little bit below 1.0f so we allways underestimate the quotient

#ifndef CHECKS_MODBASECASE
  div_160_96_d(&u, 1, f, ffd);						// u = floor(2^(80*2) / f)
#else
  div_160_96_d(&u, 1, f, ffd, modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett76: u=%x:%x:%x, ffd=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else // no doubles or GCN (low/mid)
#ifndef CHECKS_MODBASECASE
  div_160_96(&u,1,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_160_96(&u,1,f,ff,modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett76: u=%x:%x:%x, ff=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett76: a=%x:%x:%x * u = %x:%x:%x:...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3));
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett76: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif

    // bb.d0-bb.d1 are all zero due to preprocessing on the host
    // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
    a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    a.d1 = -tmp96.d1 + AS_UINT_V(tmp96.d0 > 0);
    a.d2 = bb.d2-tmp96.d2 + AS_UINT_V((tmp96.d1 | tmp96.d0) > 0);	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett76: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        bb.d2, bb.d1, bb.d0, V(a.d2), V(a.d1), V(a.d0));
#endif

  while(shifter)
  {
    square_96_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d2), V(a.d1), V(a.d0), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;

    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3));
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif

    carry= AS_UINT_V(tmp96.d0 > b.d0);
    a.d0 = b.d0 - tmp96.d0;

    a.d1  = b.d1 - tmp96.d1 + carry;
    carry= AS_UINT_V((a.d1 > b.d1) || (carry && AS_UINT_V(a.d1 == b.d1)));

    a.d2 = b.d2 - tmp96.d2 + carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        V(b.d2), V(b.d1), V(b.d0), V(a.d2), V(a.d1), V(a.d0));
#endif

    if(shifter&0x80000000)shl_96(&a);					// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        V(tmp96.d2), V(tmp96.d1), V(tmp96.d0), V(f.d2), V(f.d1), V(f.d0), V(a.d2), V(a.d1), V(a.d0) );
#endif

    shifter+=shifter;
  }
#if (TRACE_KERNEL > 0)
#if (VECTOR_SIZE > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: f=%x:%x:%x, a=%x:%x:%x\n",
        V(f.d2), V(f.d1), V(f.d0), V(a.d2), V(a.d1), V(a.d0) );
#else
    if (tid==TRACE_TID) printf((__constant char *)"loopend: f=%x:%x:%x, a=%x:%x:%x\n",
        f.d2, f.d1, f.d0, a.d2, a.d1, a.d0 );
#endif
#endif
  mod_simple_even_96_and_check_big_factor96(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , 15, 10 , modbasecase_debug
#endif
                       );
}

/*
 * TF 64-77 bits using 32-bit barrett:
 * square - shift - reduce
 */
void check_barrett32_77(uint shifter, const int96_v f, const uint tid, const int192_t bb, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int96_v  a, u, tmp96;
  __private int192_v b, tmp192;
#if defined USE_DP
  __private double_v  ffd;
#endif
  __private float_v  ff;
  __private uint_v   carry;

/*
ff = 1/f as float, needed in div_160_96().
Precalculated here since it is the same for all steps in the following loop */
    ff= CONVERT_FLOAT_RTP_V(f.d2);
    ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

    ff= as_float(0x3f7ffffc) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.

#if (TRACE_KERNEL > 2)
      if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_77: f=%x:%x:%x, ff=%G\n",
          V(f.d2), V(f.d1), V(f.d0), V(ff));
#endif

#if defined USE_DP
  // use double when available, but not for the low and mid level GCN cards as they have 1:16 DP rate which is too slow
  ffd= CONVERT_DOUBLE_RTP_V(f.d2);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d1);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d0);

  ffd= as_double(0x3feffffffffffffdL) / ffd;		// just a little bit below 1.0f so we allways underestimate the quotient

#ifndef CHECKS_MODBASECASE
  div_160_96_d(&u, 1, f, ffd);						// u = floor(2^(80*2) / f)
#else
  div_160_96_d(&u, 1, f, ffd, modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett77: u=%x:%x:%x, ffd=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else // no doubles or GCN (low/mid)
#ifndef CHECKS_MODBASECASE
  div_160_96(&u,1,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_160_96(&u,1,f,ff,modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett77: u=%x:%x:%x, ff=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  // bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
    a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = bb.d3;
    a.d2 = bb.d4;

    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
      if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_77: a=%x:%x:%x * u = %x:%x:%x:...\n",
          V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3));
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
      if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_77: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
          V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
    // bb.d0-bb.d1 are all zero due to preprocessing on the host
    // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
    a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    a.d1 = -tmp96.d1 + AS_UINT_V(tmp96.d0 > 0);
    a.d2 = bb.d2-tmp96.d2 + AS_UINT_V((tmp96.d1 | tmp96.d0) > 0);	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_77: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
          bb.d2, bb.d1, bb.d0, V(a.d2), V(a.d1), V(a.d0));
#endif

    while(shifter)
    {
      square_96_160(&b, a);						// b = a^2
      if(shifter&0x80000000)shl_192(&b);	// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 2)
      if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x (b)\n",
          shifter, V(a.d2), V(a.d1), V(a.d0), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif

      a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
      a.d1 = b.d3;
      a.d2 = b.d4;

      mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
      if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
          V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3));
#endif

      a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
      a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
      a.d2 = tmp192.d5;

      mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
      if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
          V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif

      carry= AS_UINT_V(tmp96.d0 > b.d0);
      a.d0 = b.d0 - tmp96.d0;

      a.d1  = b.d1 - tmp96.d1 + carry;
      carry= AS_UINT_V((a.d1 > b.d1) || (carry && AS_UINT_V(a.d1 == b.d1)));

      a.d2 = b.d2 - tmp96.d2 + carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
      if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
          V(b.d2), V(b.d1), V(b.d0), V(a.d2), V(a.d1), V(a.d0));
#endif

      shifter+=shifter;
    }
#if (TRACE_KERNEL > 0)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: f=%x:%x:%x, a=%x:%x:%x\n",
        V(f.d2), V(f.d1), V(f.d0), V(a.d2), V(a.d1), V(a.d0) );
#endif

    mod_simple_96_and_check_big_factor96(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , 15, 10 , modbasecase_debug
#endif
                       );
}

void check_barrett32_79(uint shifter, const int96_v f, const uint tid, const int192_t bb, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int96_v  a, u, tmp96;
  __private int192_v b, tmp192;
#if defined USE_DP
  __private double_v  ffd;
#endif
  __private float_v  ff;
  __private uint_v   tmp, carry;

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffc) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.

#if defined USE_DP
  // use double when available, but not for the low and mid level GCN cards as they have 1:16 DP rate which is too slow
  ffd= CONVERT_DOUBLE_RTP_V(f.d2);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d1);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d0);

  ffd= as_double(0x3feffffffffffffdL) / ffd;		// just a little bit below 1.0f so we allways underestimate the quotient

#ifndef CHECKS_MODBASECASE
  div_160_96_d(&u, 1, f, ffd);						// u = floor(2^(80*2) / f)
#else
  div_160_96_d(&u, 1, f, ffd, modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett79: u=%x:%x:%x, ffd=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else // no doubles or GCN (low/mid)
#ifndef CHECKS_MODBASECASE
  div_160_96(&u,1,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_160_96(&u,1,f,ff,modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett79: u=%x:%x:%x, ff=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

// bb is still the preprocessed scalar passed in to the kernel - it is widened here to the required vector size automatically
  a.d0 = bb.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = bb.d3;
  a.d2 = bb.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_79: a=%x:%x:%x * u = %x:%x:%x:...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3));
#endif

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_79: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = -tmp96.d1 + AS_UINT_V(tmp96.d0 > 0);
  tmp96.d2 = bb.d2-tmp96.d2 + AS_UINT_V((tmp96.d1 | tmp96.d0) > 0);	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_79: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
  while(shifter)
  {

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 9;					// bit_max == 79, due to decreased accuracy of mul_96_192_no_low3() above we need a higher threshold
  mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , 79 - 64, limit , modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_79: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        V(tmp96.d2), V(tmp96.d1), V(tmp96.d0), V(f.d2), V(f.d1), V(f.d0), V(a.d2), V(a.d1), V(a.d0) );
#endif

    square_96_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d2), V(a.d1), V(a.d0), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;

    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * u = %x:%x:%x:...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3));
#endif

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif

    carry= AS_UINT_V(tmp96.d0 > b.d0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp  = b.d1 - tmp96.d1 + carry;
    carry= AS_UINT_V((tmp > b.d1) || (carry && AS_UINT_V(tmp == b.d1)));
    tmp96.d1 = tmp;

    tmp96.d2 = b.d2 - tmp96.d2 + carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        V(b.d2), V(b.d1), V(b.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif

    if(shifter&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"end loop: tmp=%x:%x:%x\n",
        V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif

    shifter+=shifter;
  }
#if (TRACE_KERNEL > 0)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: f=%x:%x:%x, a=%x:%x:%x\n",
        V(f.d2), V(f.d1), V(f.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0) );
#endif
  mod_simple_even_96_and_check_big_factor96(tmp96, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , 15, 10 , modbasecase_debug
#endif
                       );
}

void check_barrett32_87(uint shifter, const int96_v f, const uint tid, const int192_t bb, const uint bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int96_v  a, u, tmp96;
  __private int192_v b, tmp192;
#if defined USE_DP
  __private double_v  ffd;
#endif
  __private float_v  ff;
  __private uint_v   carry;
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */

/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffc) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.

#if defined USE_DP
  // use double when available, but not for the low and mid level GCN cards as they have 1:16 DP rate which is too slow
  ffd= CONVERT_DOUBLE_RTP_V(f.d2);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d1);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d0);

  ffd= as_double(0x3feffffffffffffdL) / ffd;		// just a little bit below 1.0f so we allways underestimate the quotient

#ifndef CHECKS_MODBASECASE
  div_192_96_d(&u, 1 << bit_max65, f, ffd);						// u = floor(2^(80*2) / f)
#else
  div_192_96_d(&u, 1 << bit_max65, f, ffd, modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 0)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett87: u=%x:%x:%x, ffd=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else // no doubles or GCN (low/mid)
#ifndef CHECKS_MODBASECASE
  div_192_96(&u,1 << bit_max65,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_192_96(&u,1 << bit_max65,f,ff,modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett87: u=%x:%x:%x, ff=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_87: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3), V(tmp192.d2));
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_87: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
    // bb.d0-bb.d1 are all zero due to preprocessing on the host
    // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
    a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    a.d1 = -tmp96.d1 + AS_UINT_V(tmp96.d0 > 0);
    a.d2 = bb.d2-tmp96.d2 + AS_UINT_V((tmp96.d1 | tmp96.d0) > 0);	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_87: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        bb.d2, bb.d1, bb.d0, V(a.d2), V(a.d1), V(a.d0));
#endif
  while(shifter)
  {                                 // On input a is at most 93 bits (see end of this loop)
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d2), V(a.d1), V(a.d0), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32);
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32);

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3), V(tmp192.d2));
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
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
    carry= AS_UINT_V(tmp96.d0 > b.d0);
    a.d0 = b.d0 - tmp96.d0;

    a.d1 = b.d1 - tmp96.d1 + carry;
    carry= AS_UINT_V((a.d1 > b.d1) || (carry && AS_UINT_V(a.d1 == b.d1)));

    a.d2 = b.d2 - tmp96.d2 + carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        V(b.d2), V(b.d1), V(b.d0), V(a.d2), V(a.d1), V(a.d0));
#endif
    if(shifter&0x80000000)
    {
      shl_96(&a);					// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 3)
      if (tid==TRACE_TID) printf((__constant char *)"loop shl: %x:%x:%x (a)\n",
        V(a.d2), V(a.d1), V(a.d0) );
#endif
    }
    shifter+=shifter;
  }
#if (TRACE_KERNEL > 0)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: f=%x:%x:%x, a=%x:%x:%x\n",
        V(f.d2), V(f.d1), V(f.d0), V(a.d2), V(a.d1), V(a.d0) );
#endif
  mod_simple_even_96_and_check_big_factor96(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , 15, 10 , modbasecase_debug
#endif
                       );
}

void check_barrett32_88(uint shifter, const int96_v f, const uint tid, const int192_t bb, const uint bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int96_v  a, u, tmp96;
  __private int192_v b, tmp192;
#if defined USE_DP
  __private double_v  ffd;
#endif
  __private float_v  ff;
  __private uint_v   carry;
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */

/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffc) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.

#if defined USE_DP
  // use double when available, but not for the low and mid level GCN cards as they have 1:16 DP rate which is too slow
  ffd= CONVERT_DOUBLE_RTP_V(f.d2);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d1);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d0);

  ffd= as_double(0x3feffffffffffffdL) / ffd;		// just a little bit below 1.0f so we allways underestimate the quotient

#ifndef CHECKS_MODBASECASE
  div_192_96_d(&u, 1 << bit_max65, f, ffd);						// u = floor(2^(80*2) / f)
#else
  div_192_96_d(&u, 1 << bit_max65, f, ffd, modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett88: u=%x:%x:%x, ffd=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else // no doubles or GCN (low/mid)
#ifndef CHECKS_MODBASECASE
  div_192_96(&u,1 << bit_max65,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_192_96(&u,1 << bit_max65,f,ff,modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett88: u=%x:%x:%x, ff=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_88: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3), V(tmp192.d2));
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_88: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  a.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  a.d1 = -tmp96.d1 + AS_UINT_V(tmp96.d0 > 0);
  a.d2 = bb.d2-tmp96.d2 + AS_UINT_V((tmp96.d1 | tmp96.d0) > 0);	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_88: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        bb.d2, bb.d1, bb.d0, V(a.d2), V(a.d1), V(a.d0));
#endif
  while(shifter)
  {                                 // On input a is at most 93 bits (see end of this loop)
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits
    if(shifter&0x80000000)shl_192(&b);	// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d2), V(a.d1), V(a.d0), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32);
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32);

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3), V(tmp192.d2));
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
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
    carry= AS_UINT_V(tmp96.d0 > b.d0);
    a.d0 = b.d0 - tmp96.d0;

    a.d1  = b.d1 - tmp96.d1 + carry;
    carry= AS_UINT_V((a.d1 > b.d1) || (carry && AS_UINT_V(a.d1 == b.d1)));

    a.d2 = b.d2 - tmp96.d2 + carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x - tmp = %x:%x:%x (a)\n",
        V(b.d2), V(b.d1), V(b.d0), V(a.d2), V(a.d1), V(a.d0));
#endif

    shifter+=shifter;
#if (TRACE_KERNEL > 0)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        shifter, V(tmp96.d2), V(tmp96.d1), V(tmp96.d0), V(f.d2), V(f.d1), V(f.d0), V(a.d2), V(a.d1), V(a.d0) );
#endif
  }
  mod_simple_96_and_check_big_factor96(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , 15, 10 , modbasecase_debug
#endif
                       );
}


void check_barrett32_92(uint shifter, const int96_v f, const uint tid, const int192_t bb, const uint bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int96_v  a, u, tmp96;
  __private int192_v b, tmp192;
#if defined USE_DP
  __private double_v  ffd;
#endif
  __private float_v  ff;
  __private uint_v   carry;
  __private int bit_max65_32 = 32 - bit_max65; /* used for bit shifting... */

/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(f.d2);
  ff= ff * 4294967296.0f + CONVERT_FLOAT_RTP_V(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff= as_float(0x3f7ffffc) / ff;		// we rounded ff towards plus infinity, and round all other results towards zero.

#if defined USE_DP
  // use double when available, but not for the low and mid level GCN cards as they have 1:16 DP rate which is too slow
  ffd= CONVERT_DOUBLE_RTP_V(f.d2);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d1);
  ffd= ffd * 4294967296.0 + CONVERT_DOUBLE_RTP_V(f.d0);

  ffd= as_double(0x3feffffffffffffdL) / ffd;		// just a little bit below 1.0f so we allways underestimate the quotient

#ifndef CHECKS_MODBASECASE
  div_192_96_d(&u, 1 << bit_max65, f, ffd);						// u = floor(2^(80*2) / f)
#else
  div_192_96_d(&u, 1 << bit_max65, f, ffd, modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett92: u=%x:%x:%x, ffd=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else // no doubles or GCN (low/mid)
#ifndef CHECKS_MODBASECASE
  div_192_96(&u,1 << bit_max65,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_192_96(&u,1 << bit_max65,f,ff,modbasecase_debug);				// u = floor((2^80)*2 / f)
#endif
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"mfakto_cl_barrett92: u=%x:%x:%x, ff=%G\n",
        V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  a.d0 = (bb.d2 >> bit_max65) + (bb.d3 << bit_max65_32);	// a = floor(b / 2 ^ (bits_in_f - 1))
  a.d1 = (bb.d3 >> bit_max65) + (bb.d4 << bit_max65_32);
  a.d2 = (bb.d4 >> bit_max65) + (bb.d5 << bit_max65_32);

  mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_92: a=%x:%x:%x * u = %x:%x:%x:%x...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3), V(tmp192.d2));
#endif

  a.d0 = tmp192.d3;			     		// a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
  a.d1 = tmp192.d4;
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);					// tmp96 = quotient * f, we only compute the low 96-bits here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_92: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
  // bb.d0-bb.d1 are all zero due to preprocessing on the host
  // carry= AS_UINT_V((tmp96.d0 > bb.d0) ? 1 : 0);
  tmp96.d0 = -tmp96.d0;                // Compute the remainder, we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = -tmp96.d1 + AS_UINT_V(tmp96.d0 > 0);
  tmp96.d2 = bb.d2-tmp96.d2 + AS_UINT_V((tmp96.d1 | tmp96.d0) > 0);	 // if any bit of d0 or d1 is set, we'll have a borrow here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_92: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        bb.d2, bb.d1, bb.d0, V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif

  while(shifter)
  {                                 // On input a is at most 93 bits (see end of this loop)
#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );                         // Adjustment.  The code above may produce an a that is too large by up to 6 times f.

#else
  int limit = 6;
  if(bit_max65 == 1) limit = 8;						// bit_max == 65, due to decreased accuracy of mul_96_192_no_low2() above we need a higher threshold
  if(bit_max65 == 2) limit = 7;						// bit_max == 66, ...
  mod_simple_96(&a, tmp96, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max65, limit, modbasecase_debug);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_92: tmp=%x:%x:%x mod f=%x:%x:%x = %x:%x:%x (a)\n",
        V(tmp96.d2), V(tmp96.d1), V(tmp96.d0), V(f.d2), V(f.d1), V(f.d0), V(a.d2), V(a.d1), V(a.d0) );
#endif
    square_96_192(&b, a);						// b = a^2, b is at most 186 bits

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d2), V(a.d1), V(a.d0), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    a.d0 = (b.d2 >> bit_max65) + (b.d3 << bit_max65_32); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
    a.d1 = (b.d3 >> bit_max65) + (b.d4 << bit_max65_32); // this here is the reason bit_max needs to be 66 at least:
    a.d2 = (b.d4 >> bit_max65) + (b.d5 << bit_max65_32); // OpenCL does not shift by 32 bits

    mul_96_192_no_low3(&tmp192, a, u);			// tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * u = %x:%x:%x...\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp192.d5), V(tmp192.d4), V(tmp192.d3));
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
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x * f = %x:%x:%x (tmp)\n",
        V(a.d2), V(a.d1), V(a.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
    carry= AS_UINT_V(tmp96.d0 > b.d0);
    tmp96.d0 = b.d0 - tmp96.d0;

    tmp96.d1 = b.d1 - tmp96.d1 + carry;
    carry    = AS_UINT_V((tmp96.d1 > b.d1) || (carry && AS_UINT_V(tmp96.d1 == b.d1)));

    tmp96.d2 = b.d2 - tmp96.d2 + carry;	 // we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x - tmp = %x:%x:%x (tmp)\n",
        V(b.d2), V(b.d1), V(b.d0), V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
							// At this point tmp96 is 92 bits + log2 (6) bits to account for the fact that
							// the quotient was up to 6 too small.  This is 94.585 bits.

    if(shifter&0x80000000)shl_96(&tmp96);			// Optional multiply by 2.  At this point tmp96 can be 95.585 bits.

    shifter+=shifter;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, tmp=%x:%x:%x\n",
        shifter, V(tmp96.d2), V(tmp96.d1), V(tmp96.d0));
#endif
  }
  mod_simple_even_96_and_check_big_factor96(tmp96, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , 15, 10 , modbasecase_debug
#endif
                       );
}

/*
 * now the actual kernels: first the ones based on the CPU sieve
 *
 * shiftcount is used for precomputing without mod
 * b is precomputed on host ONCE.
 *
 * bit_max65 is bit_max - 65 (used in the "big" kernels only)
 */

#ifndef CL_GPU_SIEVE

__kernel void cl_barrett32_76(__private uint exponent, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           const __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF)
{
  __private int96_v f;
  __private uint    tid;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC32(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_76: exp=%d, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        exponent, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett32_76(exponent << (32 - shiftcount), f, tid, bb, RES
                     MODBASECASE_PAR);
}

__kernel void cl_barrett32_77(__private uint exponent, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           const __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF)
{
  __private int96_v f;
  __private uint    tid;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC32(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_77: exp=%d, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        exponent, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett32_77(exponent << (32 - shiftcount), f, tid, bb, RES
                     MODBASECASE_PAR);
}

__kernel void cl_barrett32_79(__private uint exponent, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF)
{
  __private int96_v f;
  __private uint    tid;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC32(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_79: exp=%d, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        exponent, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett32_79(exponent << (32 - shiftcount), f, tid, bb, RES
                     MODBASECASE_PAR);
}

__kernel void cl_barrett32_87(__private uint exponent, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF)
{
  __private int96_v f;
  __private uint    tid;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC32(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_87: exp=%d, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        exponent, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett32_87(exponent << (32 - shiftcount), f, tid, bb, bit_max65, RES
                     MODBASECASE_PAR);
}

__kernel void cl_barrett32_88(__private uint exponent, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF)
{
  __private int96_v f;
  __private uint    tid;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC32(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_88: exp=%d, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        exponent, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett32_88(exponent << (32 - shiftcount), f, tid, bb, bit_max65, RES
                     MODBASECASE_PAR);
}

__kernel void cl_barrett32_92(__private uint exponent, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF)
{
  __private int96_v f;
  __private uint    tid;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC32(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_92: exp=%d, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x, f=%x:%x:%x, shift=%d\n",
        exponent, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0, V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif
  check_barrett32_92(exponent << (32 - shiftcount), f, tid, bb, bit_max65, RES
                     MODBASECASE_PAR);
}

#else


/****************************************
 ****************************************
 * 32-bit-kernel consuming the GPU sieve
 * included by main kernel file
 ****************************************
 ****************************************/

__kernel void cl_barrett32_76_gs(__private uint exponent, const int96_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                                 const uint8 b_in,
#else
                                 __private int192_t bb,
#endif
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int96_v  my_k_base, f;
  __private uint     tid, lid=get_local_id(0);
  __private uint_v   tmp_v;
#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif
#ifdef INTEL
  // WA for another bug
  uint num_c;
#endif

tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett32_76_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_76_gs: shift=%d, shifted exp=%#x\n",
        shiftcount, initial_shifter_value);
#endif
#ifdef INTEL
  // WA for another bug
  num_c = NUM_CLASSES % (total_bit_count + 1000000);
#endif

  for (i = lid*VECTOR_SIZE; i < total_bit_count; i += 256*VECTOR_SIZE) // VECTOR_SIZE*THREADS_PER_BLOCK
  {
    // if i == total_bit_count-1, then we may read up to VECTOR_SIZE-1 elements beyond the array (uninitialized).
    // this can result in the same factor being reported up to VECTOR_SIZE times.

    uint_v k_delta;

// Get the (k - k_base) value to test

#if (VECTOR_SIZE == 1)
    k_delta = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
#elif (VECTOR_SIZE == 2)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
#elif (VECTOR_SIZE == 3)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
#elif (VECTOR_SIZE == 4)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
#elif (VECTOR_SIZE == 8)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
#elif (VECTOR_SIZE == 16)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
    k_delta.s8 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+8]));
    k_delta.s9 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+9]));
    k_delta.sa = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+10]));
    k_delta.sb = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+11]));
    k_delta.sc = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+12]));
    k_delta.sd = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+13]));
    k_delta.se = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+14]));
    k_delta.sf = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+15]));
#endif

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

#ifdef INTEL
  // WA for another bug
    my_k_base.d0 = k_base.d0 + num_c * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(num_c, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#else
    my_k_base.d0 = k_base.d0 + NUM_CLASSES * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(NUM_CLASSES, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif

    f.d0   = my_k_base.d0 * exponent;
    tmp_v  = mul_hi(my_k_base.d0, exponent);
    f.d1   = my_k_base.d1 * exponent + tmp_v;
    f.d2   = mul_hi(my_k_base.d1, exponent) - AS_UINT_V(f.d1 < tmp_v);

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_76_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x, k*p=%x:%x:%x\n",
        i, smem[i], V(k_delta), V(my_k_base.d1), V(my_k_base.d0), V(f.d2), V(f.d1), V(f.d0));
#endif

    // Compute f = 2 * k * exp + 1
    f.d2 = amd_bitalign(f.d2, f.d1, 31);
    f.d1 = amd_bitalign(f.d1, f.d0, 31);
    f.d0 = (f.d0 << 1) + 1;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID)
       printf((__constant char *)"cl_barrett32_76_gs: lid=%u, tid=%u, gid=%u, smem[%u]=%u, k_delta=%u, f=%x:%x:%x\n",
        lid, tid, get_group_id(0), i, smem[i], V(k_delta), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett32_76(initial_shifter_value, f, tid, bb, RES
                       MODBASECASE_PAR);
  }
}

__kernel void cl_barrett32_77_gs(__private uint exponent, const int96_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                                 const uint8 b_in,
#else
                                 __private int192_t bb,
#endif
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int96_v  my_k_base, f;
  __private uint     tid, lid=get_local_id(0);
  __private uint_v   tmp_v;
#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif
#ifdef INTEL
  // WA for another bug
  uint num_c;
#endif

tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett32_77_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_77_gs: shift=%d, shifted exp=%#x\n",
        shiftcount, initial_shifter_value);
#endif
#ifdef INTEL
  // WA for another bug
  num_c = NUM_CLASSES % (total_bit_count + 1000000);
#endif

  for (i = lid*VECTOR_SIZE; i < total_bit_count; i += 256*VECTOR_SIZE) // VECTOR_SIZE*THREADS_PER_BLOCK
  {
    // if i == total_bit_count-1, then we may read up to VECTOR_SIZE-1 elements beyond the array (uninitialized).
    // this can result in the same factor being reported up to VECTOR_SIZE times.

    uint_v k_delta;

// Get the (k - k_base) value to test

#if (VECTOR_SIZE == 1)
    k_delta = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
#elif (VECTOR_SIZE == 2)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
#elif (VECTOR_SIZE == 3)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
#elif (VECTOR_SIZE == 4)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
#elif (VECTOR_SIZE == 8)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
#elif (VECTOR_SIZE == 16)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
    k_delta.s8 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+8]));
    k_delta.s9 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+9]));
    k_delta.sa = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+10]));
    k_delta.sb = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+11]));
    k_delta.sc = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+12]));
    k_delta.sd = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+13]));
    k_delta.se = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+14]));
    k_delta.sf = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+15]));
#endif

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

#ifdef INTEL
  // WA for another bug
    my_k_base.d0 = k_base.d0 + num_c * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(num_c, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#else
    my_k_base.d0 = k_base.d0 + NUM_CLASSES * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(NUM_CLASSES, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif

    f.d0   = my_k_base.d0 * exponent;
    tmp_v  = mul_hi(my_k_base.d0, exponent);
    f.d1   = my_k_base.d1 * exponent + tmp_v;
    f.d2   = mul_hi(my_k_base.d1, exponent) - AS_UINT_V(f.d1 < tmp_v);

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_77_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x, k*p=%x:%x:%x\n",
        i, smem[i], V(k_delta), V(my_k_base.d1), V(my_k_base.d0), V(f.d2), V(f.d1), V(f.d0));
#endif

    // Compute f = 2 * k * exp + 1
    f.d2 = amd_bitalign(f.d2, f.d1, 31);
    f.d1 = amd_bitalign(f.d1, f.d0, 31);
    f.d0 = (f.d0 << 1) + 1;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID)
       printf((__constant char *)"cl_barrett32_77_gs: lid=%u, tid=%u, gid=%u, smem[%u]=%u, k_delta=%u, f=%x:%x:%x\n",
        lid, tid, get_group_id(0), i, smem[i], V(k_delta), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett32_77(initial_shifter_value, f, tid, bb, RES
                       MODBASECASE_PAR);
  }
}

__kernel void cl_barrett32_79_gs(__private uint exponent, const int96_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                                 const uint8 b_in,
#else
                                 __private int192_t bb,
#endif
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int96_v  my_k_base, f;
  __private uint     tid, lid=get_local_id(0);
  __private uint_v   tmp_v;
#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif
#ifdef INTEL
  // WA for another bug
  uint num_c;
#endif

tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett32_79_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_79_gs: shift=%d, shifted exp=%#x\n",
        shiftcount, initial_shifter_value);
#endif
#ifdef INTEL
  // WA for another bug
  num_c = NUM_CLASSES % (total_bit_count + 1000000);
#endif

  for (i = lid*VECTOR_SIZE; i < total_bit_count; i += 256*VECTOR_SIZE) // VECTOR_SIZE*THREADS_PER_BLOCK
  {
    // if i == total_bit_count-1, then we may read up to VECTOR_SIZE-1 elements beyond the array (uninitialized).
    // this can result in the same factor being reported up to VECTOR_SIZE times.

    uint_v k_delta;

// Get the (k - k_base) value to test

#if (VECTOR_SIZE == 1)
    k_delta = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
#elif (VECTOR_SIZE == 2)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
#elif (VECTOR_SIZE == 3)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
#elif (VECTOR_SIZE == 4)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
#elif (VECTOR_SIZE == 8)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
#elif (VECTOR_SIZE == 16)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
    k_delta.s8 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+8]));
    k_delta.s9 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+9]));
    k_delta.sa = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+10]));
    k_delta.sb = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+11]));
    k_delta.sc = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+12]));
    k_delta.sd = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+13]));
    k_delta.se = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+14]));
    k_delta.sf = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+15]));
#endif

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

#ifdef INTEL
  // WA for another bug
    my_k_base.d0 = k_base.d0 + num_c * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(num_c, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#else
    my_k_base.d0 = k_base.d0 + NUM_CLASSES * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(NUM_CLASSES, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif

    f.d0   = my_k_base.d0 * exponent;
    tmp_v  = mul_hi(my_k_base.d0, exponent);
    f.d1   = my_k_base.d1 * exponent + tmp_v;
    f.d2   = mul_hi(my_k_base.d1, exponent) - AS_UINT_V(f.d1 < tmp_v);

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_79_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x, k*p=%x:%x:%x\n",
        i, smem[i], V(k_delta), V(my_k_base.d1), V(my_k_base.d0), V(f.d2), V(f.d1), V(f.d0));
#endif

    // Compute f = 2 * k * exp + 1
    f.d2 = amd_bitalign(f.d2, f.d1, 31);
    f.d1 = amd_bitalign(f.d1, f.d0, 31);
    f.d0 = (f.d0 << 1) + 1;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID)
       printf((__constant char *)"cl_barrett32_79_gs: lid=%u, tid=%u, gid=%u, smem[%u]=%u, k_delta=%u, f=%x:%x:%x\n",
        lid, tid, get_group_id(0), i, smem[i], V(k_delta), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett32_79(initial_shifter_value, f, tid, bb, RES
                       MODBASECASE_PAR);
  }
}

__kernel void cl_barrett32_87_gs(__private uint exponent, const int96_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                                 const uint8 b_in,
#else
                                 __private int192_t bb,
#endif
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int96_v  my_k_base, f;
  __private uint     tid, lid=get_local_id(0);
  __private uint_v   tmp_v;
#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif
#ifdef INTEL
  // WA for another bug
  uint num_c;
#endif

tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett32_87_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_87_gs: shift=%d, shifted exp=%#x\n",
        shiftcount, initial_shifter_value);
#endif
#ifdef INTEL
  // WA for another bug
  num_c = NUM_CLASSES % (total_bit_count + 1000000);
#endif

  for (i = lid*VECTOR_SIZE; i < total_bit_count; i += 256*VECTOR_SIZE) // VECTOR_SIZE*THREADS_PER_BLOCK
  {
    // if i == total_bit_count-1, then we may read up to VECTOR_SIZE-1 elements beyond the array (uninitialized).
    // this can result in the same factor being reported up to VECTOR_SIZE times.

    uint_v k_delta;

// Get the (k - k_base) value to test

#if (VECTOR_SIZE == 1)
    k_delta = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
#elif (VECTOR_SIZE == 2)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
#elif (VECTOR_SIZE == 3)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
#elif (VECTOR_SIZE == 4)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
#elif (VECTOR_SIZE == 8)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
#elif (VECTOR_SIZE == 16)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
    k_delta.s8 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+8]));
    k_delta.s9 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+9]));
    k_delta.sa = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+10]));
    k_delta.sb = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+11]));
    k_delta.sc = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+12]));
    k_delta.sd = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+13]));
    k_delta.se = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+14]));
    k_delta.sf = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+15]));
#endif

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

#ifdef INTEL
  // WA for another bug
    my_k_base.d0 = k_base.d0 + num_c * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(num_c, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#else
    my_k_base.d0 = k_base.d0 + NUM_CLASSES * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(NUM_CLASSES, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif

    f.d0   = my_k_base.d0 * exponent;
    tmp_v  = mul_hi(my_k_base.d0, exponent);
    f.d1   = my_k_base.d1 * exponent + tmp_v;
    f.d2   = mul_hi(my_k_base.d1, exponent) - AS_UINT_V(f.d1 < tmp_v);

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_87_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x, k*p=%x:%x:%x\n",
        i, smem[i], V(k_delta), V(my_k_base.d1), V(my_k_base.d0), V(f.d2), V(f.d1), V(f.d0));
#endif

    // Compute f = 2 * k * exp + 1
    f.d2 = amd_bitalign(f.d2, f.d1, 31);
    f.d1 = amd_bitalign(f.d1, f.d0, 31);
    f.d0 = (f.d0 << 1) + 1;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID)
       printf((__constant char *)"cl_barrett32_87_gs: lid=%u, tid=%u, gid=%u, smem[%u]=%u, k_delta=%u, f=%x:%x:%x\n",
        lid, tid, get_group_id(0), i, smem[i], V(k_delta), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett32_87(initial_shifter_value, f, tid, bb, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

__kernel void cl_barrett32_88_gs(__private uint exponent, const int96_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                                 const uint8 b_in,
#else
                                 __private int192_t bb,
#endif
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int96_v  my_k_base, f;
  __private uint     tid, lid=get_local_id(0);
  __private uint_v   tmp_v;
#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif
#ifdef INTEL
  // WA for another bug
  uint num_c;
#endif

tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett32_88_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_88_gs: shift=%d, shifted exp=%#x\n",
        shiftcount, initial_shifter_value);
#endif
#ifdef INTEL
  // WA for another bug
  num_c = NUM_CLASSES % (total_bit_count + 1000000);
#endif

  for (i = lid*VECTOR_SIZE; i < total_bit_count; i += 256*VECTOR_SIZE) // VECTOR_SIZE*THREADS_PER_BLOCK
  {
    // if i == total_bit_count-1, then we may read up to VECTOR_SIZE-1 elements beyond the array (uninitialized).
    // this can result in the same factor being reported up to VECTOR_SIZE times.

    uint_v k_delta;

// Get the (k - k_base) value to test

#if (VECTOR_SIZE == 1)
    k_delta = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
#elif (VECTOR_SIZE == 2)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
#elif (VECTOR_SIZE == 3)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
#elif (VECTOR_SIZE == 4)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
#elif (VECTOR_SIZE == 8)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
#elif (VECTOR_SIZE == 16)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
    k_delta.s8 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+8]));
    k_delta.s9 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+9]));
    k_delta.sa = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+10]));
    k_delta.sb = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+11]));
    k_delta.sc = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+12]));
    k_delta.sd = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+13]));
    k_delta.se = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+14]));
    k_delta.sf = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+15]));
#endif

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

#ifdef INTEL
  // WA for another bug
    my_k_base.d0 = k_base.d0 + num_c * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(num_c, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#else
    my_k_base.d0 = k_base.d0 + NUM_CLASSES * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(NUM_CLASSES, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif

    f.d0   = my_k_base.d0 * exponent;
    tmp_v  = mul_hi(my_k_base.d0, exponent);
    f.d1   = my_k_base.d1 * exponent + tmp_v;
    f.d2   = mul_hi(my_k_base.d1, exponent) - AS_UINT_V(f.d1 < tmp_v);

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_88_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x, k*p=%x:%x:%x\n",
        i, smem[i], V(k_delta), V(my_k_base.d1), V(my_k_base.d0), V(f.d2), V(f.d1), V(f.d0));
#endif

    // Compute f = 2 * k * exp + 1
    f.d2 = amd_bitalign(f.d2, f.d1, 31);
    f.d1 = amd_bitalign(f.d1, f.d0, 31);
    f.d0 = (f.d0 << 1) + 1;

#if (TRACE_KERNEL > 0)
    if (tid==TRACE_TID)
       printf((__constant char *)"cl_barrett32_88_gs: lid=%u, tid=%u, gid=%u, smem[%u]=%u, k_delta=%u, f=%x:%x:%x\n",
        lid, tid, get_group_id(0), i, smem[i], V(k_delta), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett32_88(initial_shifter_value, f, tid, bb, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

__kernel void cl_barrett32_92_gs(__private uint exponent, const int96_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                                 const uint8 b_in,
#else
                                 __private int192_t bb,
#endif
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF)
/*
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int96_v  my_k_base, f;
  __private uint     tid, lid=get_local_id(0);
  __private uint_v   tmp_v;
#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif
#ifdef INTEL
  // WA for another bug
  uint num_c;
#endif

tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett32_92_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_92_gs: shift=%d, shifted exp=%#x\n",
        shiftcount, initial_shifter_value);
#endif
#ifdef INTEL
  // WA for another bug
  num_c = NUM_CLASSES % (total_bit_count + 1000000);
#endif

  for (i = lid*VECTOR_SIZE; i < total_bit_count; i += 256*VECTOR_SIZE) // VECTOR_SIZE*THREADS_PER_BLOCK
  {
    // if i == total_bit_count-1, then we may read up to VECTOR_SIZE-1 elements beyond the array (uninitialized).
    // this can result in the same factor being reported up to VECTOR_SIZE times.

    uint_v k_delta;

// Get the (k - k_base) value to test

#if (VECTOR_SIZE == 1)
    k_delta = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
#elif (VECTOR_SIZE == 2)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
#elif (VECTOR_SIZE == 3)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
#elif (VECTOR_SIZE == 4)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
#elif (VECTOR_SIZE == 8)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
#elif (VECTOR_SIZE == 16)
    k_delta.s0 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i]));
    k_delta.s1 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+1]));
    k_delta.s2 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+2]));
    k_delta.s3 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+3]));
    k_delta.s4 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+4]));
    k_delta.s5 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+5]));
    k_delta.s6 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+6]));
    k_delta.s7 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+7]));
    k_delta.s8 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+8]));
    k_delta.s9 = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+9]));
    k_delta.sa = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+10]));
    k_delta.sb = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+11]));
    k_delta.sc = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+12]));
    k_delta.sd = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+13]));
    k_delta.se = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+14]));
    k_delta.sf = mad24(bits_to_process, (uint)get_group_id(0), (uint)(smem[i+15]));
#endif

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

#ifdef INTEL
  // WA for another bug
    my_k_base.d0 = k_base.d0 + num_c * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(num_c, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#else
    my_k_base.d0 = k_base.d0 + NUM_CLASSES * k_delta;  // k_delta can exceed 2^24: don't use mul24/mad24 for it
    my_k_base.d1 = k_base.d1 + mul_hi(NUM_CLASSES, k_delta) - AS_UINT_V(k_base.d0 > my_k_base.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif

    f.d0   = my_k_base.d0 * exponent;
    tmp_v  = mul_hi(my_k_base.d0, exponent);
    f.d1   = my_k_base.d1 * exponent + tmp_v;
    f.d2   = mul_hi(my_k_base.d1, exponent) - AS_UINT_V(f.d1 < tmp_v);

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett32_92_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x, k*p=%x:%x:%x\n",
        i, smem[i], V(k_delta), V(my_k_base.d1), V(my_k_base.d0), V(f.d2), V(f.d1), V(f.d0));
#endif

    // Compute f = 2 * k * exp + 1
    f.d2 = amd_bitalign(f.d2, f.d1, 31);
    f.d1 = amd_bitalign(f.d1, f.d0, 31);
    f.d0 = (f.d0 << 1) + 1;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID)
       printf((__constant char *)"cl_barrett32_92_gs: lid=%u, tid=%u, gid=%u, smem[%u]=%u, k_delta=%u, f=%x:%x:%x\n",
        lid, tid, get_group_id(0), i, smem[i], V(k_delta), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett32_92(initial_shifter_value, f, tid, bb, bit_max65, RES
                       MODBASECASE_PAR);
  }
}
#endif