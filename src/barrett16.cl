/*
This file is part of mfaktc (mfakto).
Copyright (C) 2014 - 2014  Bertram Franz (bertramf@gmx.net)

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
 * 16-bit based kernels
 *
 ****************************************
 ****************************************/

int80_v sub_if_gte_80(const int80_v a, const int80_v b)
/* return (a>b)?a-b:a */
{
  int80_v tmp;
  /* do the subtraction and use tmp.d4 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0);
  tmp.d1 = (a.d1 - b.d1 + AS_UINT_V((tmp.d0 > 0xFFFF) ));
  tmp.d2 = (a.d2 - b.d2 + AS_UINT_V((tmp.d1 > 0xFFFF) ));
  tmp.d3 = (a.d3 - b.d3 + AS_UINT_V((tmp.d2 > 0xFFFF) ));
  tmp.d4 = (a.d4 - b.d4 + AS_UINT_V((tmp.d3 > 0xFFFF) ));
  tmp.d0&= 0xFFFF;
  tmp.d1&= 0xFFFF;
  tmp.d2&= 0xFFFF;
  tmp.d3&= 0xFFFF;

  tmp.d0 = (tmp.d4 > a.d4) ? a.d0 : tmp.d0;
  tmp.d1 = (tmp.d4 > a.d4) ? a.d1 : tmp.d1;
  tmp.d2 = (tmp.d4 > a.d4) ? a.d2 : tmp.d2;
  tmp.d3 = (tmp.d4 > a.d4) ? a.d3 : tmp.d3;
  tmp.d4 = (tmp.d4 > a.d4) ? a.d4 : tmp.d4; //  & 0xFFFF not necessary as tmp.d4 is <= a.d4

  return tmp;
}

void mul_80(int80_v * const res, const int80_v a, const int80_v b)
/* res = a * b (lower 80 bits)
 15x mul24/mad24, 12x >>, 13x &, 8x + = 48 ops (per vector element) */
{
  uint_v t1, t2, t3, t4;

  res->d0 = mul24(a.d0, b.d0);

  t1      = mad24(a.d1, b.d0, res->d0 >> 16);
  t2      = mad24(a.d0, b.d1, t1 & 0xFFFF);
  res->d1 = (t2 & 0xFFFF);
  res->d0 &= 0xFFFF;

  t1      = mad24(a.d2, b.d0, t1 >> 16);
  t2      = mad24(a.d1, b.d1, t2 >> 16);
  t3      = mad24(a.d0, b.d2, t1 & 0xFFFF);
  res->d2 = (t2 & 0xFFFF) + (t3 & 0xFFFF);

  t4      = mad24(a.d2, b.d1, t1 >> 16);
  t1      = mad24(a.d1, b.d2, t2 >> 16);
  t2      = mad24(a.d0, b.d3, t3 >> 16);
  t3      = mad24(a.d3, b.d0, res->d2 >> 16);
  res->d2 &= 0xFFFF;
  res->d3 = (t1 & 0xFFFF) + (t2 & 0xFFFF) + (t3 & 0xFFFF) + (t4 & 0xFFFF);
  res->d4 = (t1 >> 16)    + (t2 >> 16)    + (t3 >> 16)    + (t4 >> 16) + (res->d3 >> 16);
  res->d3 &= 0xFFFF;

  res->d4 = mad24(a.d4, b.d0, res->d4);  // d4 will overflow, but that's ok for this function.
  res->d4 = mad24(a.d3, b.d1, res->d4);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
  res->d4 = mad24(a.d0, b.d4, res->d4);
  res->d4 &= 0xFFFF;

}


void mul_80_160_no_low3(int160_v * const res, const int80_v a, const int80_v b)
/*
res ~= a * b
res.d0 to res.d2 are NOT computed. Carries to res.d3 are ignored,
too. So the digits res.d{3-9} might differ from mul_80_160().
 */
{
  uint_v t1, t2, t3, t4, t5;
  // 0xFFFF * 0xFFFF = 0xFFFE0001 = max result of mul24.
  // Up to two 16-bit carries can be added into 32 bits.
  // this optimized mul 5x5 requires: 19 mul/mad24, 7 shift, 6 and, 1 add = 33 ops (15-bit version)
  // this optimized mul 5x5 requires: 19 mul/mad24, 23 shift, 22 and, 17 add = 81 ops

  t1      = mul24(a.d3, b.d0);
  t2      = mad24(a.d2, b.d1, t1 & 0xFFFF);
  t3      = mad24(a.d1, b.d2, t2 & 0xFFFF);
  t4      = mad24(a.d0, b.d3, t3 & 0xFFFF);

  t5      = mad24(a.d4, b.d0, t1 >> 16);
  t1      = mad24(a.d3, b.d1, t2 >> 16);
  t2      = mad24(a.d2, b.d2, t3 >> 16);
  t3      = mad24(a.d1, b.d3, t4 >> 16);
  t4      = mad24(a.d0, b.d4, t5 & 0xFFFF);

  res->d4 = (t1 & 0xFFFF) + (t2 & 0xFFFF) + (t3 & 0xFFFF) + (t4 & 0xFFFF);
  res->d5 = (res->d4 >> 16) + (t1 >> 16);
  res->d4 &= 0xFFFF;

  t1      = mad24(a.d4, b.d1, t2 >> 16);
  t2      = mad24(a.d3, b.d2, t3 >> 16);
  t3      = mad24(a.d2, b.d3, t4 >> 16);
  t4      = mad24(a.d1, b.d4, t5 >> 16);
  res->d5 += (t1 & 0xFFFF) + (t2 & 0xFFFF) + (t3 & 0xFFFF) + (t4 & 0xFFFF);
  res->d6 = (res->d5 >> 16) + (t1 >> 16);
  res->d5 &= 0xFFFF;

  t1      = mad24(a.d2, b.d4, t2 >> 16);
  t2      = mad24(a.d3, b.d3, t3 >> 16);
  t3      = mad24(a.d4, b.d2, t4 >> 16);
  res->d6 += (t1 & 0xFFFF) + (t2 & 0xFFFF) + (t3 & 0xFFFF);
  res->d7 = (res->d6 >> 16) + (t1 >> 16);
  res->d6 &= 0xFFFF;

  t1      = mad24(a.d3, b.d4, t2 >> 16);
  t2      = mad24(a.d4, b.d3, t3 >> 16);
  res->d7 += (t1 & 0xFFFF) + (t2 & 0xFFFF);
  res->d8 = (res->d7 >> 16) + (t1 >> 16);
  res->d7 &= 0xFFFF;

  res->d8 += mad24(a.d4, b.d4, t2 >> 16);

  res->d9 = res->d8 >> 16;
  res->d8 &= 0xFFFF;
}

void mul_80_160_no_low5(int160_v * const res, const int80_v a, const int80_v b)
/*
res ~= a * b
res.d0 to res.d3 are NOT computed. res.d4 is computed only to get its upper half carried to res.d5.
Due to the missing carries from d3 into d4, at most a 17-bit value is missing in d4. This means,
d4 >> 16 can be too low by up to 3, thus d5 can be low by 3.
 */
{
  uint_v t1, t2, t3, t4, t5;
  // 0xFFFF * 0xFFFF = 0xFFFE0001 = max result of mul24.
  // Up to two 16-bit carries can be added into 32 bits.
  // this optimized mul 5x5 requires: 19 mul/mad24, 7 shift, 6 and, 1 add = 33 ops (15-bit version)
  // this optimized mul 5x5 requires: 15 mul/mad24, 18 shift, 17 and, 13 add = 63 ops

  t1      = mul24(a.d4, b.d0);
  t2      = mad24(a.d3, b.d1, t1 & 0xFFFF);
  t3      = mad24(a.d2, b.d2, t2 & 0xFFFF);
  t4      = mad24(a.d1, b.d3, t3 & 0xFFFF);
  t5      = mad24(a.d0, b.d4, t4 & 0xFFFF);

  res->d5 = (t1 >> 16);

  t1      = mad24(a.d4, b.d1, t2 >> 16);
  t2      = mad24(a.d3, b.d2, t3 >> 16);
  t3      = mad24(a.d2, b.d3, t4 >> 16);
  t4      = mad24(a.d1, b.d4, t5 >> 16);
  res->d5 += (t1 & 0xFFFF) + (t2 & 0xFFFF) + (t3 & 0xFFFF) + (t4 & 0xFFFF);
  res->d6 = (res->d5 >> 16) + (t1 >> 16);
  res->d5 &= 0xFFFF;

  t1      = mad24(a.d2, b.d4, t2 >> 16);
  t2      = mad24(a.d3, b.d3, t3 >> 16);
  t3      = mad24(a.d4, b.d2, t4 >> 16);
  res->d6 += (t1 & 0xFFFF) + (t2 & 0xFFFF) + (t3 & 0xFFFF);
  res->d7 = (res->d6 >> 16) + (t1 >> 16);
  res->d6 &= 0xFFFF;

  t1      = mad24(a.d3, b.d4, t2 >> 16);
  t2      = mad24(a.d4, b.d3, t3 >> 16);
  res->d7 += (t1 & 0xFFFF) + (t2 & 0xFFFF);
  res->d8 = (res->d7 >> 16) + (t1 >> 16);
  res->d7 &= 0xFFFF;

  res->d8 += mad24(a.d4, b.d4, t2 >> 16);

  res->d9 = res->d8 >> 16;
  res->d8 &= 0xFFFF;
}

#if 0
void mul_80_160(int160_v * const res, const int80_v a, const int80_v b)
/*
res = a * b
 */
{
  /* this is the complete implementation, no longer used, but was the basis for
     the _no_low3 and square functions */
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0xFFFF * 0xFFFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 16)
  // mul 5x5 requires: 25 mul/mad24, 10 shift, 10 and, 1 add

  res->d0 = mul24(a.d0, b.d0);

  res->d1 = mad24(a.d1, b.d0, res->d0 >> 16);
  res->d1 = mad24(a.d0, b.d1, res->d1);
  res->d0 &= 0xFFFF;

  res->d2 = mad24(a.d2, b.d0, res->d1 >> 16);
  res->d2 = mad24(a.d1, b.d1, res->d2);
  res->d2 = mad24(a.d0, b.d2, res->d2);
  res->d1 &= 0xFFFF;

  res->d3 = mad24(a.d3, b.d0, res->d2 >> 16);
  res->d3 = mad24(a.d2, b.d1, res->d3);
  res->d3 = mad24(a.d1, b.d2, res->d3);
  res->d3 = mad24(a.d0, b.d3, res->d3);
  res->d2 &= 0xFFFF;

  res->d4 = mad24(a.d4, b.d0, res->d3 >> 16);
  res->d3 &= 0xFFFF;
  res->d4 = mad24(a.d3, b.d1, res->d4);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
   // 5th mad24 can overflow d4, need to handle carry before: pull in the first d5 line
  res->d5 = mad24(a.d4, b.d1, res->d4 >> 16);
  res->d4 &= 0xFFFF;
  res->d4 = mad24(a.d0, b.d4, res->d4);  // 31-bit at most

  res->d5 = mad24(a.d3, b.d2, res->d4 >> 16) + res->d5;
  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d5 = mad24(a.d1, b.d4, res->d5);
  res->d4 &= 0xFFFF;
  // now we have in d5: 4x mad24() + 1x 17-bit carry + 1x 16-bit carry: still fits into 32 bits

  res->d6 = mad24(a.d2, b.d4, res->d5 >> 16);
  res->d6 = mad24(a.d3, b.d3, res->d6);
  res->d6 = mad24(a.d4, b.d2, res->d6);
  res->d5 &= 0xFFFF;

  res->d7 = mad24(a.d3, b.d4, res->d6 >> 16);
  res->d7 = mad24(a.d4, b.d3, res->d7);
  res->d6 &= 0xFFFF;

  res->d8 = mad24(a.d4, b.d4, res->d7 >> 16);
  res->d7 &= 0xFFFF;

  res->d9 = res->d8 >> 16;
  res->d8 &= 0xFFFF;
}
#endif

void square_80_160(int160_v * const res, const int80_v a)
/* res = a^2 = d0^2 + 2d0d1 + d1^2 + 2d0d2 + 2(d1d2 + d0d3) + d2^2 +
               2(d0d4 + d1d3) + 2(d1d4 + d2d3) + d3^2 + 2d2d4 + 2d3d4 + d4^2
   */
{
  // 0xFFFF * 0xFFFF = 0xFFFE0001 = max result of mul24.
  // Up to two 16-bit carries can be added into 32 bits.
  // square 5x5 requires: 15 mul/mad24, 20 shift, 10 and, 1 add
  // square 5x5 requires: 15 mul/mad24, 14 shift, 10 and, 1 add
  uint_v t1, t2, t3, t4, t5;

  res->d0 = mul24(a.d0, a.d0);

  t1      = mul24(a.d1, a.d0); // x2
  res->d1 = ((t1 & 0xFFFF) << 1) + (res->d0 >> 16);
  res->d0 &= 0xFFFF;

  t2      = mad24(a.d1, a.d1, res->d1 >> 16);
  t3      = mad24(a.d2, a.d0, t1 >> 16); // x2
  res->d2 = (t2 & 0xFFFF) + ((t3 & 0xFFFF) << 1);
  res->d1 &= 0xFFFF;

  t4      = mad24(a.d3, a.d0, t3 >> 16); // x2
  t5      = mad24(a.d2, a.d1, t4 & 0xFFFF); // x2
  res->d3 = (t2 >> 16) + ((t5 & 0xFFFF) << 1) + (res->d2 >> 16);
  res->d2 &= 0xFFFF;

  t1      = mad24(a.d4, a.d0, t4 >> 16); // x2
  t2      = mad24(a.d3, a.d1, t5 >> 16); // x2
  t3      = mad24(a.d2, a.d2, res->d3 >> 16);
  res->d4 = ((t1 & 0xFFFF) << 1) + ((t2 & 0xFFFF) << 1) + (t3 & 0xFFFF);
  res->d3 &= 0xFFFF;

  t4      = mad24(a.d4, a.d1, t1 >> 16); // x2
  t5      = mad24(a.d3, a.d2, t2 >> 16); // x2
  res->d5 = ((t4 & 0xFFFF) << 1) + ((t5 & 0xFFFF) << 1) + (t3 >> 16) + (res->d4 >> 16);
  res->d4 &= 0xFFFF;

  t1      = mad24(a.d4, a.d2, (t4 >> 16) + (t5 >> 16)); // x2
  t2      = mad24(a.d3, a.d3, res->d5 >> 16);
  res->d6 = (t2 & 0xFFFF) + ((t1 & 0xFFFF) << 1);
  res->d5 &= 0xFFFF;

  t3      = mad24(a.d4, a.d3, t1 >> 16); // x2
  res->d7 = ((t3 & 0xFFFF) << 1) + (t2 >> 16) + (res->d6 >> 16);
  res->d6 &= 0xFFFF;

  t4      = mad24(a.d4, a.d4, res->d7 >> 16);
  res->d8 = ((t3 >> 16) << 1) + (t4 & 0xFFFF);
  res->d7 &= 0xFFFF;

  res->d9 = (res->d8 >> 16) + (t4 >> 16);
  res->d8 &= 0xFFFF;
}


void shl_80(int80_v * const a)
/* shiftleft a one bit */
{
  a->d4 = mad24(a->d4, 2u, a->d3 >> 15); // keep the extra top bit
  a->d3 = mad24(a->d3, 2u, a->d2 >> 15) & 0xFFFF;
  a->d2 = mad24(a->d2, 2u, a->d1 >> 15) & 0xFFFF;
  a->d1 = mad24(a->d1, 2u, a->d0 >> 15) & 0xFFFF;
  a->d0 = (a->d0 << 1u) & 0xFFFF;
}
void shl_160(int160_v * const a)
/* shiftleft a one bit */
{
  a->d9 = mad24(a->d9, 2u, a->d8 >> 15); // keep the extra top bit
  a->d8 = mad24(a->d8, 2u, a->d7 >> 15) & 0xFFFF;
  a->d7 = mad24(a->d7, 2u, a->d6 >> 15) & 0xFFFF;
  a->d6 = mad24(a->d6, 2u, a->d5 >> 15) & 0xFFFF;
  a->d5 = mad24(a->d5, 2u, a->d4 >> 15) & 0xFFFF;
  a->d4 = mad24(a->d4, 2u, a->d3 >> 15) & 0xFFFF;
  a->d3 = mad24(a->d3, 2u, a->d2 >> 15) & 0xFFFF;
  a->d2 = mad24(a->d2, 2u, a->d1 >> 15) & 0xFFFF;
  a->d1 = mad24(a->d1, 2u, a->d0 >> 15) & 0xFFFF;
  a->d0 = (a->d0 << 1u) & 0xFFFF;
}


void div_160_80(int80_v * const res, const uint qhi, const int80_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
                  MODBASECASE_PAR_DEF
)/* res = q / n (integer division) */
{
  __private float_v qf;
  __private float   qf_1;   // for the first conversion which does not need vectors yet
  __private uint_v qi, qil, qih, t1, t2;
  __private int160_v nn, q;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#0: q=%x:<144x0>, n=%x:%x:%x:%x:%x, nf=%#G\n",
        qhi, n.d4.s0, n.d3.s0, n.d2.s0, n.d1.s0, n.d0.s0, nf.s0);
#endif

/********** Step 1, Offset 2^60 (4*16 + 0) **********/
  qf_1= convert_float(qhi) * 281474976710656.0f; // 65536^3

  qi=CONVERT_UINT_V(qf_1*nf);  // vectorize just here

  MODBASECASE_QI_ERROR(1<<16, 1, qi, 0);  // first step is smaller

  res->d4 = qi;
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"div_160_80#1: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:..:..:..:..\n",
                                 qf_1, nf.s0, qf_1*nf.s0, qi.s0, qi.s0, res->d4.s0);
#endif

  /*******************************************************/

// nn = n * qi
  nn.d0  = mul24(n.d0, qi);
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#1.1: nn=..:..:..:..:..:%x:..:..:..:..\n",
        nn.d0.s0);
#endif

  nn.d1  = mad24(n.d1, qi, nn.d0 >> 16);
  nn.d0 &= 0xFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#1.2: nn=..:..:..:..:%x:%x:...\n",
        nn.d1.s0, nn.d0.s0);
#endif

  nn.d2  = mad24(n.d2, qi, nn.d1 >> 16);
  nn.d1 &= 0xFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#1.3: nn=..:..:..:%x:%x:%x:...\n",
        nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d3  = mad24(n.d3, qi, nn.d2 >> 16);
  nn.d2 &= 0xFFFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#1.4: nn=..:..:%x:%x:%x:%x:...\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif
  nn.d4  = mad24(n.d4, qi, nn.d3 >> 16);
  nn.d3 &= 0xFFFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#1.5: nn=..:%x:%x:%x:%x:%x:...\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

// no shift-left
#ifdef CHECKS_MODBASECASE
  nn.d5  = nn.d4 >> 16;  // PERF: not needed as it will be gone anyway after sub
  nn.d4 &= 0xFFFF;
#endif

//  q.d0-q.d8 are all zero
  q.d4 = -nn.d0;
  q.d5 = -nn.d1 + AS_UINT_V((q.d4 > 0xFFFF));
  q.d6 = -nn.d2 + AS_UINT_V((q.d5 > 0xFFFF));
  q.d7 = -nn.d3 + AS_UINT_V((q.d6 > 0xFFFF));
  q.d8 = -nn.d4 + AS_UINT_V((q.d7 > 0xFFFF));
//#ifdef CHECKS_MODBASECASE
  q.d9 = qhi - nn.d5 + AS_UINT_V((q.d8 > 0xFFFF)); // PERF: not needed: should be zero anyway
//#endif
  q.d4 &= 0xFFFF;
  q.d5 &= 0xFFFF;
  q.d6 &= 0xFFFF;
  q.d7 &= 0xFFFF;
  q.d8 &= 0xFFFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#1.7: q=%x!%x:%x:%x:%x:%x:..:..:..:..\n",
        q.d9.s0, q.d8.s0, q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0);
#endif
  MODBASECASE_NONZERO_ERROR(q.d9, 1, 9, 1);

  /********** Step 2, Offset 2^40 (2*16 + 10) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d8, 65536u, q.d7));
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(mad24(q.d6, 65536u, q.d5)); // PERF: q.d5 not needed: q.d8 has 9 bits left, + q.d7 (16) is enough
  qf*= 32.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<21, 2, qi, 2);

  res->d3 = (qi >> 5);
  res->d2 = (qi << 11) & 0xFFFF;
  qil = qi & 0xFFFF;
  qih = (qi >> 16);
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"div_160_80#2: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:..:..\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d4.s0, res->d3.s0, res->d2.s0);
#endif

  /*******************************************************/

// nn = n * qi
  nn.d0  = mul24(n.d0, qil);
  t1     = mad24(n.d0, qih, nn.d0 >> 16);
  t2     = mad24(n.d1, qil, t1 & 0xFFFF);
  nn.d1  = t2 & 0xFFFF;
  nn.d0 &= 0xFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#2.1: nn=..:..:..:..:%x:%x:..:..\n",
        nn.d1.s0, nn.d0.s0);
#endif

  t1     = mad24(n.d1, qih, t1 >> 16);
  t2     = mad24(n.d2, qil, t2 >> 16);
  nn.d2  = (t1 & 0xFFFF) + (t2 & 0xFFFF);
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#2.2: nn=..:..:..:%x:%x:%x:..:..\n",
        nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  t1     = mad24(n.d2, qih, t1 >> 16);
  t2     = mad24(n.d3, qil, t2 >> 16);
  nn.d3  = (t1 & 0xFFFF) + (t2 & 0xFFFF) + (nn.d2 >> 16);
  nn.d2 &= 0xFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#2.3: nn=..:..:%x:%x:%x:%x:..:..\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  t1     = mad24(n.d3, qih, t1 >> 16);
  t2     = mad24(n.d4, qil, t2 >> 16);
  nn.d4  = (t1 & 0xFFFF) + (t2 & 0xFFFF) + (nn.d3 >> 16);
  nn.d3 &= 0xFFFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#2.4: nn=..:%x:%x:%x:%x:%x:..:..\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d5  = mad24(n.d4, qih, t1 >> 16) + (t2 >> 16) + (nn.d4 >> 16);
  nn.d4 &= 0xFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#2.5: nn=..:%x:%x:%x:%x:%x:%x:..:..\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

// now shift-left 11 bits
#ifdef CHECKS_MODBASECASE
  nn.d6  = nn.d5 >> 5;  // PERF: not needed as it will be gone anyway after sub
#endif
  nn.d5  = mad24(nn.d5 & 0x1F, 2048u, nn.d4 >> 5);
  nn.d4  = mad24(nn.d4 & 0x1F, 2048u, nn.d3 >> 5);
  nn.d3  = mad24(nn.d3 & 0x1F, 2048u, nn.d2 >> 5);
  nn.d2  = mad24(nn.d2 & 0x1F, 2048u, nn.d1 >> 5);
  nn.d1  = mad24(nn.d1 & 0x1F, 2048u, nn.d0 >> 5);
  nn.d0  = (nn.d0 & 0x1F) << 11;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#2.6: nn=..:%x:%x:%x:%x:%x:%x:%x:..:..\n",
        nn.d6.s0, nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

//  q = q - nn
  q.d2 = -nn.d0;
  q.d3 = -nn.d1 + AS_UINT_V((q.d2 > 0xFFFF));
  q.d4 = q.d4 - nn.d2 + AS_UINT_V((q.d3 > 0xFFFF));
  q.d5 = q.d5 - nn.d3 + AS_UINT_V((q.d4 > 0xFFFF));
  q.d6 = q.d6 - nn.d4 + AS_UINT_V((q.d5 > 0xFFFF));
  q.d7 = q.d7 - nn.d5 + AS_UINT_V((q.d6 > 0xFFFF));
#ifdef CHECKS_MODBASECASE
  q.d8 = q.d8 - nn.d6 + AS_UINT_V((q.d7 > 0xFFFF)); // PERF: not needed: should be zero anyway
#endif
  q.d2 &= 0xFFFF;
  q.d3 &= 0xFFFF;
  q.d4 &= 0xFFFF;
  q.d5 &= 0xFFFF;
  q.d6 &= 0xFFFF;
  q.d7 &= 0xFFFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#2.7: q=..:%x!%x:%x:%x:%x:%x:%x:..:..\n",
        q.d8.s0, q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0);
#endif

  MODBASECASE_NONZERO_ERROR(q.d8, 2, 8, 3);

  /********** Step 3, Offset 2^20 (1*16 + 5) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d7, 65536u, q.d6));
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(mad24(q.d5, 65536u, q.d4));
  qf*= 65536.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<26, 3, qi, 5);  // very big qi, but then we can skip the bit-shifting later

  qih = (qi >> 16);
  qil = qi & 0xFFFF;
  res->d1  = qi;  // carry to d2 is handled at the end anyway
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"div_160_80#3: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:..\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d4.s0, res->d3.s0, res->d2.s0, res->d1.s0);
#endif

  /*******************************************************/

// nn = n * qi
  nn.d0  = mul24(n.d0, qil);
  t1     = mad24(n.d0, qih, nn.d0 >> 16);
  t2     = mad24(n.d1, qil, t1 & 0xFFFF);
  nn.d1  = t2 & 0xFFFF;
  nn.d0 &= 0xFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#3.1: nn=..:..:..:..:%x:%x:..:..\n",
        nn.d1.s0, nn.d0.s0);
#endif

  t1     = mad24(n.d1, qih, t1 >> 16);
  t2     = mad24(n.d2, qil, t2 >> 16);
  nn.d2  = (t1 & 0xFFFF) + (t2 & 0xFFFF);
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#3.2: nn=..:..:..:%x:%x:%x:..:..\n",
        nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  t1     = mad24(n.d2, qih, t1 >> 16);
  t2     = mad24(n.d3, qil, t2 >> 16);
  nn.d3  = (t1 & 0xFFFF) + (t2 & 0xFFFF) + (nn.d2 >> 16);
  nn.d2 &= 0xFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#3.3: nn=..:..:%x:%x:%x:%x:..:..\n",
        nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  t1     = mad24(n.d3, qih, t1 >> 16);
  t2     = mad24(n.d4, qil, t2 >> 16);
  nn.d4  = (t1 & 0xFFFF) + (t2 & 0xFFFF) + (nn.d3 >> 16);
  nn.d3 &= 0xFFFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#3.4: nn=..:%x:%x:%x:%x:%x:..:..\n",
        nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

  nn.d5  = mad24(n.d4, qih, t1 >> 16) + (t2 >> 16) + (nn.d4 >> 16);
  nn.d4 &= 0xFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#3.5: nn=..:%x:%x:%x:%x:%x:%x:..:..\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

#ifdef CHECKS_MODBASECASE
  nn.d6  = nn.d5 >> 16;  // PERF: not needed as it will be gone anyway after sub
  nn.d5 &= 0xFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#3.6: nn=..:%x:%x:%x:%x:%x:%x:..:..\n",
        nn.d5.s0, nn.d4.s0, nn.d3.s0, nn.d2.s0, nn.d1.s0, nn.d0.s0);
#endif

//  q = q - nn
  q.d1 = -nn.d0;
  q.d2 = q.d2 - nn.d1 + AS_UINT_V((q.d1 > 0xFFFF));
  q.d3 = q.d3 - nn.d2 + AS_UINT_V((q.d2 > 0xFFFF));
  q.d4 = q.d4 - nn.d3 + AS_UINT_V((q.d3 > 0xFFFF));
  q.d5 = q.d5 - nn.d4 + AS_UINT_V((q.d4 > 0xFFFF));
  q.d6 = q.d6 - nn.d5 + AS_UINT_V((q.d5 > 0xFFFF));
#ifdef CHECKS_MODBASECASE
  q.d7 = q.d7 - nn.d6 + AS_UINT_V((q.d6 > 0xFFFF)); // PERF: not needed: should be zero anyway
  q.d7 &= 0xFFFF;
#endif
  q.d1 &= 0xFFFF;
  q.d2 &= 0xFFFF;
  q.d3 &= 0xFFFF;
  q.d4 &= 0xFFFF;
  q.d5 &= 0xFFFF;
  q.d6 &= 0xFFFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#3.7: q=..:%x:%x!%x:%x:%x:%x:%x:%x:..\n",
        q.d8.s0, q.d7.s0, q.d6.s0, q.d5.s0, q.d4.s0, q.d3.s0, q.d2.s0, q.d1.s0);
#endif

  MODBASECASE_NONZERO_ERROR(q.d7, 3, 7, 6);

  /********** Step 4, Offset 2^0 (0*16 + 0) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d6, 65536u, q.d5));
  qf= qf * 4294967296.0f + CONVERT_FLOAT_V(mad24(q.d4, 65536u, q.d3));
  qf*= 65536.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 7);

  qil = qi & 0xFFFF;
  qih = (qi >> 16);
  res->d1 += qih;
  res->d0 = qil;

  // skip the last part - it will change the result by one at most - we can live with a result that is off by one
  // but need to handle outstanding carries instead
  res->d2 += res->d1 >> 16;
  res->d1 &= 0xFFFF;
  res->d3 += res->d2 >> 16;
  res->d2 &= 0xFFFF;
  res->d4 += res->d3 >> 16;
  res->d3 &= 0xFFFF;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"div_160_80#4: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:%x\n",
                                 qf.s0, nf.s0, qf.s0*nf.s0, qi.s0, qi.s0, res->d4.s0, res->d3.s0, res->d2.s0, res->d1.s0, res->d0.s0);
#endif

}

/****
 * the trial factoring implementations for 5x16 bit
 * bit_max65 is bit_max - 65
 ****/

void check_barrett16_78(uint shifter, const int80_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
     MODBASECASE_PAR_DEF)
{
  __private int80_v a, u;
  __private int160_v b, tmp160;
  __private int80_v tmp80;
  __private float_v ff;
  __private uint bit_max_80=16-bit_max65; //bit_max is 65 .. 78
  __private uint tmp, bit_max80_mult = 1 << bit_max_80; /* used for bit shifting... */
  __private int160_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};

/*
ff = 1/f as float, needed in div_192_96().
*/
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d4, 65536u, f.d3));
  ff= ff * 65536.0f + CONVERT_FLOAT_RTP_V(f.d2);   // these are at least 30 significant bits for 60-bit FC's

  ff= as_float(0x3f7ffffc) / ff;

  tmp = 1 << bit_max65;	// tmp160 = 2^(80 + bits in f)

  // tmp160.d0 .. d8 =0
  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper part (16 bits) of a 160-bit value. The lower 144 bits are all zero implicitely

  div_160_80(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR
);						// u = floor(tmp160 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett16_78: u=%x:%x:%x:%x:%x, ff=%G, bit_max80_mult=%#x, bit_max65=%u\n",
        u.d4.s0, u.d3.s0, u.d2.s0, u.d1.s0, u.d0.s0, ff.s0, bit_max80_mult, bit_max65);
#endif
#if (TRACE_KERNEL > 12)
    mul_80_160_no_low3(&tmp160, u, f);
    if (tid==TRACE_TID) printf((__constant char *)"vrfy: u*f=%x:%x:%x:%x:%x:%x:%x:..:..:..\n",
        tmp160.d9.s0, tmp160.d8.s0, tmp160.d7.s0, tmp160.d6.s0, tmp160.d5.s0, tmp160.d4.s0, tmp160.d3.s0);
#endif
  a.d0 = mad24(bb.d5, bit_max80_mult, (bb.d4 >> bit_max65))&0xFFFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bb.d6, bit_max80_mult, (bb.d5 >> bit_max65))&0xFFFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bb.d7, bit_max80_mult, (bb.d6 >> bit_max65))&0xFFFF;			// a = b / (2^bit_max)
  a.d3 = mad24(bb.d8, bit_max80_mult, (bb.d7 >> bit_max65))&0xFFFF;			// a = b / (2^bit_max)
  a.d4 = mad24(bb.d9, bit_max80_mult, (bb.d8 >> bit_max65));		        	// a = b / (2^bit_max)

  mul_80_160_no_low5(&tmp160, a, u);					// tmp160 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett16_78: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:...\n",
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        tmp160.d9.s0, tmp160.d8.s0, tmp160.d7.s0, tmp160.d6.s0, tmp160.d5.s0, tmp160.d4.s0);
#endif

  a.d0 = tmp160.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = tmp160.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d2 = tmp160.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d3 = tmp160.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d4 = tmp160.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

  mul_80(&tmp80, a, f);							// tmp80 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett16_78: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0, tmp80.d4.s0, tmp80.d3.s0, tmp80.d2.s0, tmp80.d1.s0, tmp80.d0.s0);
#endif
    // all those bb's are 0 due to preprocessing on the host, thus always require a borrow
  a.d0 = (-tmp80.d0) & 0xFFFF;
  a.d1 = (-tmp80.d1 + AS_UINT_V((a.d0 > 0)  ));
  a.d2 = (-tmp80.d2 + AS_UINT_V((a.d1 > 0xFFFF)  ));
  a.d3 = (-tmp80.d3 + AS_UINT_V((a.d2 > 0xFFFF)  ));
  a.d4 = (bb.d4-tmp80.d4 + AS_UINT_V((a.d3 > 0xFFFF)  )) & 0xFFFF;
  a.d1 &= 0xFFFF;
  a.d2 &= 0xFFFF;
  a.d3 &= 0xFFFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett16_78: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif

  while(shifter)
  {
    square_80_160(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        b.d9.s0, b.d8.s0, b.d7.s0, b.d6.s0, b.d5.s0, b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0 );
#endif
#if (TRACE_KERNEL > 14)
    // verify squaring by dividing again.
    __private float_v f1 = CONVERT_FLOAT_RTP_V(mad24(a.d4, 65536, a.d3));
    f1= f1 * 65536.0f + CONVERT_FLOAT_RTP_V(a.d2);   // f.d1 needed?

    f1= as_float(0x3f7ffffc) / f1;
    div_160_80(&tmp80, b.d9.s0, a, f1, tid
               MODBASECASE_PAR
              );
    if (tid==TRACE_TID) printf((__constant char *)"vrfy: b = %x:<144x0> / a=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x\n",
        b.d9.s0,
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0, tmp80.d4.s0, tmp80.d3.s0, tmp80.d2.s0, tmp80.d1.s0, tmp80.d0.s0);
#endif
    a.d0 = mad24(b.d5, bit_max80_mult, (b.d4 >> bit_max65))&0xFFFF;			// a = b / (2^bit_max)
    a.d1 = mad24(b.d6, bit_max80_mult, (b.d5 >> bit_max65))&0xFFFF;			// a = b / (2^bit_max)
    a.d2 = mad24(b.d7, bit_max80_mult, (b.d6 >> bit_max65))&0xFFFF;			// a = b / (2^bit_max)
    a.d3 = mad24(b.d8, bit_max80_mult, (b.d7 >> bit_max65))&0xFFFF;			// a = b / (2^bit_max)
    a.d4 = mad24(b.d9, bit_max80_mult, (b.d8 >> bit_max65));       			// a = b / (2^bit_max)

    mul_80_160_no_low5(&tmp160, a, u);					// tmp160 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:...\n",
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0,
        tmp160.d9.s0, tmp160.d8.s0, tmp160.d7.s0, tmp160.d6.s0, tmp160.d5.s0, tmp160.d4.s0);
#endif
    a.d0 = tmp160.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = tmp160.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d2 = tmp160.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d3 = tmp160.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d4 = tmp160.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    mul_80(&tmp80, a, f);						// tmp80 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0, tmp80.d4.s0, tmp80.d3.s0, tmp80.d2.s0, tmp80.d1.s0, tmp80.d0.s0);
#endif
    a.d0 = (b.d0 - tmp80.d0) & 0xFFFF;
    a.d1 = (b.d1 - tmp80.d1 + AS_UINT_V((a.d0 > b.d0)  ));
    a.d2 = (b.d2 - tmp80.d2 + AS_UINT_V((a.d1 > 0xFFFF)  ));
    a.d3 = (b.d3 - tmp80.d3 + AS_UINT_V((a.d2 > 0xFFFF)  ));
    a.d4 = (b.d4 - tmp80.d4 + AS_UINT_V((a.d3 > 0xFFFF)  ));
    a.d1 &= 0xFFFF;
    a.d2 &= 0xFFFF;
    a.d3 &= 0xFFFF;
    a.d4 &= 0xFFFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        b.d4.s0, b.d3.s0, b.d2.s0, b.d1.s0, b.d0.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
    if(shifter&0x80000000)shl_80(&a);

    shifter+=shifter;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, a=%x:%x:%x:%x:%x\n",
        shifter, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
  }

  mod_simple_even_80_and_check_big_factor80(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , bit_max_80, 10, modbasecase_debug
#endif
                       );
}

/******
 * now the actual kernels for 5x16 bit calculations
 *
 * shiftcount is used for precomputing without mod
 * b_in is precomputed on host ONCE.
  ******/

#ifndef CL_GPU_SIEVE

__kernel void cl_barrett16_78(__private uint exponent, const int80_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                           const uint8 b_in, __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF         )
{
  __private int80_v f;
  __private uint tid;

	tid = get_global_id(0) * VECTOR_SIZE;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett16_78: b_in=%x:%x:%x:%x:%x:%x:0:0:0:0, shift=%d, bit_max65=%d\n",
        b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, shiftcount, bit_max65);
#endif

  calculate_FC80(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett16_78: f=%x:%x:%x:%x:%x\n",
        f.d4.s0, f.d3.s0, f.d2.s0, f.d1.s0, f.d0.s0);
#endif

  check_barrett16_78(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}


#else
/****************************************
 ****************************************
 * 16-bit-kernel consuming the GPU sieve
 * included by main kernel file
 ****************************************
 ****************************************/


__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett16_78_gs(const uint exponent, const int80_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int80_v  k, f;
  __private uint     tid=get_global_id(0), lid=get_local_id(0);
  __private int80_t exp80;

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett16_78_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp80.d2=exponent>>29;exp80.d1=(exponent>>14)&0xFFFF;exp80.d0=(exponent<<1)&0xFFFF;	// exp80 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett16_78_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
#endif

  for (i = lid*VECTOR_SIZE; i < total_bit_count; i += 256*VECTOR_SIZE) // VECTOR_SIZE*THREADS_PER_BLOCK
  {
    // if i == total_bit_count-1, then we may read up to VECTOR_SIZE-1 elements beyond the array (uninitialized).
    // this can result in the same factor being reported up to VECTOR_SIZE times.

    uint_v k_delta, t1, t2;

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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0xFFFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 16) + mad24(NUM_CLASSES, k_delta >> 16, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 16) + k_base.d2;
    k.d3 = (k.d2 >> 16) + k_base.d3;

    k.d0 &= 0xFFFF;
    k.d1 &= 0xFFFF;
    k.d2 &= 0xFFFF;

    f.d0 = mad24(k.d0, exp80.d0, 1u);  // exp80 = 2*exponent ==> f = 2kp+1

    t1   = mad24(k.d1, exp80.d0, f.d0 >> 16);
    t2   = mad24(k.d0, exp80.d1, t1 & 0xFFFF);
    f.d1 = t2 & 0xFFFF;
    f.d0 &= 0xFFFF;

    t1   = mad24(k.d2, exp80.d0, t1 >> 16);
    t2   = mad24(k.d1, exp80.d1, t2 >> 16);
    f.d2 = mad24(k.d0, exp80.d2, t1 & 0xFFFF) + (t2 & 0xFFFF);  // exp80.d2 is 0 or 1: the multiplication result can be 16 bit max.

    t1   = mad24(k.d3, exp80.d0, t1 >> 16);
    t2   = mad24(k.d2, exp80.d1, t2 >> 16);
    f.d3 = mad24(k.d1, exp80.d2, f.d2 >> 16) + (t1 & 0xFFFF) + (t2 & 0xFFFF);
    f.d2 &= 0xFFFF;

    f.d4 = mad24(k.d3, exp80.d1, t1 >> 16);
    f.d4 = mad24(k.d2, exp80.d2, f.d3 >> 16) + (t2 >> 16) + f.d4;
    f.d3 &= 0xFFFF;

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett16_78_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x, f=%x:%x:%x:%x:%x\n",
        i, smem[i], k_delta.s0, k.d3.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d4.s0, f.d3.s0, f.d2.s0, f.d1.s0, f.d0.s0);
#endif

    check_barrett16_78(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

#endif
