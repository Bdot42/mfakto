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
 * 15-bit based 75-bit barrett-kernels
 *
 ****************************************
 ****************************************/

// function prototypes

int75_v sub_if_gte_75(const int75_v a, const int75_v b);

void mul_75(int75_v * const res, const int75_v a, const int75_v b);

void mul_75_big(int75_v * const res, const int75_v a, const int75_v b);

void mul_75_150_no_low3(int150_v * const res, const int75_v a, const int75_v b);

void mul_75_150_no_low5(int150_v * const res, const int75_v a, const int75_v b);

void mul_75_150_no_low5_big(int150_v * const res, const int75_v a, const int75_v b);

void mul_75_150(int150_v * const res, const int75_v a, const int75_v b);

void square_75_150(int150_v * const res, const int75_v a);

void square_75_150_big(int150_v * const res, const int75_v a);

void shl_75(int75_v * const a);

void shl_150(int150_v * const a);

void div_150_75(int75_v * const res, const uint qhi, const int75_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
                  MODBASECASE_PAR_DEF
);

void check_barrett15_69(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
     MODBASECASE_PAR_DEF);

void check_barrett15_70(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
     MODBASECASE_PAR_DEF);

void check_barrett15_71(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
     MODBASECASE_PAR_DEF);

void check_barrett15_73(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
     MODBASECASE_PAR_DEF);

void check_barrett15_74(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
     MODBASECASE_PAR_DEF);

int90_v sub_if_gte_90(const int90_v a, const int90_v b);

void mul_90(int90_v * const res, const int90_v a, const int90_v b);

void mul_90_180_no_low3(int180_v * const res, const int90_v a, const int90_v b);

void mul_90_180_no_low5(int180_v * const res, const int90_v a, const int90_v b);

void mul_90_180(int180_v * const res, const int90_v a, const int90_v b);

void square_90_180(int180_v * const res, const int90_v a);

void shl_90(int90_v * const a);

void shl_180(int180_v * const a);

void div_180_90(int90_v * const res, const uint qhi, const int90_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
                  MODBASECASE_PAR_DEF
);

void check_barrett15_82(uint shifter, const int90_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);

void check_barrett15_83(uint shifter, const int90_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);

void check_barrett15_88(uint shifter, const int90_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF);

// end prototypes

int75_v sub_if_gte_75(const int75_v a, const int75_v b)
/* return (a>b)?a-b:a */
{
  int75_v tmp;
  /* do the subtraction and use tmp.d4 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0);
  tmp.d1 = (a.d1 - b.d1 + AS_UINT_V((tmp.d0 > 0x7FFF) ));
  tmp.d2 = (a.d2 - b.d2 + AS_UINT_V((tmp.d1 > 0x7FFF) ));
  tmp.d3 = (a.d3 - b.d3 + AS_UINT_V((tmp.d2 > 0x7FFF) ));
  tmp.d4 = (a.d4 - b.d4 + AS_UINT_V((tmp.d3 > 0x7FFF) ));
  tmp.d0&= 0x7FFF;
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

void mul_75(int75_v * const res, const int75_v a, const int75_v b)
/* res = a * b (low 75 bits)
  15x mul24/mad24, 4x >>, 4x &   = 23 ops  */
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

  res->d4 = mad24(a.d4, b.d0, res->d3 >> 15);  // if a.d4 is > 15 bits, then overflow can happen faster.
  res->d4 = mad24(a.d3, b.d1, res->d4);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
  res->d4 = mad24(a.d0, b.d4, res->d4);  // the 5th mad can overflow d4, but that's ok for this function.
  res->d3 &= 0x7FFF;
//  res->d4 &= 0x7FFF;
}

void mul_75_big(int75_v * const res, const int75_v a, const int75_v b)
/* res = a * b (low 75 bits)
  19x mul24/mad24, 4x >>, 4x &   = 27 ops  */
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

  // in order to get one more bit in the result, we need to add the next bigger component into each multiplicant
  res->d4 = mad24(a.d4, b.d0, res->d3 >> 15);
  res->d4 = mad24(mad24(a.d4, 32768u, a.d3), b.d1, res->d4);
  res->d4 = mad24(mad24(a.d3, 32768u, a.d2), b.d2, res->d4);
  res->d4 = mad24(mad24(a.d2, 32768u, a.d1), b.d3, res->d4);
  res->d4 = mad24(mad24(a.d1, 32768u, a.d0), b.d4, res->d4);
  res->d3 &= 0x7FFF;
//  res->d4 &= 0xFFFF;
}


void mul_75_150_no_low3(int150_v * const res, const int75_v a, const int75_v b)
/*
res ~= a * b
res.d0 to res.d2 are NOT computed. Carries to res.d3 are ignored,
too. So the digits res.d{3-9} might differ from mul_75_150().
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

void mul_75_150_no_low5(int150_v * const res, const int75_v a, const int75_v b)
/*
res ~= a * b
res.d0 to res.d3 are NOT computed. res.d4 is computed only to get its upper half carried to res.d5.
Due to the missing carries from d3 into d4, at most a 17-bit value is missing in d4. This means,
d4 >> 15 can be too low by up to 3, thus d5 can be low by 3.
 */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // this optimized mul 5x5 requires: 19 mul/mad24, 7 shift, 6 and, 1 add


  res->d4 = mul24(a.d4, b.d0);
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
  // res->d4 &= 0x7FFF;  // not needed, we won't use d4 anyway
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

void mul_75_150_no_low5_big(int150_v * const res, const int75_v a, const int75_v b)
/*
res ~= a * b
res.d0 to res.d3 are NOT computed. res.d4 is computed only to get its upper half carried to res.d5.
Due to the missing carries from d3 into d4, at most a 17-bit value is missing in d4. This means,
d4 >> 15 can be too low by up to 3, thus d5 can be low by 3.
This version allows for a "big" a, meaning, a.d4 can be up to 17 bits.
 */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // this optimized mul 5x5 requires: 19 mul/mad24, 7 shift, 6 and, 1 add

  // mad24(a.d4, ...) will already return 32 bit. Handle a.d4 first all the way through res->d8

  res->d4 = mul24(a.d4, b.d0);
  res->d5 = mad24(a.d4, b.d1, res->d4 >> 15);
  res->d4 &= 0x7FFF;
  res->d6 = mad24(a.d4, b.d2, res->d5 >> 15);
  res->d5 &= 0x7FFF;
  res->d7 = mad24(a.d4, b.d3, res->d6 >> 15);
  res->d6 &= 0x7FFF;
  res->d8 = mad24(a.d4, b.d4, res->d7 >> 15);
  res->d7 &= 0x7FFF;
  res->d9 = res->d8 >> 15;
  res->d8 &= 0x7FFF;

  res->d4 = mad24(a.d3, b.d1, res->d4);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
  res->d4 = mad24(a.d0, b.d4, res->d4);

  res->d5 = mad24(a.d3, b.d2, res->d5) + (res->d4 >> 15);
  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d5 = mad24(a.d1, b.d4, res->d5);
  // res->d4 &= 0x7FFF;  // not needed, we won't use d4 anyway

  res->d6 = mad24(a.d3, b.d3, res->d6) + (res->d5 >> 15);
  res->d6 = mad24(a.d2, b.d4, res->d6);
  res->d5 &= 0x7FFF;

  res->d7 = mad24(a.d3, b.d4, res->d7) + (res->d6 >> 15);
  res->d6 &= 0x7FFF;

  res->d8 += res->d7 >> 15;
  res->d7 &= 0x7FFF;

  res->d9 += res->d8 >> 15;
  res->d8 &= 0x7FFF;
}


void mul_75_150(int150_v * const res, const int75_v a, const int75_v b)
/*
res = a * b
 */
{
  /* this is the complete implementation, no longer used, but was the basis for
     the _no_low3 and square functions */
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
/* res = a^2 = d0^2 + 2d0d1 + d1^2 + 2d0d2 + 2(d1d2 + d0d3) + d2^2 +
               2(d0d4 + d1d3) + 2(d1d4 + d2d3) + d3^2 + 2d2d4 + 2d3d4 + d4^2
   */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // square 5x5 requires: 15 mul/mad24, 20 shift, 10 and, 1 add

  res->d0 = mul24(a.d0, a.d0);

  res->d1 = mad24(a.d1, a.d0 << 1, res->d0 >> 15);
  res->d0 &= 0x7FFF;

  res->d2 = mad24(a.d1, a.d1, res->d1 >> 15);
  res->d2 = mad24(a.d2, a.d0 << 1, res->d2);
  res->d1 &= 0x7FFF;

  res->d3 = mad24(a.d3, a.d0 << 1, res->d2 >> 15);
  res->d3 = mad24(a.d2, a.d1 << 1, res->d3);
  res->d2 &= 0x7FFF;

  res->d4 = mad24(a.d4, a.d0 << 1, res->d3 >> 15);
  res->d3 &= 0x7FFF;
  res->d4 = mad24(a.d3, a.d1 << 1, res->d4);
   // 5th mad24 can overflow d4, need to handle carry before: pull in the first d5 line
  res->d5 = mad24(a.d4, a.d1 << 1, res->d4 >> 15);
  res->d4 &= 0x7FFF;
  res->d4 = mad24(a.d2, a.d2, res->d4);  // 31-bit at most

  res->d5 = mad24(a.d3, a.d2 << 1, res->d4 >> 15) + res->d5;
  res->d4 &= 0x7FFF;
  // now we have in d5: 4x mad24() + 1x 17-bit carry + 1x 16-bit carry: still fits into 32 bits

  res->d6 = mad24(a.d4, a.d2 << 1, res->d5 >> 15);
  res->d6 = mad24(a.d3, a.d3, res->d6);
  res->d5 &= 0x7FFF;

  res->d7 = mad24(a.d4, a.d3 << 1, res->d6 >> 15);
  res->d6 &= 0x7FFF;

  res->d8 = mad24(a.d4, a.d4, res->d7 >> 15);
  res->d7 &= 0x7FFF;

  res->d9 = res->d8 >> 15;
  res->d8 &= 0x7FFF;
}

void square_75_150_big(int150_v * const res, const int75_v a)
/* res = a^2 = d0^2 + 2d0d1 + d1^2 + 2d0d2 + 2(d1d2 + d0d3) + d2^2 +
               2(d0d4 + d1d3) + 2(d1d4 + d2d3) + d3^2 + 2d2d4 + 2d3d4 + d4^2
  "big" because a.d4 can have 16 bits.
   */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // square 5x5 requires: 15 mul/mad24, 20 shift, 10 and, 1 add
  // for 16-bit: 0xFFFF * 0xFFFF = 0xFFFE0001, can add 2 15-bit carries
  // square 5x5 big requires: 15 mul/mad24, 22 shift, 12 and, 3 add  (+6 ops for "big")

  res->d0 = mul24(a.d0, a.d0); // max: 0x3FFF0001

  res->d1 = mad24(a.d1, a.d0 << 1, res->d0 >> 15); // max: 0x7FFF * 0xFFFE + 0x7FFE = 0x7FFE8000
  res->d0 &= 0x7FFF;

  res->d2 = mad24(a.d1, a.d1, res->d1 >> 15); // max: 0x3FFFFFFE
  res->d2 = mad24(a.d2, a.d0 << 1, res->d2);  // max: 0xBFFE0000
  res->d1 &= 0x7FFF;

  res->d3 = mad24(a.d3, a.d0 << 1, res->d2 >> 15); // max: 0x7FFF7FFE
  res->d3 = mad24(a.d2, a.d1 << 1, res->d3);       // max: 0xFFFD8000
  res->d2 &= 0x7FFF;

  res->d4 = mad24(a.d4, a.d0 << 1, res->d3 >> 15); // max: 0xFFFF * 0xFFFE + 0x1FFFB = 0xFFFEFFFD
  res->d3 &= 0x7FFF;                               // need to propagate the carry to the top
  res->d5 = mad24(a.d4, a.d1 << 1, res->d4 >> 15); // max: 0xFFFF * 0xFFFE + 0x1FFFD = 0xFFFEFFFF
  res->d4 &= 0x7FFF;
  res->d6 = mad24(a.d4, a.d2 << 1, res->d5 >> 15); // max: 0xFFFF * 0xFFFE + 0x1FFFD = 0xFFFEFFFF
  res->d5 &= 0x7FFF;
  res->d7 = mad24(a.d4, a.d3 << 1, res->d6 >> 15); // max: 0xFFFF * 0xFFFE + 0x1FFFD = 0xFFFEFFFF
  res->d6 &= 0x7FFF;



  res->d4 = mad24(a.d2, a.d2, res->d4);
  res->d4 = mad24(a.d3, a.d1 << 1, res->d4);  // max: 0x7FFF * 0x7FFF + 0x7FFF * 0xFFFE + 0x7FFF = 0xBFFD8002

  res->d5 = mad24(a.d3, a.d2 << 1, res->d5) + (res->d4 >> 15); // max: 0x7FFFFFFC
  res->d4 &= 0x7FFF;

  res->d6 = mad24(a.d3, a.d3, res->d6) + (res->d5 >> 15); // max: 0x40007FFF
  res->d5 &= 0x7FFF;

  res->d7 += (res->d6 >> 15); // max: 0xFFFEFFFF + 0x8000 = 0xFFFF7FFF
  res->d6 &= 0x7FFF;

  res->d8 = mad24(a.d4, a.d4, res->d7 >> 15); // max: 0xFFFF * 0xFFFF + 0x1FFFE = 0xFFFFFFFF
  res->d7 &= 0x7FFF;

  res->d9 = res->d8 >> 15; // max: 0x1FFFF
  res->d8 &= 0x7FFF;
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

void shl_150(int150_v * const a)
/* shiftleft a one bit */
{
  a->d9 = mad24(a->d9, 2u, a->d8 >> 14); // keep the extra top bit
  a->d8 = mad24(a->d8, 2u, a->d7 >> 14) & 0x7FFF;
  a->d7 = mad24(a->d7, 2u, a->d6 >> 14) & 0x7FFF;
  a->d6 = mad24(a->d6, 2u, a->d5 >> 14) & 0x7FFF;
  a->d5 = mad24(a->d5, 2u, a->d4 >> 14) & 0x7FFF;
  a->d4 = mad24(a->d4, 2u, a->d3 >> 14) & 0x7FFF;
  a->d3 = mad24(a->d3, 2u, a->d2 >> 14) & 0x7FFF;
  a->d2 = mad24(a->d2, 2u, a->d1 >> 14) & 0x7FFF;
  a->d1 = mad24(a->d1, 2u, a->d0 >> 14) & 0x7FFF;
  a->d0 = (a->d0 << 1u) & 0x7FFF;
}


#if defined USE_DP
void div_150_75_d(int75_v * const res, const uint qhi, const int75_v n, const double_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
                  MODBASECASE_PAR_DEF
)
/* res = q / n (integer division)
    during function entry, qhi contains the upper 30 bits of an 150-bit-value. The remaining bits are zero implicitly.
    this is not a vector, as the first value is the same for all FCs*/
    // do 2*45 bit reductions using double: should be sufficient for 90 bits (and 75 anyways)
{
  __private double_v qf;
  __private double   qf_1;   // for the first conversion which does not need vectors yet

  __private ulong_v qi;
  __private uint_v qil, qim, qih;
  __private int150_v nn, q;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#0: q=%x:<120x0>, n=%x:%x:%x:%x:%x, nf=%#G\n",
        qhi, V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(nf));
#endif

/********** Step 1, Offset 2^67 (4*15 + 7) **********/
  qf_1 = convert_double(qhi) * 40564819207303340847894502572032.0;

  qi=CONVERT_ULONG_V(qf_1*nf);  // vectorize just here

  MODBASECASE_QI_ERROR(1L<<46, 1, qi, 0);  // qi here is about 45 bits

  qih = res->d4 = CONVERT_UINT_V(qi >> 30);  // PERF: amd_bitalign ?
  qim = res->d3 = (CONVERT_UINT_V(qi) >> 15) & 0x7FFF;
  qil = res->d2 = CONVERT_UINT_V(qi      ) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1: qf=%#G, nf=%#G, *=%#G, qi=%lld=0x%llx, res=%x:%x:%x:..:..\n",
                                 qf_1, V(nf), qf_1*V(nf), V(qi), V(qi), V(res->d4), V(res->d3), V(res->d2));
#endif

  /*******************************************************/

// nn = n * qi
  nn.d2  = mul24(n.d0, qil);
  nn.d3  = mad24(n.d0, qim, nn.d2 >> 15);
  nn.d3  = mad24(n.d1, qil, nn.d3);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1.1: nn=..:..:..:..:..:..:%x:%x:..:..\n",
        V(nn.d3), V(nn.d2));
#endif

  nn.d4  = mad24(n.d0, qih, nn.d3 >> 15);
  nn.d4  = mad24(n.d1, qim, nn.d4);
  nn.d4  = mad24(n.d2, qil, nn.d4);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1.2: nn=..:..:..:..:..:%x:%x:%x:...\n",
        V(nn.d4), V(nn.d3), V(nn.d2));
#endif

  nn.d5  = mad24(n.d1, qih, nn.d4 >> 15);
  nn.d5  = mad24(n.d2, qim, nn.d5);
  nn.d5  = mad24(n.d3, qil, nn.d5);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1.3: nn=..:..:..:..:%x:%x:%x:%x:...\n",
        V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2));
#endif

  nn.d6  = mad24(n.d2, qih, nn.d5 >> 15);
  nn.d6  = mad24(n.d3, qim, nn.d6);
  nn.d6  = mad24(n.d4, qil, nn.d6);
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1.4: nn=..:..:..:%x:%x:%x:%x:%x:...\n",
        V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2));
#endif

  nn.d7  = mad24(n.d3, qih, nn.d6 >> 15);
  nn.d7  = mad24(n.d4, qim, nn.d7);
  nn.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1.5: nn=..:..:%x:%x:%x:%x:%x:%x:...\n",
        V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2));
#endif

  nn.d8  = mad24(n.d4, qih, nn.d7 >> 15);
  nn.d7 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1.6: nn=..:%x:%x:%x:%x:%x:%x:%x:...\n",
        V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2));
#endif
  nn.d9  = nn.d8 >> 15;
  nn.d8 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1.7: nn=..:%x:%x:%x:%x:%x:%x:%x:%x:...\n",
        V(nn.d9), V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2));
#endif

//  q = q - nn, but upon function entry, qhi contains all the bits for d9. All bits below are zero.
  q.d2 = (-nn.d2) & 0x7FFF;
  q.d3 = (-nn.d3 + AS_UINT_V((nn.d2 > 0)));
  q.d4 = (-nn.d4 + AS_UINT_V((q.d3 > 0)));
  q.d5 = (-nn.d5 + AS_UINT_V((q.d4 > 0)));
  q.d6 = (-nn.d6 + AS_UINT_V((q.d5 > 0)));
  q.d7 = (-nn.d7 + AS_UINT_V((q.d6 > 0)));
  q.d8 = (-nn.d8 + AS_UINT_V((q.d7 > 0)));
  q.d9 = (qhi - nn.d9 + AS_UINT_V((q.d8 > 0)));

  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
  q.d7 &= 0x7FFF;
  q.d8 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#1.8: q=..:%x:%x!%x:%x:%x:%x:%x:%x:..:..\n",
        V(q.d9), V(q.d8), V(q.d7), V(q.d6), V(q.d5), V(q.d4), V(q.d3), V(q.d2));
#endif
  MODBASECASE_NONZERO_ERROR(q.d9, 2, 10, 1);
  MODBASECASE_NONZERO_ERROR(q.d8, 2, 9, 2);

  /********** Step 2, Offset 2^30 (2*15 + 0) **********/

  qf= CONVERT_DOUBLE_V(mad24(q.d7, 32768u, q.d6));
  qf= qf * 1073741824.0f + CONVERT_DOUBLE_V(mad24(q.d5, 32768u, q.d4)); // now we need only 30 bits
//  qf= qf * 1073741824.0f + CONVERT_DOUBLE_V(mad24(q.d4, 32768u, q.d3));
//  qf*= 35184372088832.0;
  qf*= 1152921504606846976.0;

  qih=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1L<<30, 2, qih, 3);

//  res->d2  = CONVERT_UINT_V(qi >> 30);
  res->d1  = qih >> 15;
  res->d0  = qih & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_150_75_d#2: qf=%#G, nf=%#G, *=%#G, qi=%lld=0x%llx, res=%x:%x:%x:%x:%x\n",
                                 V(qf), V(nf), V(qf)*V(nf), V(qih), V(qih), V(res->d4), V(res->d3), V(res->d2), V(res->d1), V(res->d0));
#endif

  /*******************************************************/
  // skip the last part - it will change the result by one at most - we can live with a result that is off by one
}
#endif

void div_150_75(int75_v * const res, const uint qhi, const int75_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
                  MODBASECASE_PAR_DEF
)
/* res = q / n (integer division) */
{
  __private float_v qf;
  __private float   qf_1;   // for the first conversion which does not need vectors yet
  __private uint_v qi, qil, qih;
  __private int150_v nn, q;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#0: q=%x:<135x0>, n=%x:%x:%x:%x:%x, nf=%#G\n",
        qhi, V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(nf));
#endif

/********** Step 1, Offset 2^60 (4*15 + 0) **********/
  qf_1= convert_float(qhi) * 35184372088832.0f; // =32768.0f * 32768.0f * 32768.0f; // no vector yet

  qi=CONVERT_UINT_V(qf_1*nf);  // vectorize just here

  MODBASECASE_QI_ERROR(1<<16, 1, qi, 0);  // first step is smaller, but 74 kernel needs 16 bits here

  res->d4 = qi;
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"div_150_75#1: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:..:..:..:..\n",
                                 qf_1, V(nf), qf_1*V(nf), V(qi), V(qi), V(res->d4));
  q.d9=0; // for correct printing later
#endif

  /*******************************************************/

// nn = n * qi
  nn.d0  = mul24(n.d0, qi);
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#1.1: nn=..:..:..:..:..:%x:..:..:..:..\n",
        V(nn.d0));
#endif

  nn.d1  = mad24(n.d1, qi, nn.d0 >> 15);
  nn.d0 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#1.2: nn=..:..:..:..:%x:%x:...\n",
        V(nn.d1), V(nn.d0));
#endif

  nn.d2  = mad24(n.d2, qi, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#1.3: nn=..:..:..:%x:%x:%x:...\n",
        V(nn.d2), V(nn.d1), V(nn.d0));
#endif

  nn.d3  = mad24(n.d3, qi, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#1.4: nn=..:..:%x:%x:%x:%x:...\n",
        V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif
  nn.d4  = mad24(n.d4, qi, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#1.5: nn=..:%x:%x:%x:%x:%x:...\n",
        V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

// no shift-left
#ifdef CHECKS_MODBASECASE
  nn.d5  = nn.d4 >> 15;  // PERF: not needed as it will be gone anyway after sub
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#1.6: nn=%x:%x:%x:%x:%x:%x:...\n",
        V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif
#endif

//  q.d0-q.d8 are all zero
  q.d4 = -nn.d0;
  q.d5 = SUB_COND(-nn.d1, q.d4 > 0x7FFF);
  q.d6 = SUB_COND(-nn.d2, q.d5 > 0x7FFF);
  q.d7 = SUB_COND(-nn.d3, q.d6 > 0x7FFF);
  q.d8 = SUB_COND(-nn.d4, q.d7 > 0x7FFF);
#ifdef CHECKS_MODBASECASE
  q.d9 = SUB_COND(qhi - nn.d5, q.d8 > 0x7FFF); // PERF: not needed: should be zero anyway
  // compiler errors: qhi=8, nn.d5=7, q.d8 > 0x7fff ==> q.d9 = 2 : skip this check
#endif
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
  q.d7 &= 0x7FFF;
  q.d8 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#1.7: q=%x!%x:%x:%x:%x:%x:..:..:..:..\n",
        V(q.d9), V(q.d8), V(q.d7), V(q.d6), V(q.d5), V(q.d4));
#endif
  // MODBASECASE_NONZERO_ERROR(q.d9, 1, 9, 1); // gives false positives

  /********** Step 2, Offset 2^40 (2*15 + 10) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d8, 32768u, q.d7));
  qf= qf * 1073741824.0f + CONVERT_FLOAT_V(mad24(q.d6, 32768u, q.d5));
  qf*= 32.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<21, 2, qi, 2);

  res->d3 = (qi >> 5);
  res->d2 = (qi << 10) & 0x7FFF;
  qil = qi & 0x7FFF;
  qih = (qi >> 15);
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"div_150_75#2: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:..:..\n",
                                 V(qf), V(nf), V(qf)*V(nf), V(qi), V(qi), V(res->d4), V(res->d3), V(res->d2));
#endif

  /*******************************************************/

// nn = n * qi
  nn.d0  = mul24(n.d0, qil);
  nn.d1  = mad24(n.d0, qih, nn.d0 >> 15);
  nn.d0 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#2.1: nn=..:..:..:..:%x:%x:..:..\n",
        V(nn.d1), V(nn.d0));
#endif

  nn.d1  = mad24(n.d1, qil, nn.d1);
  nn.d2  = mad24(n.d1, qih, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#2.2: nn=..:..:..:%x:%x:%x:..:..\n",
        V(nn.d2), V(nn.d1), V(nn.d0));
#endif

  nn.d2  = mad24(n.d2, qil, nn.d2);
  nn.d3  = mad24(n.d2, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#2.3: nn=..:..:%x:%x:%x:%x:..:..\n",
        V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

  nn.d3  = mad24(n.d3, qil, nn.d3);
  nn.d4  = mad24(n.d3, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#2.4: nn=..:%x:%x:%x:%x:%x:..:..\n",
        V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

  nn.d4  = mad24(n.d4, qil, nn.d4);
  nn.d5  = mad24(n.d4, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#2.5: nn=..:%x:%x:%x:%x:%x:%x:..:..\n",
        V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

// now shift-left 10 bits
#ifdef CHECKS_MODBASECASE
  nn.d6  = nn.d5 >> 5;  // PERF: not needed as it will be gone anyway after sub
#endif
  nn.d5  = mad24(nn.d5 & 0x1F, 1024u, nn.d4 >> 5);
  nn.d4  = mad24(nn.d4 & 0x1F, 1024u, nn.d3 >> 5);
  nn.d3  = mad24(nn.d3 & 0x1F, 1024u, nn.d2 >> 5);
  nn.d2  = mad24(nn.d2 & 0x1F, 1024u, nn.d1 >> 5);
  nn.d1  = mad24(nn.d1 & 0x1F, 1024u, nn.d0 >> 5);
  nn.d0  = (nn.d0 & 0x1F) << 10;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#2.6: nn=..:%x:%x:%x:%x:%x:%x:%x:..:..\n",
        V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

//  q = q - nn
  q.d2 = -nn.d0;
  q.d3 = SUB_COND(-nn.d1, q.d2 > 0x7FFF);
  q.d4 = SUB_COND(q.d4 - nn.d2, q.d3 > 0x7FFF);
  q.d5 = SUB_COND(q.d5 - nn.d3, q.d4 > 0x7FFF);
  q.d6 = SUB_COND(q.d6 - nn.d4, q.d5 > 0x7FFF);
  q.d7 = SUB_COND(q.d7 - nn.d5, q.d6 > 0x7FFF);
#ifdef CHECKS_MODBASECASE
  q.d8 = SUB_COND(q.d8 - nn.d6, q.d7 > 0x7FFF); // PERF: not needed: should be zero anyway
#endif
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
  q.d7 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#2.7: q=..:%x!%x:%x:%x:%x:%x:%x:..:..\n",
        V(q.d8), V(q.d7), V(q.d6), V(q.d5), V(q.d4), V(q.d3), V(q.d2));
#endif

  MODBASECASE_NONZERO_ERROR(q.d8, 2, 8, 3);

  /********** Step 3, Offset 2^20 (1*15 + 5) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d7, 32768u, q.d6));
  qf= qf * 1073741824.0f + CONVERT_FLOAT_V(mad24(q.d5, 32768u, q.d4));
  qf*= 32768.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<26, 3, qi, 5);  // very big qi, but then we can skip the bit-shifting later

  qih = (qi >> 15);
  qil = qi & 0x7FFF;
  res->d1  = qi;  // carry to d2 is handled at the end anyway
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"div_150_75#3: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:..\n",
                                 V(qf), V(nf), V(qf)*V(nf), V(qi), V(qi), V(res->d4), V(res->d3), V(res->d2), V(res->d1));
#endif

  /*******************************************************/

// nn = n * qi
  nn.d0  = mul24(n.d0, qil);
  nn.d1  = mad24(n.d0, qih, nn.d0 >> 15);
  nn.d0 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#3.1: nn=..:..:..:..:%x:%x:..\n",
        V(nn.d1), V(nn.d0));
#endif

  nn.d1  = mad24(n.d1, qil, nn.d1);
  nn.d2  = mad24(n.d1, qih, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#3.2: nn=..:..:..:%x:%x:%x:..\n",
        V(nn.d2), V(nn.d1), V(nn.d0));
#endif

  nn.d2  = mad24(n.d2, qil, nn.d2);
  nn.d3  = mad24(n.d2, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#3.3: nn=..:..:%x:%x:%x:%x:..\n",
        V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

  nn.d3  = mad24(n.d3, qil, nn.d3);
  nn.d4  = mad24(n.d3, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#3.4: nn=..:%x:%x:%x:%x:%x:..\n",
        V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif
  nn.d4  = mad24(n.d4, qil, nn.d4);
  nn.d5  = mad24(n.d4, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#3.5: nn=..:%x:%x:%x:%x:%x:%x:..\n",
        V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif
#ifdef CHECKS_MODBASECASE
  nn.d6  = nn.d5 >> 15;  // PERF: not needed as it will be gone anyway after sub
  nn.d5 &= 0x7FFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#3.6: nn=..:..:%x:%x:%x:%x:%x:%x:..\n",
        V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1), V(nn.d0));
#endif

//  q = q - nn
  q.d1 = -nn.d0;
  q.d2 = SUB_COND(q.d2 - nn.d1, q.d1 > 0x7FFF);
  q.d3 = SUB_COND(q.d3 - nn.d2, q.d2 > 0x7FFF);
  q.d4 = SUB_COND(q.d4 - nn.d3, q.d3 > 0x7FFF);
  q.d5 = SUB_COND(q.d5 - nn.d4, q.d4 > 0x7FFF);
  q.d6 = SUB_COND(q.d6 - nn.d5, q.d5 > 0x7FFF);
#ifdef CHECKS_MODBASECASE
  q.d7 = SUB_COND(q.d7 - nn.d6, q.d6 > 0x7FFF); // PERF: not needed: should be zero anyway
  q.d7 &= 0x7FFF;
#endif
  q.d1 &= 0x7FFF;
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#3.7: q=..:%x:%x!%x:%x:%x:%x:%x:%x:..\n",
        V(q.d8), V(q.d7), V(q.d6), V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1));
#endif

  MODBASECASE_NONZERO_ERROR(q.d7, 3, 7, 6);

  /********** Step 4, Offset 2^0 (0*15 + 0) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d6, 32768u, q.d5));
  qf= qf * 1073741824.0f + CONVERT_FLOAT_V(mad24(q.d4, 32768u, q.d3));
  qf*= 32768.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 7);

  qil = qi & 0x7FFF;
  qih = (qi >> 15);
  res->d1 += qih;
  res->d0 = qil;

  // skip the last part - it will change the result by one at most - we can live with a result that is off by one
  // but need to handle outstanding carries instead
  res->d2 += res->d1 >> 15;
  res->d1 &= 0x7FFF;
  res->d3 += res->d2 >> 15;
  res->d2 &= 0x7FFF;
  res->d4 += res->d3 >> 15;
  res->d3 &= 0x7FFF;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"div_150_75#4: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:%x\n",
                                 V(qf), V(nf), V(qf)*V(nf), V(qi), V(qi), V(res->d4), V(res->d3), V(res->d2), V(res->d1), V(res->d0));
#endif

}

/****
 * the trial factoring implementations for 5x15 bit
 * bit_max65 is bit_max - 65
 ****/

void check_barrett15_69(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
     MODBASECASE_PAR_DEF)
{
  __private int75_v a, u;
  __private int150_v b, tmp150;
  __private int75_v tmp75;
  __private float_v ff;
  __private uint bit_max_75=11-bit_max65, bit_max_60=bit_max65+4; //bit_max is 61 .. 70
  __private uint tmp, bit_max75_mult = 1 << bit_max_75; /* used for bit shifting... */
  __private int150_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#if defined USE_DP
  __private double_v ffd;
#endif

/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d4, 32768u, f.d3));
  ff= ff * 32768.0f + CONVERT_FLOAT_RTP_V(f.d2);   // f.d1 needed?

  ff= as_float(0x3f7ffffc) / ff;

  tmp = 1 << bit_max_60;	// tmp150 = 2^(74 + bits in f)

#if defined USE_DP
  // ffd = f as double, needed in div_180_90_d).
  ffd = CONVERT_DOUBLE_RTP_V(mad24(f.d4, 32768u, f.d3));
  ffd = ffd * 1073741824.0 + CONVERT_DOUBLE_RTP_V(mad24(f.d2, 32768u, f.d1));
  ffd = ffd * 32768.0 + CONVERT_DOUBLE_RTP_V(f.d0);
  ffd = as_double(0x3feffffffffffffdL) / ffd;     // should be a bit less than 1.0

  div_150_75_d(&u, tmp, f, ffd
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69: u(d)=%x:%x:%x:%x:%x:%x, ffd=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else
  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper part (15 bits) of a 150-bit value. The lower 135 bits are all zero implicitly

  div_150_75(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR
);						// u = floor(tmp150 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69: u=%x:%x:%x:%x:%x, ff=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

#if (TRACE_KERNEL > 11)
      mul_75_150(&tmp150, f, u);					// verify division: tmp150 should be  1
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69: f=%x:%x:%x:%x:%x * u=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x\n",
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4), V(tmp150.d3), V(tmp150.d2), V(tmp150.d1), V(tmp150.d0));
#endif

  a.d0 = mad24(bb.d5, bit_max75_mult, (bb.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bb.d6, bit_max75_mult, (bb.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bb.d7, bit_max75_mult, (bb.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d3 = mad24(bb.d8, bit_max75_mult, (bb.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d4 = mad24(bb.d9, bit_max75_mult, (bb.d8 >> bit_max_60));		        	// a = b / (2^bit_max)

  mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif

  a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

  mul_75(&tmp75, a, f);							// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    // bb.d0-bb.d3 are 0 due to preprocessing on the host, thus always require a borrow
  a.d0 = (-tmp75.d0) & 0x7FFF;
  a.d1 = SUB_COND(-tmp75.d1, a.d0 > 0);
  a.d2 = SUB_COND(-tmp75.d2, a.d1 > 0x7FFF);
  a.d3 = SUB_COND(-tmp75.d3, a.d2 > 0x7FFF);
  a.d4 = SUB_COND(bb.d4-tmp75.d4, a.d3 > 0x7FFF) & 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

  while(shifter)
  {
    square_75_150(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    a.d0 = mad24(b.d5, bit_max75_mult, (b.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(b.d6, bit_max75_mult, (b.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(b.d7, bit_max75_mult, (b.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(b.d8, bit_max75_mult, (b.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(b.d9, bit_max75_mult, (b.d8 >> bit_max_60));       			// a = b / (2^bit_max)

    mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif
    a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

    mul_75(&tmp75, a, f);						// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    a.d0 = (b.d0 - tmp75.d0) & 0x7FFF;
    a.d1 = (b.d1 - tmp75.d1 + AS_UINT_V((a.d0 > b.d0)  ));
    a.d2 = (b.d2 - tmp75.d2 + AS_UINT_V((a.d1 > b.d1)  ));
    a.d3 = (b.d3 - tmp75.d3 + AS_UINT_V((a.d2 > b.d2)  ));
    a.d4 = (b.d4 - tmp75.d4 + AS_UINT_V((a.d3 > b.d3)  ));
    a.d1 &= 0x7FFF;
    a.d2 &= 0x7FFF;
    a.d3 &= 0x7FFF;
    a.d4 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

    if(shifter&0x80000000)shl_75(&a);					// "optional multiply by 2" in Prime 95 documentation

#ifdef CHECKS_MODBASECASE
// a.d4 must not exceed 0x7fff, otherwise the following squaring may overflow
#endif

    shifter+=shifter;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, a= %x:%x:%x:%x:%x\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0) );
#endif
  }

  mod_simple_even_75_and_check_big_factor75(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , bit_max_75, 10, modbasecase_debug
#endif
                       );
}

void check_barrett15_70(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
     MODBASECASE_PAR_DEF)
{
  __private int75_v a, u;
  __private int150_v b, tmp150;
  __private int75_v tmp75;
  __private float_v ff;
  __private uint bit_max_75=11-bit_max65, bit_max_60=bit_max65+4; //bit_max is 61 .. 70
  __private uint tmp, bit_max75_mult = 1 << bit_max_75; /* used for bit shifting... */
  __private int150_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#if defined USE_DP
  __private double_v ffd;
#endif

/*
ff = f as float, needed in mod_192_96() and div_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d4, 32768u, f.d3));
  ff= ff * 32768.0f + CONVERT_FLOAT_RTP_V(f.d2);   // f.d1 needed?

  ff= as_float(0x3f7ffffc) / ff;

  tmp = 1 << bit_max_60;	// tmp150 = 2^(74 + bits in f)

#if defined USE_DP
  // ffd = f as double, needed in div_180_90_d).
  ffd = CONVERT_DOUBLE_RTP_V(mad24(f.d4, 32768u, f.d3));
  ffd = ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d2, 32768u, f.d1));
  ffd = ffd * 32768.0 + CONVERT_DOUBLE_RTP_V(f.d0);
  ffd = as_double(0x3feffffffffffffdL) / ffd;     // should be a bit less than 1.0

  div_150_75_d(&u, tmp, f, ffd
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_70: u(d)=%x:%x:%x:%x:%x:%x, ffd=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else
  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper part (15 bits) of a 150-bit value. The lower 135 bits are all zero implicitly

  div_150_75(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR
);						// u = floor(tmp150 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_70: u=%x:%x:%x:%x:%x, ff=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  a.d0 = mad24(bb.d5, bit_max75_mult, (bb.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bb.d6, bit_max75_mult, (bb.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bb.d7, bit_max75_mult, (bb.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d3 = mad24(bb.d8, bit_max75_mult, (bb.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d4 = mad24(bb.d9, bit_max75_mult, (bb.d8 >> bit_max_60));		        	// a = b / (2^bit_max)

  mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_70: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif

  a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

  mul_75(&tmp75, a, f);							// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_70: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    // bb.d0-bb.d3 are 0 due to preprocessing on the host, thus always require a borrow
  a.d0 = (-tmp75.d0) & 0x7FFF;
  a.d1 = (-tmp75.d1 + AS_UINT_V((a.d0 > 0)  ));
  a.d2 = (-tmp75.d2 + AS_UINT_V((a.d1 > 0x7FFF)  ));
  a.d3 = (-tmp75.d3 + AS_UINT_V((a.d2 > 0x7FFF)  ));
  a.d4 = (bb.d4-tmp75.d4 + AS_UINT_V((a.d3 > 0x7FFF)  )) & 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_70: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

  while(shifter)
  {
    square_75_150(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    a.d0 = mad24(b.d5, bit_max75_mult, (b.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(b.d6, bit_max75_mult, (b.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(b.d7, bit_max75_mult, (b.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(b.d8, bit_max75_mult, (b.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(b.d9, bit_max75_mult, (b.d8 >> bit_max_60));       			// a = b / (2^bit_max)

    mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif
    a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

    mul_75(&tmp75, a, f);						// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    a.d0 = (b.d0 - tmp75.d0) & 0x7FFF;
    a.d1 = (b.d1 - tmp75.d1 + AS_UINT_V((a.d0 > b.d0)  ));
    a.d2 = (b.d2 - tmp75.d2 + AS_UINT_V((a.d1 > b.d1)  ));
    a.d3 = (b.d3 - tmp75.d3 + AS_UINT_V((a.d2 > b.d2)  ));
    a.d4 = (b.d4 - tmp75.d4 + AS_UINT_V((a.d3 > b.d3)  ));
    a.d1 &= 0x7FFF;
    a.d2 &= 0x7FFF;
    a.d3 &= 0x7FFF;
    a.d4 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

    if(shifter&0x80000000)shl_75(&a);					// "optional multiply by 2" in Prime 95 documentation

#ifdef CHECKS_MODBASECASE
// a.d4 must not exceed 0x7fff, otherwise the following squaring may overflow
#endif

    shifter+=shifter;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, a= %x:%x:%x:%x:%x\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0) );
#endif
  }

  mod_simple_75_and_check_big_factor75(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , bit_max_75, 10, modbasecase_debug
#endif
                       );
}


void check_barrett15_71(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int75_v a, u;
  __private int150_v b, tmp150;
  __private int75_v tmp75;
  __private float_v ff;
  __private uint bit_max_75=11-bit_max65, bit_max_60=bit_max65+4; //bit_max is 61 .. 70
  __private uint tmp, bit_max75_mult = 1 << bit_max_75; /* used for bit shifting... */
  __private int150_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#if defined USE_DP
  __private double_v ffd;
#endif

/*
ff = 1/f as float, needed in div_192_96().
*/
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d4, 32768u, f.d3));
  ff= ff * 32768.0f + CONVERT_FLOAT_RTP_V(f.d2);   // f.d1 needed?

  ff= as_float(0x3f7ffffc) / ff;

  tmp = 1 << bit_max_60;	// tmp150 = 2^(74 + bits in f)

#if defined USE_DP
  // ffd = f as double, needed in div_180_90_d).
  ffd = CONVERT_DOUBLE_RTP_V(mad24(f.d4, 32768u, f.d3));
  ffd = ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d2, 32768u, f.d1));
  ffd = ffd * 32768.0 + CONVERT_DOUBLE_RTP_V(f.d0);
  ffd = as_double(0x3feffffffffffffdL) / ffd;     // should be a bit less than 1.0

  div_150_75_d(&u, tmp, f, ffd
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_71: u(d)=%x:%x:%x:%x:%x:%x, ffd=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else
  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper part (15 bits) of a 150-bit value. The lower 135 bits are all zero implicitly

  div_150_75(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR
);						// u = floor(tmp150 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_71: u=%x:%x:%x:%x:%x, ff=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  a.d0 = mad24(bb.d5, bit_max75_mult, (bb.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bb.d6, bit_max75_mult, (bb.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bb.d7, bit_max75_mult, (bb.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d3 = mad24(bb.d8, bit_max75_mult, (bb.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d4 = mad24(bb.d9, bit_max75_mult, (bb.d8 >> bit_max_60));		        	// a = b / (2^bit_max)

  mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_71: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif

  a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

  mul_75(&tmp75, a, f);							// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_71: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    // bb.d0-bb.d3 are 0 due to preprocessing on the host, thus always require a borrow
  a.d0 = (-tmp75.d0) & 0x7FFF;
  a.d1 = (-tmp75.d1 + AS_UINT_V((a.d0 > 0)  ));
  a.d2 = (-tmp75.d2 + AS_UINT_V((a.d1 > 0x7FFF)  ));
  a.d3 = (-tmp75.d3 + AS_UINT_V((a.d2 > 0x7FFF)  ));
  a.d4 = (bb.d4-tmp75.d4 + AS_UINT_V((a.d3 > 0x7FFF)  )) & 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_71: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

  while(shifter)
  {
    square_75_150(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif

    if(shifter&0x80000000)
    {
      shl_150(&b);					// "optional multiply by 2" in Prime 95 documentation
#if (TRACE_KERNEL > 2)
      if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    }

    a.d0 = mad24(b.d5, bit_max75_mult, (b.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(b.d6, bit_max75_mult, (b.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(b.d7, bit_max75_mult, (b.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(b.d8, bit_max75_mult, (b.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(b.d9, bit_max75_mult, (b.d8 >> bit_max_60));       			// a = b / (2^bit_max)

    mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif
    a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

    mul_75(&tmp75, a, f);						// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    a.d0 = (b.d0 - tmp75.d0) & 0x7FFF;
    a.d1 = (b.d1 - tmp75.d1 + AS_UINT_V((a.d0 > b.d0)  ));
    a.d2 = (b.d2 - tmp75.d2 + AS_UINT_V((a.d1 > b.d1)  ));
    a.d3 = (b.d3 - tmp75.d3 + AS_UINT_V((a.d2 > b.d2)  ));
    a.d4 = (b.d4 - tmp75.d4 + AS_UINT_V((a.d3 > b.d3)  ));
    a.d1 &= 0x7FFF;
    a.d2 &= 0x7FFF;
    a.d3 &= 0x7FFF;
    a.d4 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

    shifter+=shifter;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, a= %x:%x:%x:%x:%x\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0) );
#endif
  }

  mod_simple_75_and_check_big_factor75(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , bit_max_75, 10, modbasecase_debug
#endif
                       );
}


void check_barrett15_73(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int75_v a, u;
  __private int150_v b, tmp150;
  __private int75_v tmp75;
  __private float_v ff;
  __private uint bit_max_75=11-bit_max65, bit_max_60=bit_max65+4; //bit_max is 61 .. 70
  __private uint tmp, bit_max75_mult = 1 << bit_max_75; /* used for bit shifting... */
  __private int150_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#if defined USE_DP
  __private double_v ffd;
#endif


/*
ff = 1/f as float, needed in div_192_96().
*/
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d4, 32768u, f.d3));
  ff= ff * 32768.0f + CONVERT_FLOAT_RTP_V(f.d2);   // these are at least 30 significant bits for 60-bit FC's

  ff= as_float(0x3f7ffffc) / ff;

  tmp = 1 << bit_max_60;	// tmp150 = 2^(74 + bits in f)

#if defined USE_DP
  // ffd = f as double, needed in div_180_90_d).
  ffd = CONVERT_DOUBLE_RTP_V(mad24(f.d4, 32768u, f.d3));
  ffd = ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d2, 32768u, f.d1));
  ffd = ffd * 32768.0 + CONVERT_DOUBLE_RTP_V(f.d0);
  ffd = as_double(0x3feffffffffffffdL) / ffd;     // should be a bit less than 1.0

  div_150_75_d(&u, tmp, f, ffd
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73: u(d)=%x:%x:%x:%x:%x:%x, ffd=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else
  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper part (15 bits) of a 150-bit value. The lower 135 bits are all zero implicitly

  div_150_75(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR
);						// u = floor(tmp150 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73: u=%x:%x:%x:%x:%x, ff=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  a.d0 = mad24(bb.d5, bit_max75_mult, (bb.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bb.d6, bit_max75_mult, (bb.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bb.d7, bit_max75_mult, (bb.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d3 = mad24(bb.d8, bit_max75_mult, (bb.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d4 = mad24(bb.d9, bit_max75_mult, (bb.d8 >> bit_max_60));		        	// a = b / (2^bit_max)

  mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif

  a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

  mul_75(&tmp75, a, f);							// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    // all those bb's are 0 due to preprocessing on the host, thus always require a borrow
  a.d0 = (-tmp75.d0) & 0x7FFF;
  a.d1 = SUB_COND(-tmp75.d1, a.d0 > 0);
  a.d2 = SUB_COND(-tmp75.d2, a.d1 > 0x7FFF);
  a.d3 = SUB_COND(-tmp75.d3, a.d2 > 0x7FFF);
  a.d4 = SUB_COND(bb.d4 - tmp75.d4, a.d3 > 0x7FFF) & 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

  for(;;)
  {
    square_75_150(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
#if (TRACE_KERNEL > 14)
    // verify squaring by dividing again.
    __private float_v f1 = CONVERT_FLOAT_RTP_V(mad24(a.d4, 32768u, a.d3));
    f1= f1 * 32768.0f + CONVERT_FLOAT_RTP_V(a.d2);   // f.d1 needed?

    f1= as_float(0x3f7ffffc) / f1;
    div_150_75(&tmp75, V(b.d9), a, f1, tid
               MODBASECASE_PAR
              );
    if (tid==TRACE_TID) printf((__constant char *)"vrfy: b = %x:0:0:0:0:0:0:0:0:0 / a=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x\n",
        V(b.d9),
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    a.d0 = mad24(b.d5, bit_max75_mult, (b.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(b.d6, bit_max75_mult, (b.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(b.d7, bit_max75_mult, (b.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(b.d8, bit_max75_mult, (b.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(b.d9, bit_max75_mult, (b.d8 >> bit_max_60));       			// a = b / (2^bit_max)

    mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif
    a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    mul_75(&tmp75, a, f);						// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    tmp75.d0 = (b.d0 - tmp75.d0) & 0x7FFF;
    tmp75.d1 = SUB_COND(b.d1 - tmp75.d1, tmp75.d0 > b.d0);
    tmp75.d2 = SUB_COND(b.d2 - tmp75.d2, tmp75.d1 > 0x7FFF);
    tmp75.d3 = SUB_COND(b.d3 - tmp75.d3, tmp75.d2 > 0x7FFF);
    tmp75.d4 = SUB_COND(b.d4 - tmp75.d4, tmp75.d3 > 0x7FFF);
    tmp75.d1 &= 0x7FFF;
    tmp75.d2 &= 0x7FFF;
    tmp75.d3 &= 0x7FFF;
    tmp75.d4 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (tmp)\n",
        V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    if (shifter & 0x80000000) shl_75(&tmp75);
    if (shifter == 0x80000000) break;
    shifter+=shifter;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, tmp=%x:%x:%x:%x:%x mod f=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x (a)\n",
        shifter, V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0),
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0) );
#endif

    #ifndef CHECKS_MODBASECASE
    mod_simple_75(&a, tmp75, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 10;
    if(bit_max_75 == 2) limit = 12;
    if(bit_max_75 == 3) limit = 11;
    mod_simple_75(&a, tmp75, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max_75, limit, modbasecase_debug);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73: tmp=%x:%x:%x:%x:%x mod f=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x (a)\n",
        V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0),
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0) );
#endif
  }

  mod_simple_even_75_and_check_big_factor75(tmp75, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , bit_max_75, 10, modbasecase_debug
#endif
                       );
}


void check_barrett15_74(uint shifter, const int75_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int75_v a, u;
  __private int150_v b, tmp150;
  __private int75_v tmp75;
  __private float_v ff;
  __private uint bit_max_75=11-bit_max65, bit_max_60=bit_max65+4; //bit_max is 61 .. 70
  __private uint tmp, bit_max75_mult = 1 << bit_max_75; /* used for bit shifting... */
  __private int150_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#if defined USE_DP
  __private double_v ffd;
#endif

  // this kernel is based on the 73-bit kernel but stores one more bit in the top word, allowing to factor up to 74 bits.
/*
ff = 1/f as float, needed in div_192_96().
*/
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d4, 32768u, f.d3));
  ff= ff * 32768.0f + CONVERT_FLOAT_RTP_V(f.d2);   // these are at least 30 significant bits for 60-bit FC's

  ff= as_float(0x3f7ffffc) / ff;

  tmp = 1 << bit_max_60;	// tmp150 = 2^(74 + bits in f)

#if defined USE_DP
  // ffd = f as double, needed in div_180_90_d).
  ffd = CONVERT_DOUBLE_RTP_V(mad24(f.d4, 32768u, f.d3));
  ffd = ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d2, 32768u, f.d1));
  ffd = ffd * 32768.0 + CONVERT_DOUBLE_RTP_V(f.d0);
  ffd = as_double(0x3feffffffffffffdL) / ffd;     // should be a bit less than 1.0

  div_150_75_d(&u, tmp, f, ffd
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74: u(d)=%x:%x:%x:%x:%x:%x, ffd=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else
  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper part (15 bits) of a 150-bit value. The lower 135 bits are all zero implicitly

  div_150_75(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR
);						// u = floor(tmp150 / f)

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74: u=%x:%x:%x:%x:%x, ff=%G\n",
        V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

#if (TRACE_KERNEL > 10)
    // verify u
    mul_75_150(&tmp150, u, f);
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74: vrfy: f*u=%x:%x:%x:%x:%x:%x:%x:%x:%x:%x\n",
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5),
        V(tmp150.d4), V(tmp150.d3), V(tmp150.d2), V(tmp150.d1), V(tmp150.d0));
#endif
  a.d0 = mad24(bb.d5, bit_max75_mult, (bb.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d1 = mad24(bb.d6, bit_max75_mult, (bb.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d2 = mad24(bb.d7, bit_max75_mult, (bb.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d3 = mad24(bb.d8, bit_max75_mult, (bb.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
  a.d4 = mad24(bb.d9, bit_max75_mult, (bb.d8 >> bit_max_60));		        	// a = b / (2^bit_max)

  mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif

  a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)

  mul_75_big(&tmp75, a, f);							// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    // all those bb's are 0 due to preprocessing on the host, thus always require a borrow
  a.d0 = (-tmp75.d0) & 0x7FFF;
  a.d1 = SUB_COND(-tmp75.d1, a.d0 > 0);
  a.d2 = SUB_COND(-tmp75.d2, a.d1 > 0x7FFF);
  a.d3 = SUB_COND(-tmp75.d3, a.d2 > 0x7FFF);
  a.d4 = SUB_COND(mad24(bb.d5, 32768u, bb.d4) - tmp75.d4, a.d3 > 0x7FFF) & 0xFFFF;  // keep one extra bit
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (a)\n",
        mad24(bb.d5, 32768u, bb.d4), bb.d3, bb.d2, bb.d1, bb.d0, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

  for(;;)
  {
    square_75_150(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
#if (TRACE_KERNEL > 14)
    // verify squaring by dividing again.
    __private float_v f1 = CONVERT_FLOAT_RTP_V(mad24(a.d4, 32768u, a.d3));
    f1= f1 * 32768.0f + CONVERT_FLOAT_RTP_V(a.d2);   // f.d1 needed?

    f1= as_float(0x3f7ffffc) / f1;
    div_150_75(&tmp75, V(b.d9), a, f1, tid
               MODBASECASE_PAR
              );
    if (tid==TRACE_TID) printf((__constant char *)"vrfy: b = %x:0:0:0:0:0:0:0:0:0 / a=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x\n",
        V(b.d9),
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    a.d0 = mad24(b.d5, bit_max75_mult, (b.d4 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(b.d6, bit_max75_mult, (b.d5 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(b.d7, bit_max75_mult, (b.d6 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(b.d8, bit_max75_mult, (b.d7 >> bit_max_60))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(b.d9, bit_max75_mult, (b.d8 >> bit_max_60));       			// a = b / (2^bit_max)

    mul_75_150_no_low5(&tmp150, a, u);					// tmp150 = (b / (2^bit_max)) * u # at least close to ;)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:...\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp150.d9), V(tmp150.d8), V(tmp150.d7), V(tmp150.d6), V(tmp150.d5), V(tmp150.d4));
#endif
    a.d0 = tmp150.d5;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = tmp150.d6;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d2 = tmp150.d7;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d3 = tmp150.d8;			// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d4 = tmp150.d9;		        	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    mul_75_big(&tmp75, a, f);						// tmp75 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x (tmp)\n",
        V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    tmp75.d0 = (b.d0 - tmp75.d0) & 0x7FFF;
    tmp75.d1 = SUB_COND(b.d1 - tmp75.d1, tmp75.d0 > b.d0);
    tmp75.d2 = SUB_COND(b.d2 - tmp75.d2, tmp75.d1 > 0x7FFF);
    tmp75.d3 = SUB_COND(b.d3 - tmp75.d3, tmp75.d2 > 0x7FFF);
    tmp75.d4 = SUB_COND(mad24(b.d5, 32768u, b.d4) - tmp75.d4, tmp75.d3 > 0x7FFF);
    tmp75.d1 &= 0x7FFF;
    tmp75.d2 &= 0x7FFF;
    tmp75.d3 &= 0x7FFF;
    tmp75.d4 &= 0x1FFFF; // keep 2 extra bits

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x (tmp)\n",
        mad24(V(b.d5), 32768u, V(b.d4)), V(b.d3), V(b.d2), V(b.d1), V(b.d0), V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0));
#endif
    if (shifter & 0x80000000) shl_75(&tmp75);
    if (shifter == 0x80000000) break;
    shifter+=shifter;

#ifndef CHECKS_MODBASECASE
    mod_simple_75_big(&a, tmp75, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 14;
    if(bit_max_75 == 2) limit = 16;						// bit_max == 65, due to decreased accuracy of mul_96_192_no_low2() above we need a higher threshold
    if(bit_max_75 == 3) limit = 15;						// bit_max == 66, ...
    mod_simple_75_big(&a, tmp75, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max_75, limit, modbasecase_debug);
#endif

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, tmp=%x:%x:%x:%x:%x mod f=%x:%x:%x:%x:%x = %x:%x:%x:%x:%x (a)\n",
        shifter, V(tmp75.d4), V(tmp75.d3), V(tmp75.d2), V(tmp75.d1), V(tmp75.d0),
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0) );
#endif
  }

  mod_simple_even_75_and_check_big_factor75_big(tmp75, f, ff, RES
#ifdef CHECKS_MODBASECASE
                       , bit_max_75, 10, modbasecase_debug
#endif
                       );
}

/******
 * now the actual kernels for 5x15 bit calculations
 *
 * shiftcount is used for precomputing without mod
 * b_in is precomputed on host ONCE.
  ******/

__kernel void cl_barrett15_69(__private uint exponent, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                           const uint8 b_in, __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF         )
{
  __private int75_v f;
  __private uint tid;

	tid = 	mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC75(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69: f=%x:%x:%x:%x:%x, shift=%d\n",
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett15_69(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}


__kernel void cl_barrett15_70(__private uint exponent, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                           const uint8 b_in, __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF         )
{
  __private int75_v f;
  __private uint tid;

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC75(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_70: f=%x:%x:%x:%x:%x, shift=%d\n",
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett15_70(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}


__kernel void cl_barrett15_71(__private uint exponent, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                           const uint8 b_in, __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF         )
{
  __private int75_v f;
  __private uint tid;

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC75(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_71: f=%x:%x:%x:%x:%x, shift=%d\n",
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett15_71(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}


__kernel void cl_barrett15_73(__private uint exponent, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                           const uint8 b_in, __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF         )
{
  __private int75_v f;
  __private uint tid;

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC75(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73: f=%x:%x:%x:%x:%x, shift=%d\n",
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett15_73(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}

__kernel void cl_barrett15_74(__private uint exponent, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                           const uint8 b_in, __global uint * restrict RES, const int bit_max65
                           MODBASECASE_PAR_DEF         )
{
  __private int75_v f;
  __private uint tid;

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC75(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74: f=%x:%x:%x:%x:%x, shift=%d\n",
        V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif

  check_barrett15_74(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}


/****************************************
 ****************************************
 * 15-bit based 90-bit barrett-kernels
 *
 ****************************************
 ****************************************/

int90_v sub_if_gte_90(const int90_v a, const int90_v b)
/* return (a>b)?a-b:a */
{
  int90_v tmp;
  /* do the subtraction and use tmp.d5 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0) & 0x7FFF;
  tmp.d1 = (a.d1 - b.d1 + AS_UINT_V((b.d0 > a.d0) ));
  tmp.d2 = (a.d2 - b.d2 + AS_UINT_V((tmp.d1 > a.d1) ));
  tmp.d3 = (a.d3 - b.d3 + AS_UINT_V((tmp.d2 > a.d2) ));
  tmp.d4 = (a.d4 - b.d4 + AS_UINT_V((tmp.d3 > a.d3) ));
  tmp.d5 = (a.d5 - b.d5 + AS_UINT_V((tmp.d4 > a.d4) ));
  tmp.d1&= 0x7FFF;
  tmp.d2&= 0x7FFF;
  tmp.d3&= 0x7FFF;
  tmp.d4&= 0x7FFF;

  tmp.d0 = (tmp.d5 > a.d5) ? a.d0 : tmp.d0;
  tmp.d1 = (tmp.d5 > a.d5) ? a.d1 : tmp.d1;
  tmp.d2 = (tmp.d5 > a.d5) ? a.d2 : tmp.d2;
  tmp.d3 = (tmp.d5 > a.d5) ? a.d3 : tmp.d3;
  tmp.d4 = (tmp.d5 > a.d5) ? a.d4 : tmp.d4;
  tmp.d5 = (tmp.d5 > a.d5) ? a.d5 : tmp.d5; //  & 0x7FFF not necessary as tmp.d5 is <= a.d5

  return tmp;
}


void mul_90(int90_v * const res, const int90_v a, const int90_v b)
/* res = a * b
   21 mul/mad24, 6 >>, 7 &=, 1 +*/
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
  res->d2 &= 0x7FFF;
  res->d3 = mad24(a.d2, b.d1, res->d3);
  res->d3 = mad24(a.d1, b.d2, res->d3);
  res->d3 = mad24(a.d0, b.d3, res->d3);

  res->d4 = mad24(a.d4, b.d0, res->d3 >> 15);
  res->d3 &= 0x7FFF;
  res->d4 = mad24(a.d3, b.d1, res->d4);
  res->d4 = mad24(a.d2, b.d2, res->d4);
  res->d4 = mad24(a.d1, b.d3, res->d4);
  res->d5 = mad24(a.d5, b.d0, res->d4 >> 15);  // the 5th mad can overflow d4, need to handle carry before
  res->d4 &= 0x7FFF;
  res->d4 = mad24(a.d0, b.d4, res->d4);

  res->d5 += mad24(a.d4, b.d1, res->d4 >> 15);
  res->d4 &= 0x7FFF;
  res->d5 = mad24(a.d3, b.d2, res->d5);
  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d5 = mad24(a.d1, b.d4, res->d5);  // the 5th mad can overflow d5, but that's ok for this function.
  res->d5 = mad24(a.d0, b.d5, res->d5);  // the 6th mad can overflow d5, but that's ok for this function.
  res->d5 &= 0x7FFF;

}


void mul_90_180_no_low3(int180_v * const res, const int90_v a, const int90_v b)
/*
res ~= a * b
res.d0 to res.d2 are NOT computed. Carries to res.d3 are ignored,
too. So the digits res.d{3-b} might differ from mul_90_180().
 */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // this optimized mul 6x6 requires: 30 mul/mad24, 11 shift, 10 and, 3 add

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
  res->d5 = mad24(a.d5, b.d0, res->d4 >> 15);
  res->d4 &= 0x7FFF;
  res->d4 = mad24(a.d0, b.d4, res->d4);  // 31-bit at most

  res->d5 = mad24(a.d4, b.d1, res->d4 >> 15) + res->d5;
  res->d4 &= 0x7FFF;
  res->d5 = mad24(a.d3, b.d2, res->d5);
  // handle carry after 3 of 6 mad's for d5: pull in the first d6 line
  res->d6 = mad24(a.d1, b.d5, res->d5 >> 15);
  res->d5 &= 0x7FFF;

  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d5 = mad24(a.d1, b.d4, res->d5);
  res->d5 = mad24(a.d0, b.d5, res->d5);

  res->d6 = mad24(a.d2, b.d4, res->d5 >> 15) + res->d6;
  res->d5 &= 0x7FFF;
  res->d6 = mad24(a.d3, b.d3, res->d6);
   // handle carry after 3 of 5 mad's for d6: pull in the first d7 line
  res->d7 = mad24(a.d2, b.d5, res->d6 >> 15);
  res->d6 &= 0x7FFF;

  res->d6 = mad24(a.d4, b.d2, res->d6);
  res->d6 = mad24(a.d5, b.d1, res->d6);

  res->d7 = mad24(a.d3, b.d4, res->d6 >> 15) + res->d7;
  res->d7 = mad24(a.d4, b.d3, res->d7);
  res->d7 = mad24(a.d5, b.d2, res->d7);  // in d7 we have 4 mad's, and 2 carries (both not full 17 bits)
  res->d6 &= 0x7FFF;

  res->d8 = mad24(a.d5, b.d3, res->d7 >> 15);
  res->d8 = mad24(a.d4, b.d4, res->d8);
  res->d8 = mad24(a.d3, b.d5, res->d8);
  res->d7 &= 0x7FFF;

  res->d9 = mad24(a.d5, b.d4, res->d8 >> 15);
  res->d9 = mad24(a.d4, b.d5, res->d9);
  res->d8 &= 0x7FFF;

  res->da = mad24(a.d5, b.d5, res->d9 >> 15);
  res->d9 &= 0x7FFF;

  res->db = res->da >> 15;
  res->da &= 0x7FFF;
}

void mul_90_180_no_low5(int180_v * const res, const int90_v a, const int90_v b)
/*
res ~= a * b
res.d0 to res.d4 are NOT computed. res.d5 is computed only to provide carries to res.d6.
 */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // this optimized mul 6x6 requires: 21 mul/mad24, 8 shift, 7 and, 2 add

   // 5th mad24 can overflow d4, need to handle carry before: pull in the first d5 line

  res->d5 = mul24(a.d5, b.d0);
  res->d5 = mad24(a.d4, b.d1, res->d5);
  res->d5 = mad24(a.d3, b.d2, res->d5);
  // handle carry after 3 of 6 mad's for d5: pull in the first d6 line
  res->d6 = mad24(a.d1, b.d5, res->d5 >> 15);
  res->d5 &= 0x7FFF;

  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d5 = mad24(a.d1, b.d4, res->d5);
  res->d5 = mad24(a.d0, b.d5, res->d5);

  res->d6 = mad24(a.d2, b.d4, res->d5 >> 15) + res->d6;
//  res->d5 &= 0x7FFF;
  res->d6 = mad24(a.d3, b.d3, res->d6);
   // handle carry after 3 of 5 mad's for d6: pull in the first d7 line
  res->d7 = mad24(a.d2, b.d5, res->d6 >> 15);
  res->d6 &= 0x7FFF;

  res->d6 = mad24(a.d4, b.d2, res->d6);
  res->d6 = mad24(a.d5, b.d1, res->d6);

  res->d7 = mad24(a.d3, b.d4, res->d6 >> 15) + res->d7;
  res->d7 = mad24(a.d4, b.d3, res->d7);
  res->d7 = mad24(a.d5, b.d2, res->d7);  // in d7 we have 4 mad's, and 2 carries (both not full 17 bits)
  res->d6 &= 0x7FFF;

  res->d8 = mad24(a.d5, b.d3, res->d7 >> 15);
  res->d8 = mad24(a.d4, b.d4, res->d8);
  res->d8 = mad24(a.d3, b.d5, res->d8);
  res->d7 &= 0x7FFF;

  res->d9 = mad24(a.d5, b.d4, res->d8 >> 15);
  res->d9 = mad24(a.d4, b.d5, res->d9);
  res->d8 &= 0x7FFF;

  res->da = mad24(a.d5, b.d5, res->d9 >> 15);
  res->d9 &= 0x7FFF;

  res->db = res->da >> 15;
  res->da &= 0x7FFF;
}


void mul_90_180(int180_v * const res, const int90_v a, const int90_v b)

//  res = a * b
{
  // this is the complete implementation, used in montgomery mul, and was the basis for
  // the _no_low3 and square functions
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // mul 6x6 requires: 36 mul/mad24, 14 shift, 14 and, 3 add

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
  res->d5 = mad24(a.d5, b.d0, res->d4 >> 15);
  res->d4 &= 0x7FFF;
  res->d4 = mad24(a.d0, b.d4, res->d4);  // 31-bit at most

  res->d5 = mad24(a.d4, b.d1, res->d4 >> 15) + res->d5;
  res->d4 &= 0x7FFF;
  res->d5 = mad24(a.d3, b.d2, res->d5);
  // handle carry after 3 of 6 mad's for d5: pull in the first d6 line
  res->d6 = mad24(a.d1, b.d5, res->d5 >> 15);
  res->d5 &= 0x7FFF;

  res->d5 = mad24(a.d2, b.d3, res->d5);
  res->d5 = mad24(a.d1, b.d4, res->d5);
  res->d5 = mad24(a.d0, b.d5, res->d5);

  res->d6 = mad24(a.d2, b.d4, res->d5 >> 15) + res->d6;
  res->d5 &= 0x7FFF;
  res->d6 = mad24(a.d3, b.d3, res->d6);
   // handle carry after 3 of 5 mad's for d6: pull in the first d7 line
  res->d7 = mad24(a.d2, b.d5, res->d6 >> 15);
  res->d6 &= 0x7FFF;

  res->d6 = mad24(a.d4, b.d2, res->d6);
  res->d6 = mad24(a.d5, b.d1, res->d6);

  res->d7 = mad24(a.d3, b.d4, res->d6 >> 15) + res->d7;
  res->d7 = mad24(a.d4, b.d3, res->d7);
  res->d7 = mad24(a.d5, b.d2, res->d7);  // in d7 we have 4 mad's, and 2 carries (both not full 17 bits)
  res->d6 &= 0x7FFF;

  res->d8 = mad24(a.d5, b.d3, res->d7 >> 15);
  res->d8 = mad24(a.d4, b.d4, res->d8);
  res->d8 = mad24(a.d3, b.d5, res->d8);
  res->d7 &= 0x7FFF;

  res->d9 = mad24(a.d5, b.d4, res->d8 >> 15);
  res->d9 = mad24(a.d4, b.d5, res->d9);
  res->d8 &= 0x7FFF;

  res->da = mad24(a.d5, b.d5, res->d9 >> 15);
  res->d9 &= 0x7FFF;

  res->db = res->da >> 15;
  res->da &= 0x7FFF;
}

void square_90_180(int180_v * const res, const int90_v a)
/* res = a^2 = d0^2 + 2d0d1 + d1^2 + 2d0d2 + 2(d1d2 + d0d3) + d2^2 +
               2(d0d4 + d1d3) + 2(d1d4 + d2d3) + d3^2 + 2d2d4 + 2d3d4 + d4^2
   */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // square 6x6 requires: 21 mul/mad24, 29 shift (10 of them cacheable), 14 and, 3 add

  res->d0 = mul24(a.d0, a.d0);

  res->d1 = mad24(a.d1, a.d0 << 1, res->d0 >> 15);
  res->d0 &= 0x7FFF;

  res->d2 = mad24(a.d1, a.d1, res->d1 >> 15);
  res->d2 = mad24(a.d2, a.d0 << 1, res->d2);
  res->d1 &= 0x7FFF;

  res->d3 = mad24(a.d3, a.d0 << 1, res->d2 >> 15);
  res->d3 = mad24(a.d2, a.d1 << 1, res->d3);
  res->d2 &= 0x7FFF;

  res->d4 = mad24(a.d4, a.d0 << 1, res->d3 >> 15);
  res->d3 &= 0x7FFF;
  res->d4 = mad24(a.d3, a.d1 << 1, res->d4);
   // 5th mad24 can overflow d4, need to handle carry before: pull in the first d5 line
  res->d5 = mad24(a.d4, a.d1 << 1, res->d4 >> 15);
  res->d4 &= 0x7FFF;
  res->d4 = mad24(a.d2, a.d2, res->d4);  // 31-bit at most

  res->d5 = mad24(a.d3, a.d2 << 1, res->d4 >> 15) + res->d5;
  res->d4 &= 0x7FFF;
  res->d6 = mad24(a.d5, a.d1 << 1, res->d5 >> 15); // d5 carry handling before overflowing
  res->d5 &= 0x7FFF;
  res->d5 = mad24(a.d5, a.d0 << 1, res->d5);

  res->d7 = mad24(a.d5, a.d2 << 1, res->d6 >> 15); // d6 carry handling before overflowing
  res->d6 &= 0x7FFF;
  res->d6 = mad24(a.d4, a.d2 << 1, res->d5 >> 15) + res->d6;
  res->d6 = mad24(a.d3, a.d3, res->d6);
  res->d5 &= 0x7FFF;

  res->d7 = mad24(a.d4, a.d3 << 1, res->d6 >> 15) + res->d7;
  res->d6 &= 0x7FFF;

  res->d8 = mad24(a.d4, a.d4, res->d7 >> 15);
  res->d8 = mad24(a.d5, a.d3 << 1, res->d8);
  res->d7 &= 0x7FFF;

  res->d9 = mad24(a.d5, a.d4 << 1, res->d8 >> 15);
  res->d8 &= 0x7FFF;

  res->da = mad24(a.d5, a.d5, res->d9 >> 15);
  res->d9 &= 0x7FFF;

  res->db = res->da >> 15;
  res->da &= 0x7FFF;
}


void shl_90(int90_v * const a)
/* shiftleft a one bit */
{
  a->d5 = mad24(a->d5, 2u, a->d4 >> 14); // keep the extra top bit
  a->d4 = mad24(a->d4, 2u, a->d3 >> 14) & 0x7FFF;
  a->d3 = mad24(a->d3, 2u, a->d2 >> 14) & 0x7FFF;
  a->d2 = mad24(a->d2, 2u, a->d1 >> 14) & 0x7FFF;
  a->d1 = mad24(a->d1, 2u, a->d0 >> 14) & 0x7FFF;
  a->d0 = (a->d0 << 1u) & 0x7FFF;
}

void shl_180(int180_v * const a)
/* shiftleft a one bit */
{
  a->db = mad24(a->db, 2u, a->da >> 14); // keep the extra top bit
  a->da = mad24(a->da, 2u, a->d9 >> 14) & 0x7FFF;
  a->d9 = mad24(a->d9, 2u, a->d8 >> 14) & 0x7FFF;
  a->d8 = mad24(a->d8, 2u, a->d7 >> 14) & 0x7FFF;
  a->d7 = mad24(a->d7, 2u, a->d6 >> 14) & 0x7FFF;
  a->d6 = mad24(a->d6, 2u, a->d5 >> 14) & 0x7FFF;
  a->d5 = mad24(a->d5, 2u, a->d4 >> 14) & 0x7FFF;
  a->d4 = mad24(a->d4, 2u, a->d3 >> 14) & 0x7FFF;
  a->d3 = mad24(a->d3, 2u, a->d2 >> 14) & 0x7FFF;
  a->d2 = mad24(a->d2, 2u, a->d1 >> 14) & 0x7FFF;
  a->d1 = mad24(a->d1, 2u, a->d0 >> 14) & 0x7FFF;
  a->d0 = (a->d0 << 1u) & 0x7FFF;
}

#if defined USE_DP
void div_180_90_d(int90_v * const res, const uint qhi, const int90_v n, const double_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
                  MODBASECASE_PAR_DEF
)/* res = q / n (integer division)
    during function entry, qhi contains the upper 30 bits of an 180-bit-value. The remaining bits are zero implicitly.
    this is not a vector, as the first value is the same for all FCs*/
    // do 2*45 bit reductions using double: should be sufficient for 90 bits (and 86 anyways)
{
  __private double_v qf;
  __private double   qf_1;   // for the first conversion which does not need vectors yet

  __private ulong_v qi;
  __private uint_v qil, qim, qih;
  __private int180_v nn, q;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#0: q=%x:<150x0>, n=%x:%x:%x:%x:%x:%x, nf=%#G\n",
        qhi, V(n.d5), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(nf));
#endif

/********** Step 1, Offset 2^67 (4*15 + 7) **********/
//  qf_1 = convert_float(qhi) * 4294967296.0f; // no vector yet, saving a few conversions!
//  qf_1 = qf_1 * 32768.0f * 64.0f;
//  qf_1 = convert_float(qhi) * 9007199254740992.0f; // no vector yet, saving a few conversions! 9007199254740992=4294967296*32768*64, which the compiler does not combine automatically
  qf_1 = convert_double(qhi) * 40564819207303340847894502572032.0;

  qi=CONVERT_ULONG_V(qf_1*nf);  // vectorize just here

  MODBASECASE_QI_ERROR(1L<<46, 1, qi, 0);  // qi here is about 45 bits

  qih = res->d5 = CONVERT_UINT_V(qi >> 30);  // PERF: amd_bitalign ?
  qim = res->d4 = (CONVERT_UINT_V(qi) >> 15) & 0x7FFF;
  qil = res->d3 = CONVERT_UINT_V(qi      ) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1: qf=%#G, nf=%#G, *=%#G, qi=%lld=0x%llx, res=%x:%x:%x:..:..:..\n",
                                 qf_1, V(nf), qf_1*V(nf), V(qi), V(qi), V(res->d5), V(res->d4), V(res->d3));
#endif

  /*******************************************************/

// nn = n * qi
  nn.d3  = mul24(n.d0, qil);
  nn.d4  = mad24(n.d0, qim, nn.d3 >> 15);
  nn.d4  = mad24(n.d1, qil, nn.d4);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1.1: nn=..:..:..:..:..:..:%x:%x:..:..:..\n",
        V(nn.d4), V(nn.d3));
#endif

  nn.d5  = mad24(n.d0, qih, nn.d4 >> 15);
  nn.d5  = mad24(n.d1, qim, nn.d5);
  nn.d5  = mad24(n.d2, qil, nn.d5);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1.2: nn=..:..:..:..:..:%x:%x:%x:...\n",
        V(nn.d5), V(nn.d4), V(nn.d3));
#endif

  nn.d6  = mad24(n.d1, qih, nn.d5 >> 15);
  nn.d6  = mad24(n.d2, qim, nn.d6);
  nn.d6  = mad24(n.d3, qil, nn.d6);
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1.3: nn=..:..:..:..:%x:%x:%x:%x:...\n",
        V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif

  nn.d7  = mad24(n.d2, qih, nn.d6 >> 15);
  nn.d7  = mad24(n.d3, qim, nn.d7);
  nn.d7  = mad24(n.d4, qil, nn.d7);
  nn.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1.4: nn=..:..:..:%x:%x:%x:%x:%x:...\n",
        V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif

  nn.d8  = mad24(n.d3, qih, nn.d7 >> 15);
  nn.d8  = mad24(n.d4, qim, nn.d8);
  nn.d8  = mad24(n.d5, qil, nn.d8);
  nn.d7 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1.5: nn=..:..:%x:%x:%x:%x:%x:%x:...\n",
        V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif

#if defined CHECKS_MODBASECASE || (TRACE_KERNEL > 3)
  nn.d9  = mad24(n.d4, qih, nn.d8 >> 15);
  nn.d9  = mad24(n.d5, qim, nn.d9);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1.6: nn=..:%x:%x:%x:%x:%x:%x:%x:...\n",
        V(nn.d9), V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif
  nn.da  = mad24(n.d5, qih, nn.d9 >> 15);  // can be up to 30 bits, just as the input qhi
  nn.d9 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1.7: nn=..:%x:%x:%x:%x:%x:%x:%x:%x:...\n",
        V(nn.da), V(nn.d9), V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif
#endif
  nn.d8 &= 0x7FFF;

//  q = q - nn, but upon function entry, qhi contains all the bits for db:da. All bits below are zero.
  q.d3 = (-nn.d3) & 0x7FFF;
  q.d4 = (-nn.d4 + AS_UINT_V((nn.d3 > 0)));
  q.d5 = (-nn.d5 + AS_UINT_V((q.d4 > 0)));
  q.d6 = (-nn.d6 + AS_UINT_V((q.d5 > 0)));
  q.d7 = (-nn.d7 + AS_UINT_V((q.d6 > 0)));
  q.d8 = (-nn.d8 + AS_UINT_V((q.d7 > 0)));

#if defined CHECKS_MODBASECASE || (TRACE_KERNEL > 2)
  q.d9 = (-nn.d9 + AS_UINT_V((q.d8 > 0)));
  q.da = qhi - nn.da + AS_UINT_V((q.d9 > 0));
  q.d9 &= 0x7FFF;
  q.da &= 0x7FFF;
#endif

  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
  q.d7 &= 0x7FFF;
  q.d8 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#1.8: q=..:%x:%x!%x:%x:%x:%x:%x:%x:..:..\n",
        V(q.da), V(q.d9), V(q.d8), V(q.d7), V(q.d6), V(q.d5), V(q.d4), V(q.d3));
#endif
  MODBASECASE_NONZERO_ERROR(q.da, 2, 10, 1);
  MODBASECASE_NONZERO_ERROR(q.d9, 2, 9, 2);

  /********** Step 2, Offset 2^30 (2*15 + 0) **********/

  qf= CONVERT_DOUBLE_V(mad24(q.d8, 32768u, q.d7));
  qf= qf * 1073741824.0f + CONVERT_DOUBLE_V(mad24(q.d6, 32768u, q.d5));
  qf= qf * 1073741824.0f + CONVERT_DOUBLE_V(mad24(q.d4, 32768u, q.d3));
  qf*= 35184372088832.0;

  qi=CONVERT_ULONG_V(qf*nf);

  MODBASECASE_QI_ERROR(1L<<46, 2, qi, 3);

  res->d2  = CONVERT_UINT_V(qi >> 30);
  res->d1  = (CONVERT_UINT_V(qi) >> 15) & 0x7FFF;
  res->d0  = CONVERT_UINT_V(qi) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_180_90_d#2: qf=%#G, nf=%#G, *=%#G, qi=%lld=0x%llx, res=%x:%x:%x:%x:%x:%x\n",
                                 V(qf), V(nf), V(qf)*V(nf), V(qi), V(qi), V(res->d5), V(res->d4), V(res->d3), V(res->d2), V(res->d1), V(res->d0));
#endif

  /*******************************************************/
  // skip the last part - it will change the result by one at most - we can live with a result that is off by one
}
#else
// no support for doubles
void div_180_90(int90_v * const res, const uint qhi, const int90_v n, const float_v nf
#if (TRACE_KERNEL > 1)
                  , const uint tid
#endif
                  MODBASECASE_PAR_DEF
)
/* res = q / n (integer division)
    during function entry, qhi contains the upper 30 bits of an 180-bit-value. The remaining bits are zero implicitly.
    this is not a vector, as the first value is the same for all FCs*/
    // try with 4 * 23 bit reductions: should be sufficient for 90 bits (and 86 anyways)
{
  __private float_v qf;
  __private float   qf_1;   // for the first conversion which does not need vectors yet

  __private uint_v qi, qil, qih;
  __private int180_v nn, q;  // PERF: reduce register usage by always using nn.d0-nn.d6 instead of shifting ?

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#0: q=%x:<150x0>, n=%x:%x:%x:%x:%x:%x, nf=%#G\n",
        qhi, V(n.d5), V(n.d4), V(n.d3), V(n.d2), V(n.d1), V(n.d0), V(nf));
#endif

/********** Step 1, Offset 2^67 (4*15 + 7) **********/
//  qf_1 = convert_float(qhi) * 4294967296.0f; // no vector yet, saving a few conversions!
//  qf_1 = qf_1 * 32768.0f * 64.0f;
  qf_1 = convert_float(qhi) * 9007199254740992.0f; // no vector yet, saving a few conversions! 9007199254740992=4294967296*32768*64, which the compiler does not combine automatically

  qi=CONVERT_UINT_V(qf_1*nf);  // vectorize just here

  MODBASECASE_QI_ERROR(1<<24, 1, qi, 0);  // qi here is about 23 bits

  res->d5 = (qi >> 8);
  res->d4 = (qi << 7) & 0x7FFF;
  qil = qi & 0x7FFF;
  qih = (qi >> 15);
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:..:..:..:..\n",
                                 qf_1, V(nf), qf_1*V(nf), V(qi), V(qi), V(res->d5), V(res->d4));
#endif

  /*******************************************************/

// nn = n * qi
  nn.d4  = mul24(n.d0, qil);
  nn.d5  = mad24(n.d0, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1.1: nn=..:..:..:..:..:%x:%x:..:..:..\n",
        V(nn.d5), V(nn.d4));
#endif

  nn.d5  = mad24(n.d1, qil, nn.d5);
  nn.d6  = mad24(n.d1, qih, nn.d5 >> 15);
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1.2: nn=..:..:..:..:%x:%x:%x:...\n",
        V(nn.d6), V(nn.d5), V(nn.d4));
#endif

  nn.d6  = mad24(n.d2, qil, nn.d6);
  nn.d7  = mad24(n.d2, qih, nn.d6 >> 15);
  nn.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1.3: nn=..:..:..:%x:%x:%x:%x:...\n",
        V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4));
#endif

  nn.d7  = mad24(n.d3, qil, nn.d7);
  nn.d8  = mad24(n.d3, qih, nn.d7 >> 15);
  nn.d7 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1.4: nn=..:..:%x:%x:%x:%x:%x:...\n",
        V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4));
#endif
  nn.d8  = mad24(n.d4, qil, nn.d8);
  nn.d9  = mad24(n.d4, qih, nn.d8 >> 15);
  nn.d8 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1.5: nn=..:%x:%x:%x:%x:%x:%x:...\n",
        V(nn.d9), V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4));
#endif
  nn.d9  = mad24(n.d5, qil, nn.d9);
  nn.da  = mad24(n.d5, qih, nn.d9 >> 15);
  nn.d9 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1.6: nn=..:%x:%x:%x:%x:%x:%x:%x:...\n",
        V(nn.da), V(nn.d9), V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4));
#endif

// now shift-left 7 bits  PERF: would that still fit into qi to avoid the long shift?
#ifdef CHECKS_MODBASECASE
  nn.db  = nn.da >> 8;  // PERF: not needed as it will be gone anyway after sub
#endif
  nn.da  = mad24(nn.da & 0xFF, 128u, nn.d9 >> 8);
  nn.d9  = mad24(nn.d9 & 0xFF, 128u, nn.d8 >> 8);
  nn.d8  = mad24(nn.d8 & 0xFF, 128u, nn.d7 >> 8);
  nn.d7  = mad24(nn.d7 & 0xFF, 128u, nn.d6 >> 8);
  nn.d6  = mad24(nn.d6 & 0xFF, 128u, nn.d5 >> 8);
  nn.d5  = mad24(nn.d5 & 0xFF, 128u, nn.d4 >> 8);
  nn.d4  = (nn.d4 & 0x3FF) << 7;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1.7: nn=%x!%x:%x:%x:%x:%x:%x:%x:..:..:..\n",
        V(nn.db), V(nn.da), V(nn.d9), V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4));
#endif

//  q = q - nn, but upon function entry, qhi contains all the bits for db:da. All bits below are zero.
  q.d4 = (-nn.d4) & 0x7FFF;
  q.d5 = (-nn.d5 + AS_UINT_V((nn.d4 > 0)));
  q.d6 = (-nn.d6 + AS_UINT_V((q.d5 > 0)));
  q.d7 = (-nn.d7 + AS_UINT_V((q.d6 > 0)));
  q.d8 = (-nn.d8 + AS_UINT_V((q.d7 > 0)));
  q.d9 = (-nn.d9 + AS_UINT_V((q.d8 > 0)));
  q.da = (qhi & 0x7FFF) - nn.da + AS_UINT_V((q.d9 > 0));
#ifdef CHECKS_MODBASECASE
  q.db = (qhi >> 15)    - nn.db + AS_UINT_V((q.da > 0x7FFF)); // PERF: not needed: should be zero anyway
  q.db &= 0x7FFF;
#endif
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
  q.d7 &= 0x7FFF;
  q.d8 &= 0x7FFF;
  q.d9 &= 0x7FFF;
  q.da &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#1.8: q=%x!%x:%x:%x:%x:%x:%x:%x:..:..\n",
        V(q.db), V(q.da), V(q.d9), V(q.d8), V(q.d7), V(q.d6), V(q.d5), V(q.d4));
#endif
  MODBASECASE_NONZERO_ERROR(q.db, 1, 11, 1);

  /********** Step 2, Offset 2^45 (3*15 + 0) **********/

  qf= CONVERT_FLOAT_V(mad24(q.da, 32768u, q.d9));
  qf= qf * 1073741824.0f + CONVERT_FLOAT_V(mad24(q.d8, 32768u, q.d7));
  qf*= 4294967296.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<23, 2, qi, 2); // here, we need 2^23 ...

  qih = (qi >> 15);
  qil = qi & 0x7FFF;
  res->d4 += (qi >> 17);
  res->d3  = (qi >>  2) &0x7FFF;
  res->d2  = (qi << 13) &0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:..:..\n",
                                 V(qf), V(nf), V(qf)*V(nf), V(qi), V(qi), V(res->d5), V(res->d4), V(res->d3), V(res->d2));
#endif

  /*******************************************************/

// nn = n * qi
  nn.d3  = mul24(n.d0, qil);
  nn.d4  = mad24(n.d0, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2.1: nn=..:..:..:..:%x:%x:..:..:..\n",
        V(nn.d4), V(nn.d3));
#endif

  nn.d4  = mad24(n.d1, qil, nn.d4);
  nn.d5  = mad24(n.d1, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2.2: nn=..:..:..:%x:%x:%x:..:..:..\n",
        V(nn.d5), V(nn.d4), V(nn.d3));
#endif

  nn.d5  = mad24(n.d2, qil, nn.d5);
  nn.d6  = mad24(n.d2, qih, nn.d5 >> 15);
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2.3: nn=..:..:%x:%x:%x:%x:..:..:..\n",
        V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif

  nn.d6  = mad24(n.d3, qil, nn.d6);
  nn.d7  = mad24(n.d3, qih, nn.d6 >> 15);
  nn.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2.4: nn=..:%x:%x:%x:%x:%x:..:..:..\n",
        V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif

  nn.d7  = mad24(n.d4, qil, nn.d7);
  nn.d8  = mad24(n.d4, qih, nn.d7 >> 15);
  nn.d7 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2.5: nn=..:%x:%x:%x:%x:%x:%x:..:..:..\n",
        V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif

  nn.d8  = mad24(n.d5, qil, nn.d8);
  nn.d9  = mad24(n.d5, qih, nn.d8 >> 15);
  nn.d8 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2.6: nn=..:..:%x:%x:%x:%x:%x:%x:%x:..:..:..\n",
        V(nn.d9), V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3));
#endif
#ifdef CHECKS_MODBASECASE
  nn.da  = nn.d9 >> 15;
  nn.d9 &= 0x7FFF;
#endif

  // shift-right 2 bits
  nn.d2 = (nn.d3 << 13) & 0x7FFF;
  nn.d3 = mad24(nn.d4 & 3, 8192u, nn.d3 >> 2);
  nn.d4 = mad24(nn.d5 & 3, 8192u, nn.d4 >> 2);
  nn.d5 = mad24(nn.d6 & 3, 8192u, nn.d5 >> 2);
  nn.d6 = mad24(nn.d7 & 3, 8192u, nn.d6 >> 2);
  nn.d7 = mad24(nn.d8 & 3, 8192u, nn.d7 >> 2);
  nn.d8 = mad24(nn.d9 & 3, 8192u, nn.d8 >> 2);
#ifdef CHECKS_MODBASECASE
  nn.d9 = mad24(nn.da & 3, 8192u, nn.d9 >> 2);
  nn.da = nn.da >> 2;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2.8: nn=..:%x:%x!%x:%x:%x:%x:%x:%x:%x:..:..\n",
        V(nn.da), V(nn.d9), V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2));
#endif
  //  q = q - nn; q.d2 and q.d3 are still 0
  q.d2 = (-nn.d2) & 0x7FFF;
  q.d3 = (-nn.d3 + AS_UINT_V((nn.d2 > 0)));
  q.d4 = q.d4 - nn.d4 + AS_UINT_V((q.d3 > 0x7FFF));
  q.d5 = q.d5 - nn.d5 + AS_UINT_V((q.d4 > 0x7FFF));
  q.d6 = q.d6 - nn.d6 + AS_UINT_V((q.d5 > 0x7FFF));
  q.d7 = q.d7 - nn.d7 + AS_UINT_V((q.d6 > 0x7FFF));
  q.d8 = q.d8 - nn.d8 + AS_UINT_V((q.d7 > 0x7FFF));
#ifdef CHECKS_MODBASECASE
  q.d9 = q.d9 - nn.d9 + AS_UINT_V((q.d8 > 0x7FFF)); // PERF: not needed: should be zero anyway
  q.d9 &= 0x7FFF;
#endif
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
  q.d7 &= 0x7FFF;
  q.d8 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#2.9: q=..:..:%x:%x:%x:%x:%x:%x:%x:..:..\n",
        V(q.d9), V(q.d8), V(q.d7), V(q.d6), V(q.d5), V(q.d4), V(q.d3));
#endif

  MODBASECASE_NONZERO_ERROR(q.da, 2, 10, 3);
  MODBASECASE_NONZERO_ERROR(q.d9, 2, 9, 4);

  /********** Step 3, Offset 2^22 (1*15 + 7) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d8, 32768u, q.d7));
  qf= qf * 1073741824.0f + CONVERT_FLOAT_V(mad24(q.d6, 32768u, q.d5));
  qf*= 8388608.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<23, 3, qi, 5);

  qih = (qi >> 15);
  qil = qi & 0x7FFF;
  res->d2 += (qi >> 8);
  res->d1  = (qi << 7) & 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:%x:..\n",
                                 V(qf), V(nf), V(qf)*V(nf), V(qi), V(qi), V(res->d5), V(res->d4), V(res->d3), V(res->d2), V(res->d1));
#endif

  /*******************************************************/

// nn = n * qi
  nn.d1  = mul24(n.d0, qil);
  nn.d2  = mad24(n.d0, qih, nn.d1 >> 15);
  nn.d1 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3.1: nn=..:..:..:..:%x:%x:..\n",
        V(nn.d2), V(nn.d1));
#endif

  nn.d2  = mad24(n.d1, qil, nn.d2);
  nn.d3  = mad24(n.d1, qih, nn.d2 >> 15);
  nn.d2 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3.2: nn=..:..:..:%x:%x:%x:..\n",
        V(nn.d3), V(nn.d2), V(nn.d1));
#endif

  nn.d3  = mad24(n.d2, qil, nn.d3);
  nn.d4  = mad24(n.d2, qih, nn.d3 >> 15);
  nn.d3 &= 0x7FFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3.3: nn=..:..:%x:%x:%x:%x:..\n",
        V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1));
#endif

  nn.d4  = mad24(n.d3, qil, nn.d4);
  nn.d5  = mad24(n.d3, qih, nn.d4 >> 15);
  nn.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3.4: nn=..:%x:%x:%x:%x:%x:..\n",
        V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1));
#endif

  nn.d5  = mad24(n.d4, qil, nn.d5);
  nn.d6  = mad24(n.d4, qih, nn.d5 >> 15);
  nn.d5 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3.5: nn=..:%x:%x:%x:%x:%x:%x:..\n",
        V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1));
#endif

  nn.d6  = mad24(n.d5, qil, nn.d6);
  nn.d7  = mad24(n.d5, qih, nn.d6 >> 15);
  nn.d6 &= 0x7FFF;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3.6: nn=..:%x:%x:%x:%x:%x:%x:%x:..\n",
        V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1));
#endif

// nn.d7 also contains the nn.d8 bits, will be distributed by the shifting below

// now shift-left 7 bits
#ifdef CHECKS_MODBASECASE
  nn.d8  = nn.d7 >> 8;  // PERF: not needed as it will be gone anyway after sub
#endif
  nn.d7  = mad24(nn.d7 & 0xFF, 128u, nn.d6 >> 8);
  nn.d6  = mad24(nn.d6 & 0xFF, 128u, nn.d5 >> 8);
  nn.d5  = mad24(nn.d5 & 0xFF, 128u, nn.d4 >> 8);
  nn.d4  = mad24(nn.d4 & 0xFF, 128u, nn.d3 >> 8);
  nn.d3  = mad24(nn.d3 & 0xFF, 128u, nn.d2 >> 8);
  nn.d2  = mad24(nn.d2 & 0xFF, 128u, nn.d1 >> 8);
  nn.d1  = (nn.d1 & 0xFF) << 7;
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3.7: nn=..:..:%x!%x:%x:%x:%x:%x:%x:%x:..\n",
        V(nn.d8), V(nn.d7), V(nn.d6), V(nn.d5), V(nn.d4), V(nn.d3), V(nn.d2), V(nn.d1));
#endif

//  q = q - nn;  q.d1 is still 0
  q.d1 = (-nn.d1) & 0x7FFF;
  q.d2 = q.d2 - nn.d2 + AS_UINT_V((nn.d1 > 0));
  q.d3 = q.d3 - nn.d3 + AS_UINT_V((q.d2 > 0x7FFF));
  q.d4 = q.d4 - nn.d4 + AS_UINT_V((q.d3 > 0x7FFF));
  q.d5 = q.d5 - nn.d5 + AS_UINT_V((q.d4 > 0x7FFF));
  q.d6 = q.d6 - nn.d6 + AS_UINT_V((q.d5 > 0x7FFF));
  q.d7 = q.d7 - nn.d7 + AS_UINT_V((q.d6 > 0x7FFF));
#ifdef CHECKS_MODBASECASE
  q.d8 = q.d8 - nn.d8 + AS_UINT_V((q.d7 > 0x7FFF)); // PERF: not needed: should be zero anyway
  q.d8 &= 0x7FFF;
#endif
  q.d2 &= 0x7FFF;
  q.d3 &= 0x7FFF;
  q.d4 &= 0x7FFF;
  q.d5 &= 0x7FFF;
  q.d6 &= 0x7FFF;
  q.d7 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"div_180_90#3.8: q=..:%x!%x:%x:%x:%x:%x:%x:%x:..\n",
        V(q.d8), V(q.d7), V(q.d6), V(q.d5), V(q.d4), V(q.d3), V(q.d2), V(q.d1));
#endif

  MODBASECASE_NONZERO_ERROR(q.d8, 3, 8, 6);

  /********** Step 4, Offset 2^0 (0*15 + 0) **********/

  qf= CONVERT_FLOAT_V(mad24(q.d7, 32768u, q.d6));
  qf= qf * 1073741824.0f + CONVERT_FLOAT_V(mad24(q.d5, 32768u, q.d4));
  qf*= 1073741824.0f;

  qi=CONVERT_UINT_V(qf*nf);

  MODBASECASE_QI_ERROR(1<<23, 4, qi, 7);

  qil = qi & 0x7FFF;
  qih = (qi >> 15);
  res->d1 += qih;
  res->d0 = qil;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"div_180_90#4: qf=%#G, nf=%#G, *=%#G, qi=%d=0x%x, res=%x:%x:%x:%x:%x:%x\n",
                                 V(qf), V(nf), V(qf)*V(nf), V(qi), V(qi), V(res->d5), V(res->d4), V(res->d3), V(res->d2), V(res->d1), V(res->d0));
#endif

  // skip the last part - it will change the result by one at most - we can live with a result that is off by one
  // but need to handle outstanding carries instead
  res->d2 += res->d1 >> 15;
  res->d1 &= 0x7FFF;
  res->d3 += res->d2 >> 15;
  res->d2 &= 0x7FFF;
  res->d4 += res->d3 >> 15;
  res->d3 &= 0x7FFF;
  res->d5 += res->d4 >> 15;
  res->d4 &= 0x7FFF;
}
#endif

void check_barrett15_82(uint shifter, const int90_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int90_v a, u;
  __private int180_v b, tmp180;
  __private int90_v tmp90;
  __private float_v ff;
#if defined USE_DP
  __private double_v ffd;
#endif
  __private uint tmp, bit_max_bot, bit_max_mult;
  __private int180_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5, b_in.s6, b_in.s7};

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82: bb=%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x, bit_max65=%d\n",
        bb.db, bb.da, bb.d9, bb.d8, bb.d7, bb.d6, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, bit_max65);
#endif

  // ff = f as float, needed only for the final mod_simple
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d5, 32768u, f.d4));
  ff= ff * 1073741824.0f+ CONVERT_FLOAT_RTP_V(mad24(f.d3, 32768u, f.d2));
  ff = as_float(0x3f7ffffc) / ff;

  tmp = 1 << (bit_max65+4);	// tmp180 = 2^(89 + bits in f)

#if defined USE_DP
  // ffd = f as double, needed in div_180_90_d).
  ffd= CONVERT_DOUBLE_RTP_V(mad24(f.d5, 32768u, f.d4));
  ffd= ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d3, 32768u, f.d2));
  ffd= ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d1, 32768u, f.d0));
  ffd = as_double(0x3feffffffffffffdL) / ffd;     // should be a bit less than 1.0

  div_180_90_d(&u, tmp, f, ffd
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82: u(d)=%x:%x:%x:%x:%x:%x, ffd=%G\n",
        V(u.d5), V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif

#else

  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper 2 parts (30 bits) of a 180-bit value. The lower 150 bits are all zero implicitly

  div_180_90(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82: u=%x:%x:%x:%x:%x:%x, ff=%G\n",
        V(u.d5), V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  if (bit_max65 > 10)  // need to distiguish how far to shift; the same branch will be taken by all threads
  {
    //bit_max is 76 .. 89
    bit_max_bot  = bit_max65-11;
    bit_max_mult = 1 << (26-bit_max65);

    // a.d<n> = bb.d<n+5> >> bit_max_bot + bb.d<n+6> << top_bit_max

    //PERF: min limit of bb? bit_max > 75 ==> bb > 2^150 ==> d0..d9=0
    a.d0 = mad24(bb.d6, bit_max_mult, (bb.d5 >> bit_max_bot))&0x7FFF;			// a = floor(b / 2 ^ (bits_in_f - 1))
    a.d1 = mad24(bb.d7, bit_max_mult, (bb.d6 >> bit_max_bot))&0x7FFF;
    a.d2 = mad24(bb.d8, bit_max_mult, (bb.d7 >> bit_max_bot))&0x7FFF;
    a.d3 = mad24(bb.d9, bit_max_mult, (bb.d8 >> bit_max_bot))&0x7FFF;
    a.d4 = mad24(bb.da, bit_max_mult, (bb.d9 >> bit_max_bot))&0x7FFF;
    a.d5 = mad24(bb.db, bit_max_mult, (bb.da >> bit_max_bot));
  }
  else
  {
    //bit_max is 61 .. 75
    bit_max_bot  = bit_max65+4;
    bit_max_mult = 1 << (11-bit_max65);

    // a.d<n> = bb.d<n+4> >> bit_max_bot + bb.d<n+5> << top_bit_max

    //PERF: min limit of bb? bit_max >= 60 ==> bb >= 2^120 ==> d0..d7=0
    a.d0 = mad24(bb.d5, bit_max_mult, (bb.d4 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(bb.d6, bit_max_mult, (bb.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(bb.d7, bit_max_mult, (bb.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(bb.d8, bit_max_mult, (bb.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(bb.d9, bit_max_mult, (bb.d8 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
    a.d5 = mad24(bb.da, bit_max_mult, (bb.d9 >> bit_max_bot));		       	// a = b / (2^bit_max)
  }
      // PERF: could be no_low_5
  mul_90_180_no_low5(&tmp180, a, u); // tmp180 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (89 + bits_in_f) / f)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82: a=%x:%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:%x:...\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp180.db), V(tmp180.da), V(tmp180.d9), V(tmp180.d8), V(tmp180.d7), V(tmp180.d6), V(tmp180.d5));
#endif
  a.d0 = tmp180.d6;		             	// a = tmp180 / 2^90, which is b / f
  a.d1 = tmp180.d7;
  a.d2 = tmp180.d8;
  a.d3 = tmp180.d9;
  a.d4 = tmp180.da;
  a.d5 = tmp180.db;

  mul_90(&tmp90, a, f);							// tmp90 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82: a=%x:%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x:%x (tmp)\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp90.d5), V(tmp90.d4), V(tmp90.d3), V(tmp90.d2), V(tmp90.d1), V(tmp90.d0));
#endif
    // bb.d0-bb.d3 are all 0
  a.d0 = (-tmp90.d0) & 0x7FFF;
  a.d1 = (-tmp90.d1 + AS_UINT_V((a.d0 > 0)  ));
  a.d2 = (-tmp90.d2 + AS_UINT_V((a.d1 > 0x7FFF)  ));
  a.d3 = (-tmp90.d3 + AS_UINT_V((a.d2 > 0x7FFF)  ));
  a.d4 = (bb.d4-tmp90.d4 + AS_UINT_V((a.d3 > 0x7FFF)  ));
  a.d5 = (bb.d5-tmp90.d5 + AS_UINT_V((a.d4 > 0x7FFF)  ));
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;
  a.d5 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82: b=%x:%x:0:0:0:0 - tmp = %x:%x:%x:%x:%x:%x (a)\n",
        bb.d5, bb.d4, V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif
    ///
    ///// here it starts to become different between the 3 6x15bit kernels
    ///

  while(shifter)
  {
    square_90_180(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.db), V(b.da), V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    if (bit_max65 > 10)  // need to distiguish how far to shift
    {
      a.d0 = mad24(b.d6, bit_max_mult, (b.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d1 = mad24(b.d7, bit_max_mult, (b.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d2 = mad24(b.d8, bit_max_mult, (b.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d3 = mad24(b.d9, bit_max_mult, (b.d8 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d4 = mad24(b.da, bit_max_mult, (b.d9 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
      a.d5 = mad24(b.db, bit_max_mult, (b.da >> bit_max_bot));		       	// a = b / (2^bit_max)
    }
    else
    {
      a.d0 = mad24(b.d5, bit_max_mult, (b.d4 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d1 = mad24(b.d6, bit_max_mult, (b.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d2 = mad24(b.d7, bit_max_mult, (b.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d3 = mad24(b.d8, bit_max_mult, (b.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d4 = mad24(b.d9, bit_max_mult, (b.d8 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
      a.d5 = mad24(b.da, bit_max_mult, (b.d9 >> bit_max_bot));		       	// a = b / (2^bit_max)
    }
      // PERF: could be no_low_5
    mul_90_180_no_low5(&tmp180, a, u); // tmp180 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (89 + bits_in_f) / f)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loopl: a=%x:%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:%x:...\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp180.db), V(tmp180.da), V(tmp180.d9), V(tmp180.d8), V(tmp180.d7), V(tmp180.d6), V(tmp180.d5));
#endif

    a.d0 = tmp180.d6;		             	// a = tmp180 / 2^90, which is b / f
    a.d1 = tmp180.d7;
    a.d2 = tmp180.d8;
    a.d3 = tmp180.d9;
    a.d4 = tmp180.da;
    a.d5 = tmp180.db;

    mul_90(&tmp90, a, f);						// tmp90 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x:%x (tmp)\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp90.d5), V(tmp90.d4), V(tmp90.d3), V(tmp90.d2), V(tmp90.d1), V(tmp90.d0));
#endif
    a.d0 = (b.d0 - tmp90.d0) & 0x7FFF;
    a.d1 = (b.d1 - tmp90.d1 + AS_UINT_V((a.d0 > b.d0)  ));
    a.d2 = (b.d2 - tmp90.d2 + AS_UINT_V((a.d1 > b.d1)  ));
    a.d3 = (b.d3 - tmp90.d3 + AS_UINT_V((a.d2 > b.d2)  ));
    a.d4 = (b.d4 - tmp90.d4 + AS_UINT_V((a.d3 > b.d3)  ));
    a.d5 = (b.d5 - tmp90.d5 + AS_UINT_V((a.d4 > b.d4)  ));
    a.d1 &= 0x7FFF;
    a.d2 &= 0x7FFF;
    a.d3 &= 0x7FFF;
    a.d4 &= 0x7FFF;
    a.d5 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=...:%x:%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x:%x (a)\n",
        V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0), V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

    if(shifter&0x80000000)shl_90(&a);					// "optional multiply by 2" in Prime 95 documentation

    shifter+=shifter;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, a= %x:%x:%x:%x:%x:%x \n",
        shifter, V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0) );
#endif
  }

  mod_simple_even_90_and_check_big_factor90(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                                      , bit_max65, 10, modbasecase_debug
#endif
                                      );
}

void check_barrett15_83(uint shifter, const int90_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int90_v a, u;
  __private int180_v b, tmp180;
  __private int90_v tmp90;
  __private float_v ff;
#if defined USE_DP
  __private double_v ffd;
#endif
  __private uint tmp, bit_max_bot, bit_max_mult;
  __private int180_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5, b_in.s6, b_in.s7};

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83: bb=%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x, bit_max65=%d\n",
        bb.db, bb.da, bb.d9, bb.d8, bb.d7, bb.d6, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, bit_max65);
#endif

  // ff = f as float, needed only for the final mod_simple
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d5, 32768u, f.d4));
  ff= ff * 1073741824.0f+ CONVERT_FLOAT_RTP_V(mad24(f.d3, 32768u, f.d2));
  ff = as_float(0x3f7ffffc) / ff;

  tmp = 1 << (bit_max65+4);	// tmp180 = 2^(89 + bits in f)

#if defined USE_DP
  // ffd = f as double, needed in div_180_90_d).
  ffd= CONVERT_DOUBLE_RTP_V(mad24(f.d5, 32768u, f.d4));
  ffd= ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d3, 32768u, f.d2));
  ffd= ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d1, 32768u, f.d0));
  ffd = as_double(0x3feffffffffffffdL) / ffd;     // should be a bit less than 1.0

  div_180_90_d(&u, tmp, f, ffd
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83: u(d)=%x:%x:%x:%x:%x:%x, ffd=%G\n",
        V(u.d5), V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ffd));
#endif
#else

  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper 2 parts (30 bits) of a 180-bit value. The lower 150 bits are all zero implicitly

  div_180_90(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83: u=%x:%x:%x:%x:%x:%x, ff=%G\n",
        V(u.d5), V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif
#endif

  if (bit_max65 > 10)  // need to distiguish how far to shift; the same branch will be taken by all threads
  {
    //bit_max is 76 .. 89
    bit_max_bot  = bit_max65-11;
    bit_max_mult = 1 << (26-bit_max65);

    // a.d<n> = bb.d<n+5> >> bit_max_bot + bb.d<n+6> << top_bit_max

    //PERF: min limit of bb? bit_max > 75 ==> bb > 2^150 ==> d0..d9=0
    a.d0 = mad24(bb.d6, bit_max_mult, (bb.d5 >> bit_max_bot))&0x7FFF;			// a = floor(b / 2 ^ (bits_in_f - 1))
    a.d1 = mad24(bb.d7, bit_max_mult, (bb.d6 >> bit_max_bot))&0x7FFF;
    a.d2 = mad24(bb.d8, bit_max_mult, (bb.d7 >> bit_max_bot))&0x7FFF;
    a.d3 = mad24(bb.d9, bit_max_mult, (bb.d8 >> bit_max_bot))&0x7FFF;
    a.d4 = mad24(bb.da, bit_max_mult, (bb.d9 >> bit_max_bot))&0x7FFF;
    a.d5 = mad24(bb.db, bit_max_mult, (bb.da >> bit_max_bot));
  }
  else
  {
    //bit_max is 61 .. 75
    bit_max_bot  = bit_max65+4;
    bit_max_mult = 1 << (11-bit_max65);

    // a.d<n> = bb.d<n+4> >> bit_max_bot + bb.d<n+5> << top_bit_max

    //PERF: min limit of bb? bit_max >= 60 ==> bb >= 2^120 ==> d0..d7=0
    a.d0 = mad24(bb.d5, bit_max_mult, (bb.d4 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(bb.d6, bit_max_mult, (bb.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(bb.d7, bit_max_mult, (bb.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(bb.d8, bit_max_mult, (bb.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(bb.d9, bit_max_mult, (bb.d8 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
    a.d5 = mad24(bb.da, bit_max_mult, (bb.d9 >> bit_max_bot));		       	// a = b / (2^bit_max)
  }
      // PERF: could be no_low_5
  mul_90_180_no_low5(&tmp180, a, u); // tmp180 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (89 + bits_in_f) / f)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83: a=%x:%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:%x:...\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp180.db), V(tmp180.da), V(tmp180.d9), V(tmp180.d8), V(tmp180.d7), V(tmp180.d6), V(tmp180.d5));
#endif

  a.d0 = tmp180.d6;		             	// a = tmp180 / 2^90, which is b / f
  a.d1 = tmp180.d7;
  a.d2 = tmp180.d8;
  a.d3 = tmp180.d9;
  a.d4 = tmp180.da;
  a.d5 = tmp180.db;

  mul_90(&tmp90, a, f);							// tmp90 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83: a=%x:%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x:%x (tmp)\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp90.d5), V(tmp90.d4), V(tmp90.d3), V(tmp90.d2), V(tmp90.d1), V(tmp90.d0));
#endif
  // bb.d0-bb.d3 are 0
  a.d0 = (-tmp90.d0) & 0x7FFF;
  a.d1 = (-tmp90.d1 + AS_UINT_V((a.d0 > 0)  ));
  a.d2 = (-tmp90.d2 + AS_UINT_V((a.d1 > 0x7FFF)  ));
  a.d3 = (-tmp90.d3 + AS_UINT_V((a.d2 > 0x7FFF)  ));
  a.d4 = (bb.d4-tmp90.d4 + AS_UINT_V((a.d3 > 0x7FFF)  ));
  a.d5 = (bb.d5-tmp90.d5 + AS_UINT_V((a.d4 > 0x7FFF)  ));
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;
  a.d5 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83: b=%x:%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x:%x (a)\n",
        bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

  while(shifter)
  {
    square_90_180(&b, a);						// b = a^2
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.db), V(b.da), V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    if(shifter&0x80000000)
    {
      shl_180(&b);					// "optional multiply by 2" in Prime 95 documentation

#if (TRACE_KERNEL > 3)
      if (tid==TRACE_TID) printf((__constant char *)"loop: shl: %x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        V(b.db), V(b.da), V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    }

    if (bit_max65 > 10)  // need to distiguish how far to shift
    {
      a.d0 = mad24(b.d6, bit_max_mult, (b.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d1 = mad24(b.d7, bit_max_mult, (b.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d2 = mad24(b.d8, bit_max_mult, (b.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d3 = mad24(b.d9, bit_max_mult, (b.d8 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d4 = mad24(b.da, bit_max_mult, (b.d9 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
      a.d5 = mad24(b.db, bit_max_mult, (b.da >> bit_max_bot));		       	// a = b / (2^bit_max)
    }
    else
    {
      a.d0 = mad24(b.d5, bit_max_mult, (b.d4 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d1 = mad24(b.d6, bit_max_mult, (b.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d2 = mad24(b.d7, bit_max_mult, (b.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d3 = mad24(b.d8, bit_max_mult, (b.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d4 = mad24(b.d9, bit_max_mult, (b.d8 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
      a.d5 = mad24(b.da, bit_max_mult, (b.d9 >> bit_max_bot));		       	// a = b / (2^bit_max)
    }
      // PERF: could be no_low_5
    mul_90_180_no_low5(&tmp180, a, u); // tmp180 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (89 + bits_in_f) / f)

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:%x:...\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp180.db), V(tmp180.da), V(tmp180.d9), V(tmp180.d8), V(tmp180.d7), V(tmp180.d6), V(tmp180.d5));
#endif
    a.d0 = tmp180.d6;		             	// a = tmp180 / 2^90, which is b / f
    a.d1 = tmp180.d7;
    a.d2 = tmp180.d8;
    a.d3 = tmp180.d9;
    a.d4 = tmp180.da;
    a.d5 = tmp180.db;

    mul_90(&tmp90, a, f);						// tmp90 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x:%x (tmp)\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp90.d5), V(tmp90.d4), V(tmp90.d3), V(tmp90.d2), V(tmp90.d1), V(tmp90.d0));
#endif
    // PERF: faster to compare against 0x7fff instead of b.dx?
    a.d0 = (b.d0 - tmp90.d0) & 0x7FFF;
    a.d1 = (b.d1 - tmp90.d1 + AS_UINT_V((a.d0 > b.d0)  ));
    a.d2 = (b.d2 - tmp90.d2 + AS_UINT_V((a.d1 > b.d1)  ));
    a.d3 = (b.d3 - tmp90.d3 + AS_UINT_V((a.d2 > b.d2)  ));
    a.d4 = (b.d4 - tmp90.d4 + AS_UINT_V((a.d3 > b.d3)  ));
    a.d5 = (b.d5 - tmp90.d5 + AS_UINT_V((a.d4 > b.d4)  ));
    a.d1 &= 0x7FFF;
    a.d2 &= 0x7FFF;
    a.d3 &= 0x7FFF;
    a.d4 &= 0x7FFF;
    a.d5 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x:%x (a)\n",
        V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0), V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

    shifter+=shifter;
  }

  mod_simple_90_and_check_big_factor90(a, f, ff, RES
#ifdef CHECKS_MODBASECASE
                                      , bit_max65, 10, modbasecase_debug
#endif
                                      );
}


void check_barrett15_88(uint shifter, const int90_v f, const uint tid, const uint8 b_in, const int bit_max65, __global uint * restrict RES
                        MODBASECASE_PAR_DEF)
{
  __private int90_v a, u;
  __private int180_v b, tmp180;
  __private int90_v tmp90;
  __private float_v ff;
#if defined USE_DP
  __private double_v ffd;
#endif
  __private uint tmp, bit_max_bot, bit_max_mult;
  __private int180_t bb={0, 0, 0, 0, b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5, b_in.s6, b_in.s7};

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_88: bb=%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x, bit_max65=%d\n",
        bb.db, bb.da, bb.d9, bb.d8, bb.d7, bb.d6, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, bit_max65);
#endif

  // ff = f as float, needed only for the final mod_simple
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d5, 32768u, f.d4));
  ff= ff * 1073741824.0f+ CONVERT_FLOAT_RTP_V(mad24(f.d3, 32768u, f.d2));
  ff = as_float(0x3f7ffffc) / ff;

  tmp = 1 << (bit_max65+4);	// tmp180 = 2^(89 + bits in f)

#if defined USE_DP
  // ffd = f as double, needed in div_180_90_d).
  ffd= CONVERT_DOUBLE_RTP_V(mad24(f.d5, 32768u, f.d4));
  ffd= ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d3, 32768u, f.d2));
  ffd= ffd * 1073741824.0+ CONVERT_DOUBLE_RTP_V(mad24(f.d1, 32768u, f.d0));
  ffd = as_double(0x3feffffffffffffdL) / ffd;     // should be a bit less than 1.0

  div_180_90_d(&u, tmp, f, ffd
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#else

  // PERF: as div is only used here, use all those zeros directly in there
  //       here, no vectorized data is necessary yet: the precalculated "b" value is the same for all
  //       tmp contains the upper 2 parts (30 bits) of a 180-bit value. The lower 150 bits are all zero implicitly

  div_180_90(&u, tmp, f, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
                  MODBASECASE_PAR);						// u = floor(tmp180 / f)
#endif

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_88: u=%x:%x:%x:%x:%x:%x, ff=%G\n",
        V(u.d5), V(u.d4), V(u.d3), V(u.d2), V(u.d1), V(u.d0), V(ff));
#endif

  if (bit_max65 > 10)  // need to distiguish how far to shift; the same branch will be taken by all threads
  {
    //bit_max is 76 .. 89
    bit_max_bot  = bit_max65-11;
    bit_max_mult = 1 << (26-bit_max65);

    // a.d<n> = bb.d<n+5> >> bit_max_bot + bb.d<n+6> << top_bit_max

    //PERF: min limit of bb? bit_max > 75 ==> bb > 2^150 ==> d0..d9=0
    a.d0 = mad24(bb.d6, bit_max_mult, (bb.d5 >> bit_max_bot))&0x7FFF;			// a = floor(b / 2 ^ (bits_in_f - 1))
    a.d1 = mad24(bb.d7, bit_max_mult, (bb.d6 >> bit_max_bot))&0x7FFF;
    a.d2 = mad24(bb.d8, bit_max_mult, (bb.d7 >> bit_max_bot))&0x7FFF;
    a.d3 = mad24(bb.d9, bit_max_mult, (bb.d8 >> bit_max_bot))&0x7FFF;
    a.d4 = mad24(bb.da, bit_max_mult, (bb.d9 >> bit_max_bot))&0x7FFF;
    a.d5 = mad24(bb.db, bit_max_mult, (bb.da >> bit_max_bot));
  }
  else
  {
    //bit_max is 61 .. 75
    bit_max_bot  = bit_max65+4;
    bit_max_mult = 1 << (11-bit_max65);

    // a.d<n> = bb.d<n+4> >> bit_max_bot + bb.d<n+5> << top_bit_max

    //PERF: min limit of bb? bit_max >= 60 ==> bb >= 2^120 ==> d0..d7=0
    a.d0 = mad24(bb.d5, bit_max_mult, (bb.d4 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d1 = mad24(bb.d6, bit_max_mult, (bb.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d2 = mad24(bb.d7, bit_max_mult, (bb.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d3 = mad24(bb.d8, bit_max_mult, (bb.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
    a.d4 = mad24(bb.d9, bit_max_mult, (bb.d8 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
    a.d5 = mad24(bb.da, bit_max_mult, (bb.d9 >> bit_max_bot));		       	// a = b / (2^bit_max)
  }
      // PERF: could be no_low_5
  mul_90_180_no_low5(&tmp180, a, u); // tmp180 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (89 + bits_in_f) / f)
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_88: a=%x:%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:%x:...\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp180.db), V(tmp180.da), V(tmp180.d9), V(tmp180.d8), V(tmp180.d7), V(tmp180.d6), V(tmp180.d5));
#endif

  a.d0 = tmp180.d6;		             	// a = tmp180 / 2^90, which is b / f
  a.d1 = tmp180.d7;
  a.d2 = tmp180.d8;
  a.d3 = tmp180.d9;
  a.d4 = tmp180.da;
  a.d5 = tmp180.db;

  mul_90(&tmp90, a, f);							// tmp90 = quotient * f, we only compute the low 90-bits here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_88: a=%x:%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x:%x (tmp)\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp90.d5), V(tmp90.d4), V(tmp90.d3), V(tmp90.d2), V(tmp90.d1), V(tmp90.d0));
#endif
    // bb.d0-bb.d3 are 0
  a.d0 = (-tmp90.d0) & 0x7FFF;
  a.d1 = (-tmp90.d1 + AS_UINT_V((a.d0 > 0)  ));
  a.d2 = (-tmp90.d2 + AS_UINT_V((a.d1 > 0x7FFF)  ));
  a.d3 = (-tmp90.d3 + AS_UINT_V((a.d2 > 0x7FFF)  ));
  a.d4 = (bb.d4-tmp90.d4 + AS_UINT_V((a.d3 > 0x7FFF)  ));
  a.d5 = (bb.d5-tmp90.d5 + AS_UINT_V((a.d4 > 0x7FFF)  ));
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;
  a.d5 &= 0x7FFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_88: b=%x:%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x:%x (a)\n",
        bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0));
#endif

for(;;)
{
    square_90_180(&b, a);						// b = a^2

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: exp=%.8x, a=%x:%x:%x:%x:%x:%x ^2 = %x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x:%x (b)\n",
        shifter, V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(b.db), V(b.da), V(b.d9), V(b.d8), V(b.d7), V(b.d6), V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0) );
#endif
    if (bit_max65 > 10)  // need to distiguish how far to shift
    {
      a.d0 = mad24(b.d6, bit_max_mult, (b.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d1 = mad24(b.d7, bit_max_mult, (b.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d2 = mad24(b.d8, bit_max_mult, (b.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d3 = mad24(b.d9, bit_max_mult, (b.d8 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d4 = mad24(b.da, bit_max_mult, (b.d9 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
      a.d5 = mad24(b.db, bit_max_mult, (b.da >> bit_max_bot));		       	// a = b / (2^bit_max)
    }
    else
    {
      a.d0 = mad24(b.d5, bit_max_mult, (b.d4 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d1 = mad24(b.d6, bit_max_mult, (b.d5 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d2 = mad24(b.d7, bit_max_mult, (b.d6 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d3 = mad24(b.d8, bit_max_mult, (b.d7 >> bit_max_bot))&0x7FFF;			// a = b / (2^bit_max)
      a.d4 = mad24(b.d9, bit_max_mult, (b.d8 >> bit_max_bot))&0x7FFF;		 	// a = b / (2^bit_max)
      a.d5 = mad24(b.da, bit_max_mult, (b.d9 >> bit_max_bot));		       	// a = b / (2^bit_max)
    }
      // PERF: could be no_low_5
    mul_90_180_no_low5(&tmp180, a, u); // tmp180 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (89 + bits_in_f) / f)
#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x:%x * u = %x:%x:%x:%x:%x:%x:...\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0),
        V(tmp180.db), V(tmp180.da), V(tmp180.d9), V(tmp180.d8), V(tmp180.d7), V(tmp180.d6));
#endif

    a.d0 = tmp180.d6;		             	// a = tmp180 / 2^90, which is b / f
    a.d1 = tmp180.d7;
    a.d2 = tmp180.d8;
    a.d3 = tmp180.d9;
    a.d4 = tmp180.da;
    a.d5 = tmp180.db;

    mul_90(&tmp90, a, f);							// tmp90 = quotient * f, we only compute the low 90-bits here

#if (TRACE_KERNEL > 3)
    if (tid==TRACE_TID) printf((__constant char *)"loop: a=%x:%x:%x:%x:%x:%x * f = %x:%x:%x:%x:%x:%x (tmp)\n",
        V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0), V(tmp90.d5), V(tmp90.d4), V(tmp90.d3), V(tmp90.d2), V(tmp90.d1), V(tmp90.d0));
#endif
    tmp90.d0 = (b.d0 - tmp90.d0) & 0x7FFF;
    tmp90.d1 = (b.d1 - tmp90.d1 + AS_UINT_V((tmp90.d0 > b.d0)  ));
    tmp90.d2 = (b.d2 - tmp90.d2 + AS_UINT_V((tmp90.d1 > b.d1)  ));
    tmp90.d3 = (b.d3 - tmp90.d3 + AS_UINT_V((tmp90.d2 > b.d2)  ));
    tmp90.d4 = (b.d4 - tmp90.d4 + AS_UINT_V((tmp90.d3 > b.d3)  ));
    tmp90.d5 = (b.d5 - tmp90.d5 + AS_UINT_V((tmp90.d4 > b.d4)  ));
    tmp90.d1 &= 0x7FFF;
    tmp90.d2 &= 0x7FFF;
    tmp90.d3 &= 0x7FFF;
    tmp90.d4 &= 0x7FFF;
    tmp90.d5 &= 0x7FFF;

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"loop: b=%x:%x:%x:%x:%x:%x - tmp = %x:%x:%x:%x:%x:%x (tmp)\n",
        V(b.d5), V(b.d4), V(b.d3), V(b.d2), V(b.d1), V(b.d0), V(tmp90.d5), V(tmp90.d4), V(tmp90.d3), V(tmp90.d2), V(tmp90.d1), V(tmp90.d0));
#endif
    if (shifter & 0x80000000) shl_90(&tmp90);
    if (shifter == 0x80000000) break;
    shifter+=shifter;

#ifndef CHECKS_MODBASECASE
    mod_simple_90(&a, tmp90, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    mod_simple_90(&a, tmp90, f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max65, limit, modbasecase_debug);
#endif

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"loopend: exp=%x, tmp=%x:%x:%x:%x:%x:%x mod f=%x:%x:%x:%x:%x:%x = %x:%x:%x:%x:%x:%x (a)\n",
        shifter, V(tmp90.d5), V(tmp90.d4), V(tmp90.d3), V(tmp90.d2), V(tmp90.d1), V(tmp90.d0),
        V(f.d5), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), V(a.d5), V(a.d4), V(a.d3), V(a.d2), V(a.d1), V(a.d0) );
#endif
  }

  mod_simple_even_90_and_check_big_factor90(tmp90, f, ff, RES
#ifdef CHECKS_MODBASECASE
                                      , bit_max65, 10, modbasecase_debug
#endif
                                      );
}


#ifndef CL_GPU_SIEVE
/****
 * the actual kernels for handling 6x15bit computations
 ****/

__kernel void cl_barrett15_82(__private uint exponent, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                              const uint8 b_in, __global uint * restrict RES, const int bit_max65
                              MODBASECASE_PAR_DEF         )
{
  __private int90_v f;
  __private uint tid;

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC90(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82: tid=%d, f=%x:%x:%x:%x:%x:%x, shift=%d\n",
        tid, V(f.d5), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif
  check_barrett15_82(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}

__kernel void cl_barrett15_83(__private uint exponent, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                              const uint8 b_in, __global uint * restrict RES, const int bit_max65
                              MODBASECASE_PAR_DEF         )
{
  __private int90_v f;
  __private uint tid;

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC90(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83: tid=%d, f=%x:%x:%x:%x:%x:%x, shift=%d\n",
        tid, V(f.d5), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif
  check_barrett15_83(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}

__kernel void cl_barrett15_88(__private uint exponent, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                              const uint8 b_in, __global uint * restrict RES, const int bit_max65
                              MODBASECASE_PAR_DEF         )
{
  __private int90_v f;
  __private uint tid;

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), (uint)get_local_id(0)) * VECTOR_SIZE;

  calculate_FC90(exponent, tid, k_tab, k_base, &f);

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_88: tid=%d, f=%x:%x:%x:%x:%x:%x, shift=%d\n",
        tid, V(f.d5), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0), shiftcount);
#endif
  check_barrett15_88(exponent << (32 - shiftcount), f, tid, b_in, bit_max65, RES
                     MODBASECASE_PAR);
}


#else
/****************************************
 ****************************************
 * 15-bit-kernel consuming the GPU sieve
 * included by main kernel file
 ****************************************
 ****************************************/


__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett15_69_gs(const uint exponent, const int75_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int75_v  k, f;
  __private uint     tid, lid=get_local_id(0);
  __private int75_t exp75;

	tid = mad24((uint)get_group_id(0), (uint)get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett15_69_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0x7FFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 15) + mad24(NUM_CLASSES, k_delta >> 15, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 15) + k_base.d2;
    k.d3 = (k.d2 >> 15) + k_base.d3;
    k.d4 = (k.d3 >> 15) + k_base.d4;

    k.d0 &= 0x7FFF;
    k.d1 &= 0x7FFF;
    k.d2 &= 0x7FFF;
    k.d3 &= 0x7FFF;

    f.d0 = mad24(k.d0, exp75.d0, 1u);  // exp75 = 2*exponent ==> f = 2kp+1

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
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_69_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x\n",
        i, smem[i], V(k_delta), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett15_69(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett15_70_gs(const uint exponent, const int75_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int75_v  k, f;
  __private uint     tid, lid=get_local_id(0);
  __private int75_t exp75;

	tid = mad24(get_group_id(0), get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett15_70_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_70_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0x7FFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 15) + mad24(NUM_CLASSES, k_delta >> 15, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 15) + k_base.d2;
    k.d3 = (k.d2 >> 15) + k_base.d3;
    k.d4 = (k.d3 >> 15) + k_base.d4;

    k.d0 &= 0x7FFF;
    k.d1 &= 0x7FFF;
    k.d2 &= 0x7FFF;
    k.d3 &= 0x7FFF;

    f.d0 = mad24(k.d0, exp75.d0, 1u);  // exp75 = 2*exponent ==> f = 2kp+1

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
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_70_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x\n",
        i, smem[i], V(k_delta), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett15_70(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett15_71_gs(const uint exponent, const int75_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int75_v  k, f;
  __private uint     tid, lid=get_local_id(0);
  __private int75_t exp75;

	tid = mad24(get_group_id(0), get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett15_71_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_71_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0x7FFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 15) + mad24(NUM_CLASSES, k_delta >> 15, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 15) + k_base.d2;
    k.d3 = (k.d2 >> 15) + k_base.d3;
    k.d4 = (k.d3 >> 15) + k_base.d4;

    k.d0 &= 0x7FFF;
    k.d1 &= 0x7FFF;
    k.d2 &= 0x7FFF;
    k.d3 &= 0x7FFF;

    f.d0 = mad24(k.d0, exp75.d0, 1u);  // exp75 = 2*exponent ==> f = 2kp+1

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

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_71_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x\n",
        i, smem[i], V(k_delta), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett15_71(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett15_73_gs(const uint exponent, const int75_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int75_v  k, f;
  __private uint     tid, lid=get_local_id(0);
  __private int75_t exp75;

	tid = mad24(get_group_id(0), get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett15_73_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0x7FFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 15) + mad24(NUM_CLASSES, k_delta >> 15, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 15) + k_base.d2;
    k.d3 = (k.d2 >> 15) + k_base.d3;
    k.d4 = (k.d3 >> 15) + k_base.d4;

    k.d0 &= 0x7FFF;
    k.d1 &= 0x7FFF;
    k.d2 &= 0x7FFF;
    k.d3 &= 0x7FFF;

    f.d0 = mad24(k.d0, exp75.d0, 1u);  // exp75 = 2*exponent ==> f = 2kp+1

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

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_73_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x\n",
        i, smem[i], V(k_delta), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett15_73(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett15_74_gs(const uint exponent, const int75_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int75_v  k, f;
  __private uint     tid, lid=get_local_id(0);
  __private int75_t exp75;

	tid = mad24(get_group_id(0), get_local_size(0), lid);

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0x7FFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 15) + mad24(NUM_CLASSES, k_delta >> 15, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 15) + k_base.d2;
    k.d3 = (k.d2 >> 15) + k_base.d3;
    k.d4 = (k.d3 >> 15) + k_base.d4;

    k.d0 &= 0x7FFF;
    k.d1 &= 0x7FFF;
    k.d2 &= 0x7FFF;
    k.d3 &= 0x7FFF;

    f.d0 = mad24(k.d0, exp75.d0, 1u);  // exp75 = 2*exponent ==> f = 2kp+1

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

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_74_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x\n",
        i, smem[i], V(k_delta), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett15_74(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}


/****************************************
 ****************************************
 * 15-bit based 90-bit barrett-kernels based on GPU sieve
 *
 ****************************************
 ****************************************/

__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett15_82_gs(const uint exponent, const int75_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int75_v  k;
  __private int90_v  f;
  __private uint     tid, lid=get_local_id(0);
  __private int75_t  exp75;

  tid = mad24(get_group_id(0), get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (tid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett15_82_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);
// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0x7FFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 15) + mad24(NUM_CLASSES, k_delta >> 15, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 15) + k_base.d2;
    k.d3 = (k.d2 >> 15) + k_base.d3;
    k.d4 = (k.d3 >> 15) + k_base.d4;

    k.d0 &= 0x7FFF;
    k.d1 &= 0x7FFF;
    k.d2 &= 0x7FFF;
    k.d3 &= 0x7FFF;

    f.d0 = mad24(k.d0, exp75.d0, 1u);  // exp75 = 2*exponent ==> f = 2kp+1

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

    f.d5 = mad24(k.d4, exp75.d1, f.d4 >> 15);  // PERF: see above
    f.d5 = mad24(k.d3, exp75.d2, f.d5);
    f.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_82_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x:%x\n",
        i, smem[i], V(k_delta), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0), V(f.d5), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett15_82(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett15_83_gs(const uint exponent, const int75_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int75_v  k;
  __private int90_v  f;
  __private uint     tid, lid=get_local_id(0);
  __private int75_t  exp75;

	tid = mad24(get_group_id(0), get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett15_83_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0x7FFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 15) + mad24(NUM_CLASSES, k_delta >> 15, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 15) + k_base.d2;
    k.d3 = (k.d2 >> 15) + k_base.d3;
    k.d4 = (k.d3 >> 15) + k_base.d4;

    k.d0 &= 0x7FFF;
    k.d1 &= 0x7FFF;
    k.d2 &= 0x7FFF;
    k.d3 &= 0x7FFF;

    f.d0 = mad24(k.d0, exp75.d0, 1u);  // exp75 = 2*exponent ==> f = 2kp+1

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

    f.d5 = mad24(k.d4, exp75.d1, f.d4 >> 15);  // PERF: see above
    f.d5 = mad24(k.d3, exp75.d2, f.d5);
    f.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_83_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x:%x\n",
        i, smem[i], V(k_delta), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0), V(f.d5), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett15_83(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}

__kernel void __attribute__((reqd_work_group_size(256, 1, 1)))
              cl_barrett15_88_gs(const uint exponent, const int75_t k_base,
                                 const __global uint * restrict bit_array,
                                 const uint bits_to_process, __local ushort *smem,
                                 const int shiftcount, const uint8 b_in,
                                 __global uint * restrict RES, const int bit_max65,
                                 const uint shared_mem_allocated // only used to verify assumptions
                                 MODBASECASE_PAR_DEF         )
{
  __private uint     i, initial_shifter_value, total_bit_count;
  __local   ushort   bitcount[256];	// Each thread of our block puts bit-counts here
  __private int75_v  k;
  __private int90_v  f;
  __private uint     tid, lid=get_local_id(0);
  __private int75_t  exp75;

	tid = mad24(get_group_id(0), get_local_size(0), lid);

#if (TRACE_SIEVE_KERNEL > 0)
    if (lid==TRACE_SIEVE_TID) printf((__constant char *)"cl_barrett15_88_gs: exp=%d=%#x, k=%x:%x:%x, bits=%d, shift=%d, bit_max65=%d, b_in=%x:%x:%x:%x:%x:%x:%x:%x, base addr=%#x\n",
        exponent, exponent, k_base.d2, k_base.d1, k_base.d0, bits_to_process, shiftcount, bit_max65, b_in.s7, b_in.s6, b_in.s5, b_in.s4, b_in.s3, b_in.s2, b_in.s1, b_in.s0, bit_array);
#endif

  // extract the bits set in bit_array into smem and get the total count (call to gpusieve.cl)
  total_bit_count = extract_bits(bits_to_process, tid, lid, bitcount, smem, bit_array);

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested  <== this makes the OpenCL compiler abort, supposed to be fixed in Cat 13.4
// Compute factor corresponding to first sieve bit in this block.

  initial_shifter_value = exponent << (32 - shiftcount);	// Initial shifter value

  exp75.d2=exponent>>29;exp75.d1=(exponent>>14)&0x7FFF;exp75.d0=(exponent<<1)&0x7FFF;	// exp75 = 2 * exponent  // PERF: exp.d1=amd_bfe(exp, 15, 14)

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_88_gs: exp=%u, shift=%d, shifted exp=%#x, total_bit_count=%u, shared_mem_size=%u\n",
        exponent, shiftcount, initial_shifter_value, total_bit_count, shared_mem_allocated);
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

    k.d0 =                mad24(NUM_CLASSES, k_delta & 0x7FFF, k_base.d0);  // k_delta can exceed 2^24
    k.d1 = (k.d0 >> 15) + mad24(NUM_CLASSES, k_delta >> 15, k_base.d1);  // k is limited to 2^64 -1
    k.d2 = (k.d1 >> 15) + k_base.d2;
    k.d3 = (k.d2 >> 15) + k_base.d3;
    k.d4 = (k.d3 >> 15) + k_base.d4;

    k.d0 &= 0x7FFF;
    k.d1 &= 0x7FFF;
    k.d2 &= 0x7FFF;
    k.d3 &= 0x7FFF;

    f.d0 = mad24(k.d0, exp75.d0, 1u);  // exp75 = 2*exponent ==> f = 2kp+1

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

    f.d5 = mad24(k.d4, exp75.d1, f.d4 >> 15);  // PERF: see above
    f.d5 = mad24(k.d3, exp75.d2, f.d5);
    f.d4 &= 0x7FFF;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"cl_barrett15_88_gs: x: smem[%d]=%d, k_delta=%d, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x:%x\n",
        i, smem[i], V(k_delta), V(k.d4), V(k.d3), V(k.d2), V(k.d1), V(k.d0), V(f.d5), V(f.d4), V(f.d3), V(f.d2), V(f.d1), V(f.d0));
#endif

    check_barrett15_88(initial_shifter_value, f, tid, b_in, bit_max65, RES
                       MODBASECASE_PAR);
  }
}
#endif