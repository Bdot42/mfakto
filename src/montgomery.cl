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

/*
The Montgomery reduction algorithm Redc(T) calculates TR^{-1} mod{N} as follows:

N: the factor to test, odd
R: 2^64 (or a higher power of 2), R > N
R^{-1}: the modular inverse of R,  RR^{-1} == 1 mod {N}  =>  RR^{-1} = kN+1
T: the intermediate result of squaring and doubling

    m := (T mod {R})k mod {R}
    t := (T + mN)/R
    if t > N return t - N else return t.

*/

ulong_v invmod2pow64 (const ulong_v n)
{
  ulong_v r;
  const uint_v in = CONVERT_UINT_V(n);
  uint_v ir;
  // (3*n) XOR 2 is the correct inverse modulo 32 (5 bits),
  // then run 4 (for 64 bit) Newton iterations.
  ir = mul24(in, 3u) ^ 2;

  ir += ir - mul24(mul24(ir, ir), in);
  ir += ir - mul24(mul24(ir, ir), in);
  // ir should now be the inverse mod 2^20 - yet a mul24 for any of
  // the following multiplications delivers a wrong result
  r   = CONVERT_ULONG_V(ir+ir - ir * ir * in);
  r  += r - r * r * n;
  return r;
}

ulong_v neginvmod2pow64 (const ulong_v n)
{
  ulong_v r;
  const uint_v in = CONVERT_UINT_V(n);
  uint_v ir;
  // (3*n) XOR 2 is the correct inverse modulo 32 (5 bits),
  // then run 4 (for 64 bit) Newton iterations.
  ir = mul24(in, 3u) ^ 2;

  ir += ir - mul24(mul24(ir, ir), in);
  ir += ir - mul24(mul24(ir, ir), in);
  // ir should now be the inverse mod 2^20 - yet a mul24 for any of
  // the following multiplications delivers a wrong result
  r   = CONVERT_ULONG_V(ir+ir - ir * ir * in);
  r  = r * r * n - (r+r);
  return r;
}

ulong_v mulmod_REDC64 (const ulong_v a, const ulong_v b, const ulong_v N, const ulong_v Ns)
{
  ulong_v r1, r2;

  // Compute T=a*b; m = (T*Ns)%2^64; T += m*N; if (T>N) T-= N;
  r1 = a*b;
  r2 = mul_hi(a,b); // r2:r1 = T
  
  r1 *= Ns;	// (T*Ns) mod 2^64 = m
  r2 += (r1!=0)? 1UL : 0UL;
  r1 = mul_hi(r1, N) + r2;

  r2 = r1 - N;
  r1 = (r1>N)?r2:r1;

  return r1;
}

// mulmod_REDC(1, 1, N, Ns)
// Note that mulmod_REDC(a, 1, N, Ns) == mulmod_REDC(1, 1, N, Ns*a).
ulong_v onemod_REDC64 (const ulong_v N, const ulong_v Ns)
{
  ulong_v r;

  r = mad_hi(N, Ns, 1UL);

  return (r>N)?r-N:r;
}

// Like mulmod_REDC(a, 1, N, Ns) == mulmod_REDC(1, 1, N, Ns*a).
ulong_v mod_REDC64(const ulong_v a, const ulong_v N, const ulong_v Ns)
{
  return onemod_REDC64(N, Ns*a);
}

__kernel void __attribute__((work_group_size_hint(256, 8192, 1))) cl_mg62(__private uint exp, const int96_t k_base, const __global uint * restrict k_tab, const int shiftcount,
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
  __private int96_v k;
  __private ulong_v a, f, f_inv, As;
  __private uint tid;
  __private uint_v t;

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_mg62: exp=%d, k_base=%x:%x:%x\n",
        exp, k_base.d2, k_base.d1, k_base.d0);
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
  k.d1 = mad_hi(t, 4620u, k_base.d1) - AS_UINT_V(k_base.d0 > k.d0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */

  f = upsample(k.d1, k.d0) * ((ulong)exp + exp) + 1;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_mg62: k_tab[%d]=%x, f=%#llx, shift=%d\n",
        tid, t.s0, f.s0, shiftcount);
#endif

  f_inv = neginvmod2pow64(f);

  /* the montgomery repr. of A, As, can be calculated by either calculating Rs = R^2 mod P
   and then As = mulmod_REDC(A, Rs)  -or-
   directly calculating As = A*R mod P.
   R = 2^64 for this kernel.
   
   In the later case, we can start with A=1 (no preshifting), which means As = R mod P = R-P mod P (R-P fits in ulong)
   */

// first case
  /*
  shiftcount=4;  // no exp below 2^10 ;-)
  while((A=exp>>shiftcount) > 63)shiftcount++;
  A=(ulong)1<<A;
  exp=exp<<(32-shiftcount);
  Rs = R-P; // R=2^64, Rs = R - P = R (mod P), but Rs is now less than 2^64
        Rs = Rs % P;
        Rs = Rs * Rs;  // works only if P < 2^32
        Rs = Rs % P;

//  printf ("\nR=%llu ==> Rs=%llu\n", R, Rs);

   As=mulmod_REDC64 (A, Rs, P, Pinv);

   */
// second case
//   while ((exp&0x80000000) == 0) exp<<=1; // shift exp to the very left of the 32 bits
   exp <<= clz(exp); // shift exp to the very left of the 32 bits
   As = (0 - f) % f;

   // verify As
   // if (A != (B=mod_REDC64 (As, P, Pinv))) printf((__constant char *)"ERROR: A (%llu)!= mod_REDC(As=%llu,1) = %llu\n", A, As, B);

//   printf ("A=%#llx ==> Am=%llu, P=%llu (%#llx..<32>): ", A, As, P, P>>32);

   // A=1 => A*A=1 => As*As=As => skip the first mulmod
   exp <<=1;
   As  <<=1;

   while(exp)
   {
     As=mulmod_REDC64 (As, As, f, f_inv);  // square
 //    printf ("square=%llu\n", As);
     if (exp&0x80000000) As <<=1;         // mul by 2
     exp<<=1;
//     printf ("loopend:exp=%#x, As=%llu (%#llx..<32>)\n", exp, As, As>>32UL);
   }

   
   a = mod_REDC64 (As, f, f_inv);
//   printf ("result = %llu\n", A);

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( a==1 )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf((__constant char *)"cl_mg62: tid=%ld found factor: q=%#llx, k=%x:%x:%x\n", tid, f.s0, k.d2.s0, k.d1.s0, k.d0.s0);
#endif
/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f != 1 */  
    tid=ATOMIC_INC(RES[0]);
    if(tid<10)				/* limit to 10 factors per class */
    {
      RES[tid*3 + 1]=0;
      RES[tid*3 + 2]=CONVERT_UINT_V(f>>32);
      RES[tid*3 + 3]=CONVERT_UINT_V(f);
    }
  }
#elif (VECTOR_SIZE == 2)
  EVAL_RES_l(x)
  EVAL_RES_l(y)
#elif (VECTOR_SIZE == 3)
  EVAL_RES_l(x)
  EVAL_RES_l(y)
  EVAL_RES_l(z)
#elif (VECTOR_SIZE == 4)
  EVAL_RES_l(x)
  EVAL_RES_l(y)
  EVAL_RES_l(z)
  EVAL_RES_l(w)
#elif (VECTOR_SIZE == 8)
  EVAL_RES_l(s0)
  EVAL_RES_l(s1)
  EVAL_RES_l(s2)
  EVAL_RES_l(s3)
  EVAL_RES_l(s4)
  EVAL_RES_l(s5)
  EVAL_RES_l(s6)
  EVAL_RES_l(s7)
#elif (VECTOR_SIZE == 16)
  EVAL_RES_l(s0)
  EVAL_RES_l(s1)
  EVAL_RES_l(s2)
  EVAL_RES_l(s3)
  EVAL_RES_l(s4)
  EVAL_RES_l(s5)
  EVAL_RES_l(s6)
  EVAL_RES_l(s7)
  EVAL_RES_l(s8)
  EVAL_RES_l(s9)
  EVAL_RES_l(sa)
  EVAL_RES_l(sb)
  EVAL_RES_l(sc)
  EVAL_RES_l(sd)
  EVAL_RES_l(se)
  EVAL_RES_l(sf)
#endif
}

/****************** 90-bit impl. ***********************/

void square_45_90(int90_v * const res, const int90_v a)
/* res = (low 3 components of a)^2 = d0^2 + 2d0d1 + d1^2 + 2d0d2 + 2d1d2 + d2^2
              
   */
{
  // assume we have enough spare bits and can do all the carries at the very end:
  // 0x7FFF * 0x7FFF = 0x3FFF0001 = max result of mul24, up to 4 of these can be
  // added into 32-bit: 0x3FFF0001 * 4 = 0xFFFC0004, which even leaves room for
  // one (almost two) carry of 17 bit (32-bit >> 15)
  // square 5x5 requires: 15 mul/mad24, 14 shift, 10 and, 1 add

  res->d0 = mul24(a.d0, a.d0);

  res->d1 = mad24(a.d1, a.d0 << 1, res->d0 >> 15);
  res->d0 &= 0x7FFF;

  res->d2 = mad24(a.d1, a.d1, res->d1 >> 15);
  res->d2 = mad24(a.d2, a.d0 << 1, res->d2);
  res->d1 &= 0x7FFF;

  res->d3 = mad24(a.d2, a.d1 << 1, res->d2 >> 15);
  res->d3 = mad24(a.d3, a.d0 << 1, res->d3);
  res->d2 &= 0x7FFF;

  res->d4 = mad24(a.d2, a.d2, res->d3 >> 15);
  res->d4 = mad24(a.d3, a.d1 << 1, res->d4);
  res->d5 = res->d4 >> 15;
  res->d4 &= 0x7FFF;
}

int90_v sub_90(const int90_v a, const int90_v b)
/* return a-b */
{
  int90_v tmp;

  tmp.d0 = (a.d0 - b.d0) & 0x7FFF;
  tmp.d1 = (a.d1 - b.d1 + AS_UINT_V(b.d0 > a.d0));
  tmp.d2 = (a.d2 - b.d2 + AS_UINT_V(tmp.d1 > a.d1));
  tmp.d3 = (a.d3 - b.d3 + AS_UINT_V(tmp.d2 > a.d2));
  tmp.d4 = (a.d4 - b.d4 + AS_UINT_V(tmp.d3 > a.d3));
  tmp.d5 = (a.d5 - b.d5 + AS_UINT_V(tmp.d4 > a.d4));
  tmp.d1&= 0x7FFF;
  tmp.d2&= 0x7FFF;
  tmp.d3&= 0x7FFF;
  tmp.d4&= 0x7FFF;
  tmp.d5&= 0x7FFF;

  return tmp;
}

int90_v neg_90(const int90_v b)
/* return -b for odd b */
{
  int90_v tmp;

  tmp.d0 = (-b.d0) & 0x7FFF; // for odd b, we'll always have a borrow
  tmp.d1 = (-b.d1 - 1) & 0x7FFF;
  tmp.d2 = (-b.d2 - 1) & 0x7FFF;
  tmp.d3 = (-b.d3 - 1) & 0x7FFF;
  tmp.d4 = (-b.d4 - 1) & 0x7FFF;
  tmp.d5 = (-b.d5 - 1) & 0x7FFF;

  return tmp;
}

void shl_45(int90_v * const a)
/* shiftleft a one bit */
{
  a->d5 = 0;
  a->d4 = 0;
  a->d3 = a->d2 >> 14;
  a->d2 = mad24(a->d2, 2u, a->d1 >> 14) & 0x7FFF;
  a->d1 = mad24(a->d1, 2u, a->d0 >> 14) & 0x7FFF;
  a->d0 = (a->d0 << 1u) & 0x7FFF;
}

uint_v invmod2pow15 (const uint_v n)
{
  uint_v r;

  // (3*n) XOR 2 is the correct inverse modulo 32 (5 bits),
  // then run 2 (for 15 bit) Newton iterations.
  // r = (n+n+n) ^ 2UL;
  r = mul24(n, 3u) ^ 2;

  r += r - mul24(mul24(r, r), n);
  r += r - mul24(mul24(r, r), n);

  return r & 0x7FFF;
}

uint_v neginvmod2pow15 (const uint_v n)
{
  uint_v r;

  // (3*n) XOR 2 is the correct inverse modulo 32 (5 bits),
  // then run 2 (for 15 bit) Newton iterations.
  // r = (n+n+n) ^ 2UL;
  r = mul24(n, 3u) ^ 2;

  r += r - mul24(mul24(r, r), n);
  r = mul24(mul24(r, r), n) - r - r;  // negate the output

  return r & 0x7FFF;
}


int90_v squaremod_REDC90 (const int90_v x, const int90_v m, const uint_v t)
{
  /*Alex Kruppa:
Unless your modulus is quite large so that sub-quadratic multiplication algorithms become attractive, the best way to do REDC is usually just running k sequential one-word REDC passes, if the modulus has k words.

E.g., given

t = -1/m (mod 2^64)
a = x*y < m^2

do

for i = 1, ..., k {
a += (a % 2^64 * t) % 2^64 * m
a /= 2^64
}
if a >= m {
a -= m
}

In our case, the word size is just 2^15 instead of 2^64.

*/

  uint_v   r;
  int90_v ret;
  int180_v a;

  square_90_180(&a, x);

  // loop unrolled 6 times
  r = mul24(a.d0, t) & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);  // assigning the d1 value to d0 (etc.) is the div by 2^15
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15) + a.d6;  // always add in the next higher component
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15) + a.d7;  // this was actually d6 if we had done the div by 2^15 for the upper components as well
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15) + a.d8;
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15) + a.d9;
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15) + a.da;
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15) + a.db;
  ret.d0 = a.d0 & 0x7FFF;
  ret.d1 = a.d1 & 0x7FFF;
  ret.d2 = a.d2 & 0x7FFF;
  ret.d3 = a.d3 & 0x7FFF;
  ret.d4 = a.d4 & 0x7FFF;
  ret.d5 = a.d5;
  
  return sub_if_gte_90(ret, m);
}


int90_v mod_REDC90(int90_v a, const int90_v m, const uint_v t)
{
  uint_v   r;
  // loop unrolled 6 times
  r = mul24(a.d0, t) & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);  // assigning the d1 value to d0 (etc.) is the div by 2^15
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15);
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15);
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15);
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15);
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15);
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;

  r = mul24(a.d0, t)  & 0x7FFF;

  a.d0 = mad24(r, m.d0, a.d0);   // this is not needed anymore, just to provide the carry to d1
  a.d0 = (a.d0 >> 15) + mad24(r, m.d1, a.d1);
  a.d1 = (a.d0 >> 15) + mad24(r, m.d2, a.d2);
  a.d2 = (a.d1 >> 15) + mad24(r, m.d3, a.d3);
  a.d3 = (a.d2 >> 15) + mad24(r, m.d4, a.d4);
  a.d4 = (a.d3 >> 15) + mad24(r, m.d5, a.d5);
  a.d5 = (a.d4 >> 15);
  a.d0 &= 0x7FFF;
  a.d1 &= 0x7FFF;
  a.d2 &= 0x7FFF;
  a.d3 &= 0x7FFF;
  a.d4 &= 0x7FFF;
  
  return sub_if_gte_90(a, m);
}


__kernel void __attribute__((work_group_size_hint(256, 8192, 1))) cl_mg88(__private uint exp, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_t bb,
#endif
                           __global uint * restrict RES, const int bit_max
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
  __private int90_v a, f, As, k;
  __private int75_t exp75;
  __private uint tid;
  __private uint_v t, f_inv;
  __private float_v ff;

	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * VECTOR_SIZE;
  exp75.d2=exp>>29;exp75.d1=(exp>>14)&0x7FFF;exp75.d0=(exp<<1)&0x7FFF;	// exp75 = 2 * exp

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf((__constant char *)"cl_mg88: exp=%d, x2=%x:%x:%x, k_base=%x:%x:%x:%x\n",
        exp, exp75.d2, exp75.d1, exp75.d0, k_base.d3, k_base.d2, k_base.d1, k_base.d0);
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
  if (tid==TRACE_TID) printf((__constant char *)"cl_mg88: k_tab[%d]=%x, k_base+k*4620=%x:%x:%x:%x:%x\n",
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

//  f.d5 = mad24(k.d5, exp75.d0, f.d4 >> 15);  // k.d5 = 0
  f.d5 = mad24(k.d4, exp75.d1, f.d4 >> 15);
  f.d5 = mad24(k.d3, exp75.d2, f.d5);
  f.d4 &= 0x7FFF;

#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf((__constant char *)"cl_mg88: k_tab[%d]=%x, k=%x:%x:%x:%x:%x, f=%x:%x:%x:%x:%x:%x, shift=%d\n",
        tid, t.s0, k.d4.s0, k.d3.s0, k.d2.s0, k.d1.s0, k.d0.s0, f.d5.s0, f.d4.s0, f.d3.s0, f.d2.s0, f.d1.s0, f.d0.s0, shiftcount);
#endif

  f_inv = neginvmod2pow15(f.d0);

  /* the montgomery repr. As of A can be calculated by either calculating Rs = R^2 mod f
   and then As = mulmod_REDC(A, Rs)  -or-
   directly calculating As = A*R mod f.
   R = 2^90 for this kernel.
   
   In the later case, we can start with A=1 (no preshifting), which means As = R mod f = R-f mod f (which is just a 90-bit mod).
   If we furthermore limit f > 2^75, then we can use mod_simple for this calculation - with a little more effort mod_simple could handle f>2^68
   */

// first case - not used because of the full-mod required.
  /*
  shiftcount=4;  // no exp below 2^10 ;-)
  while((A=exp>>shiftcount) > 63)shiftcount++;
  A=(ulong)1<<A;
  exp=exp<<(32-shiftcount);
  Rs = R-P; // R=2^64, Rs = R - P = R (mod P), but Rs is now less than 2^64
        Rs = Rs % P;
        Rs = Rs * Rs;  // works only if P < 2^32
        Rs = Rs % P;

//  printf ("\nR=%llu ==> Rs=%llu\n", R, Rs);

   As=mulmod_REDC64 (A, Rs, P, Pinv);

   */
// second case
  ff= CONVERT_FLOAT_RTP_V(mad24(f.d5, 32768u, f.d4));
  ff= ff * 1073741824.0f+ CONVERT_FLOAT_RTP_V(mad24(f.d3, 32768u, f.d2));   // f.d1 needed?

  ff = as_float(0x3f7ffffd) / ff;

//   while ((exp&0x80000000) == 0) exp<<=1; // shift exp to the very left of the 32 bits
exp <<= clz(exp); // shift exp to the very left of the 32 bits

#ifndef CHECKS_MODBASECASE
  mod_simple_90(&As, neg_90(f), f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 );					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  mod_simple_90(&As, neg_90(f), f, ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                   , bit_max, 1 << (90-bit_max), modbasecase_debug);
#endif

  // As is now the montgomery-representation of 1
#if (TRACE_KERNEL > 2)
   if (tid==TRACE_TID) printf((__constant char *)"cl_mg88: exp=0x%x, As=%x:%x:%x:%x:%x:%x, f_inv=%x\n",
        exp, As.d5.s0, As.d4.s0, As.d3.s0, As.d2.s0, As.d1.s0, As.d0.s0, f_inv.s0);
#endif
#if (TRACE_KERNEL > 3)
     a = mod_REDC90 (As, f, f_inv);
     if (tid==TRACE_TID) printf((__constant char *)"cl_mg88-beforeshift: exp=0x%x, As=%x:%x:%x:%x:%x:%x (a=%x:%x:%x:%x:%x:%x)\n",
        exp, As.d5.s0, As.d4.s0, As.d3.s0, As.d2.s0, As.d1.s0, As.d0.s0, a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif

   // A=1 => A*A=1 => As*As=As => skip the first mulmod
   exp <<=1;
   shl_90(&As);

#if (TRACE_KERNEL > 3)
     a = mod_REDC90 (As, f, f_inv);
     if (tid==TRACE_TID) printf((__constant char *)"cl_mg88-beforeloop: exp=0x%x, As=%x:%x:%x:%x:%x:%x (a=%x:%x:%x:%x:%x:%x)\n",
        exp, As.d5.s0, As.d4.s0, As.d3.s0, As.d2.s0, As.d1.s0, As.d0.s0, a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
   while(exp)
   {
     As=squaremod_REDC90 (As, f, f_inv);      // square
     if (exp&0x80000000) shl_90(&As);         // mul by 2
#if (TRACE_KERNEL > 3)
     a = mod_REDC90 (As, f, f_inv);
     if (tid==TRACE_TID) printf((__constant char *)"cl_mg88-loop: exp=0x%x, As=%x:%x:%x:%x:%x:%x (a=%x:%x:%x:%x:%x:%x)\n",
        exp, As.d5.s0, As.d4.s0, As.d3.s0, As.d2.s0, As.d1.s0, As.d0.s0, a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif
     exp<<=1;
   }

   
   a = mod_REDC90 (As, f, f_inv);
#if (TRACE_KERNEL > 1)
   if (tid==TRACE_TID) printf((__constant char *)"cl_mg88-end: exp=0x%x, As=%x:%x:%x:%x:%x:%x, a=%x:%x:%x:%x:%x:%x\n",
        exp, As.d5.s0, As.d4.s0, As.d3.s0, As.d2.s0, As.d1.s0, As.d0.s0, a.d5.s0, a.d4.s0, a.d3.s0, a.d2.s0, a.d1.s0, a.d0.s0);
#endif

/* finally check if we found a factor and write the factor to RES[] */
#if (VECTOR_SIZE == 1)
  if( ((a.d5|a.d4|a.d3|a.d2|a.d1)==0 && a.d0==1) )
  {
/* in contrast to the other kernels this barrett based kernel is only allowed for factors above 2^60 so there is no need to check for f != 1 */  
    tid=ATOMIC_INC(RES[0]);
    if(tid<10)				/* limit to 10 factors per class */
    {
      RES[tid*3 + 1]=mad24(f.d5,0x8000u, f.d4);
      RES[tid*3 + 2]=mad24(f.d3,0x8000u, f.d2);  // that's now 30 bits per int
      RES[tid*3 + 3]=mad24(f.d1,0x8000u, f.d0);  
    }
  }
#elif (VECTOR_SIZE == 2)
  EVAL_RES_90(x)
  EVAL_RES_90(y)
#elif (VECTOR_SIZE == 3)
  EVAL_RES_90(x)
  EVAL_RES_90(y)
  EVAL_RES_90(z)
#elif (VECTOR_SIZE == 4)
  EVAL_RES_90(x)
  EVAL_RES_90(y)
  EVAL_RES_90(z)
  EVAL_RES_90(w)
#elif (VECTOR_SIZE == 8)
  EVAL_RES_90(s0)
  EVAL_RES_90(s1)
  EVAL_RES_90(s2)
  EVAL_RES_90(s3)
  EVAL_RES_90(s4)
  EVAL_RES_90(s5)
  EVAL_RES_90(s6)
  EVAL_RES_90(s7)
#elif (VECTOR_SIZE == 16)
  EVAL_RES_90(s0)
  EVAL_RES_90(s1)
  EVAL_RES_90(s2)
  EVAL_RES_90(s3)
  EVAL_RES_90(s4)
  EVAL_RES_90(s5)
  EVAL_RES_90(s6)
  EVAL_RES_90(s7)
  EVAL_RES_90(s8)
  EVAL_RES_90(s9)
  EVAL_RES_90(sa)
  EVAL_RES_90(sb)
  EVAL_RES_90(sc)
  EVAL_RES_90(sd)
  EVAL_RES_90(se)
  EVAL_RES_90(sf)
#endif
 
}
