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

Version 0.12

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

  // (3*n) XOR 2 is the correct inverse modulo 32 (5 bits),
  // then run 4 (for 64 bit) Newton iterations.
  r = (n+n+n) ^ 2UL;
  
  r += r - CONVERT_ULONG_V(CONVERT_UINT_V(r) * CONVERT_UINT_V(r) * in);
  r += r - CONVERT_ULONG_V(CONVERT_UINT_V(r) * CONVERT_UINT_V(r) * in);
  r += r - CONVERT_ULONG_V(CONVERT_UINT_V(r) * CONVERT_UINT_V(r) * in);
  r += r - r * r * n;

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

__kernel void cl_mg64(__private uint exp, const int96_1t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int192_1t bb,
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
  __private int96_t k;
  __private ulong_v a, f, f_inv, A, As;
  __private uint tid;
  __private uint_v t;

#ifdef WA_FOR_CATALYST11_10_BUG
  __private int192_1t bb={b_in.s0, b_in.s1, b_in.s2, b_in.s3, b_in.s4, b_in.s5};
#endif

	//tid = (get_global_id(0)+get_global_size(0)*get_global_id(1)) * BARRETT_VECTOR_SIZE;
	tid = mad24((uint)get_global_id(1), (uint)get_global_size(0), (uint)get_global_id(0)) * BARRETT_VECTOR_SIZE;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("cl_mg64: exp=%d, b=%x:%x:%x:%x:%x:%x, k_base=%x:%x:%x\n",
        exp, bb.d5, bb.d4, bb.d3, bb.d2, bb.d1, bb.d0, k_base.d2, k_base.d1, k_base.d0);
#endif

#if (BARRETT_VECTOR_SIZE == 1)
  t    = k_tab[tid];
#elif (BARRETT_VECTOR_SIZE == 2)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
#elif (BARRETT_VECTOR_SIZE == 3)
  t.x  = k_tab[tid];
  t.y  = k_tab[tid+1];
  t.z  = k_tab[tid+2];
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
  //tmp  = t * 4620u; // NUM_CLASSES
  //k.d0 = k_base.d0 + tmp;
  //k.d1 = k_base.d1 + mul_hi(t, 4620u) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */
  k.d0 = mad24(t, 4620u, k_base.d0);
  k.d1 = mad_hi(t, 4620u, k_base.d1) + AS_UINT_V((k_base.d0 > k.d0)? 1 : 0);	/* k is limited to 2^64 -1 so there is no need for k.d2 */

  f = upsample(k.d1, k.d0) * 2UL * exp + 1;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("cl_mg64: k_tab[%d]=%x, f=%#llx, shift=%d\n",
        tid, t.s0, f.s0, shiftcount);
#endif

  f_inv = 0 - invmod2pow64(f);

  /* the montgomery repr. As of A can be calculated by either calculating Rs = R^2 mod P
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
   while ((exp&0x80000000) == 0) exp<<=1; // shift exp to the very left of the 32 bits

   As = (0 - f) % f;

   // verify As
   // if (A != (B=mod_REDC64 (As, P, Pinv))) printf("ERROR: A (%llu)!= mod_REDC(As=%llu,1) = %llu\n", A, As, B);

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
#if (BARRETT_VECTOR_SIZE == 1)
  if( a==1 )
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
    printf("cl_mg64: tid=%ld found factor: q=%#llx, k=%x:%x:%x\n", tid, f.s0, k.d2.s0, k.d1.s0, k.d0.s0);
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
#elif (BARRETT_VECTOR_SIZE == 2)
  EVAL_RES_l(x)
  EVAL_RES_l(y)
#elif (BARRETT_VECTOR_SIZE == 3)
  EVAL_RES_l(x)
  EVAL_RES_l(y)
  EVAL_RES_l(z)
#elif (BARRETT_VECTOR_SIZE == 4)
  EVAL_RES_l(x)
  EVAL_RES_l(y)
  EVAL_RES_l(z)
  EVAL_RES_l(w)
#elif (BARRETT_VECTOR_SIZE == 8)
  EVAL_RES_l(s0)
  EVAL_RES_l(s1)
  EVAL_RES_l(s2)
  EVAL_RES_l(s3)
  EVAL_RES_l(s4)
  EVAL_RES_l(s5)
  EVAL_RES_l(s6)
  EVAL_RES_l(s7)
#elif (BARRETT_VECTOR_SIZE == 16)
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

