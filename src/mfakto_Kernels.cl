/*
 All OpenCL kernels for mfakto Trial-Factoring

   is 2^p-1 divisible by q (q=2kp+1)? 
	                        Remove   Optional   
            Square        top bit  mul by 2       mod 47
            ------------  -------  -------------  ------
            1*1 = 1       1  0111  1*2 = 2           2
            2*2 = 4       0   111     no             4
            4*4 = 16      1    11  16*2 = 32        32
            32*32 = 1024  1     1  1024*2 = 2048    27
            27*27 = 729   1        729*2 = 1458      1

    Thus, 2^23 = 1 mod 47. Subtract 1 from both sides. 2^23-1 = 0 mod 47. Since we've shown that 47 is a factor, 2^23-1 is not prime.

 */

// TRACE_KERNEL: higher is more trace, 0-5 currently used
#define TRACE_KERNEL 0

// If above tracing is on, only the thread with the ID below will trace
#define TRACE_TID 0

// safety-check in modulo-functions
// #define BETTER_BE_SAFE_THAN_SORRY

#if (TRACE_KERNEL > 0)
// available on all platforms so far ...
#pragma  OPENCL EXTENSION cl_amd_printf : enable
#pragma  OPENCL EXTENSION cl_khr_fp64 : enable
#endif

void square_64_128(ulong *res_hi, ulong *res_lo, const ulong in
#if (TRACE_KERNEL > 1)
                   , __private uint tid
#endif
)
{
  *res_hi = mul_hi(in, in);
  *res_lo = in * in;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("square_64_128: %llx ^ 2 = %llx : %llx\n", in, *res_hi, *res_lo);
#endif
}

int gte_128(ulong v1_hi, ulong v1_lo, ulong v2_hi, ulong v2_lo)
{
  if (v1_hi == v2_hi)
    return (v1_lo >= v2_lo);
  return (v1_hi >= v2_hi);
}

void sub_128(ulong *v1_hi, ulong *v1_lo, ulong v2_hi, ulong v2_lo
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("sub_128:   %llx:%llx - %llx:%llx = ", *v1_hi, *v1_lo, v2_hi, v2_lo);
#endif
  *v1_hi = *v1_hi - v2_hi - ((*v1_lo < v2_lo) ? 1 : 0);
  *v1_lo = *v1_lo - v2_lo;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("%llx:%llx\n", *v1_hi, *v1_lo);
#endif
}

void sub_if_gte_128(ulong *v1_hi, ulong *v1_lo, ulong v2_hi, ulong v2_lo
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{ /* if (v1 >= v2) v1=v1-v2 */
  ulong tmp_hi, tmp_lo;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("sub_if_gte_128:   %llx:%llx - %llx:%llx = ", *v1_hi, *v1_lo, v2_hi, v2_lo);
#endif
  tmp_lo = *v1_lo - v2_lo;
  tmp_hi = *v1_hi - v2_hi - ((*v1_lo < v2_lo) ? 1 : 0);


  *v1_hi = (tmp_hi > *v1_hi) ? *v1_hi : tmp_hi;
  *v1_lo = (tmp_hi > *v1_hi) ? *v1_lo : tmp_lo;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("%llx:%llx\n", *v1_hi, *v1_lo);
#endif
}

void square_96_192(ulong *res_hi, ulong *res_mid, ulong *res_lo, const ulong in_hi, const ulong in_lo
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{
  __private ulong tmp1, tmp2;   // (in_hi + in_lo) ^2 = in_hi^2 + 2*in_hi*in_lo + in_lo^2
  // PERF: better when using private copies for *res*?
  // PERF: better using 32-bit parts? Or 24-bit?
  *res_lo  = in_lo * in_lo;
  tmp1     = in_lo * in_hi;
  tmp2     = mul_hi(in_lo, in_lo);
  *res_mid = tmp1 << 1;
  *res_hi  = in_hi * in_hi +
             ((*res_mid < tmp1) ? 1 : 0) +   // "carry" from previous left-shift
             (mul_hi(in_lo, in_hi) << 1);    // shift cannot overflow as in_hi uses only 32 of 64 bit.
  *res_mid = *res_mid + tmp2;
  *res_hi  = *res_hi + ((*res_mid < tmp2) ? 1 : 0); // "carry" from above
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf ("square_96_192: %llx : %llx ^ 2 = %llx : %llx : %llx\n", in_hi, in_lo, *res_hi, *res_mid, *res_lo);
#endif
 }
  
// modulo by shift - cmp - sub
ulong mod_128_64_s(__private ulong hi, __private ulong lo, const ulong q, const uint lshift
#if (TRACE_KERNEL > 1)
, __private uint tid
#endif
)
{
  __private int i = clz(q) - clz(hi); // hi is i bitpositions larger than q;  
  __const ulong mask= 0x8000000000000000 ;  // first bit of ulong

#if (TRACE_KERNEL > 2)
  if (q&mask)
  {
    if (tid==TRACE_TID) printf("ERROR: q >= 2^63: %llx (mask=%llx)\n", q, mask);
  }
  if (tid==TRACE_TID) printf("mod_128_64_s: i=%d: hi=%llx, lo=%llx, q=%llx, mask=%llx, shift=%u\n", i, hi, lo, q, mask, lshift);
#endif

#ifdef BETTER_BE_SAFE_THAN_SORRY
  __private ulong a  = q << ( (i>0) ? i : 0);  // a = q shifted to ~same magnitude as hi
  for ( ; i>0 ; i--)  
  {
    hi = hi - ( (hi>a) ? a : 0 );  // subtract multiples of q
    a = a >> 1;                    // slowly shift back until we have q again
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_128_64_s: i=%d: hi= %llx, a=%llx\n", i, hi, a);
#endif
  }
#endif

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("mod_128_64_s: hi= %llx, lo=%llx, q=%llx\n", hi, lo, q);
#endif

 for (i=0; i<64; i++)  //process the 64 bits of lo.  PERF: unroll loop later
  {
    hi = (hi << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster?
    lo = lo << 1;
    hi = hi - ( (hi>q) ? q : 0 );  // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_128_64_s: i=%d: hi= %llx, lo=%llx, a=%llx\n", i, hi, lo, q);
#endif
  }
  hi = hi << lshift;
  hi = hi - ( (hi>q) ? q : 0 );  // subtract q

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mod_128_64_s: return %llx\n", hi);
#endif
  return hi;
}

// modulo by shift - cmp - sub
void mod_192_96_s(__private ulong hi, __private ulong mid, __private ulong lo,
                  __private ulong q_hi, __private ulong q_lo,
                  __private uint lshift,
                  __private ulong *r_hi, __private ulong *r_lo
#if (TRACE_KERNEL > 1)
                  , __private uint tid
#endif
)
{
  __private long i = clz(q_hi) - clz(hi);  // hi is i bitpositions larger than q (q at least 2^63)
  const ulong mask= 0x8000000000000000 ;  // first bit of ulong
  __private ulong a_hi, a_lo;

#if (TRACE_KERNEL > 2)
  if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: %llx:%llx:%llx mod %llx:%llx, shift=%u\n", i, hi, mid, lo, q_hi, q_lo, lshift);
#endif

  if (i>0)
  {
    a_hi = (q_hi << i) | (q_lo >> (64-i));  // a = q shifted to ~same magnitude as hi
    a_lo = q_lo << i;
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%lld: hi= %llx:%llx, a=%llx:%llx\n", i, hi, mid, a_hi, a_lo);
#endif
  }
  for ( ; i>0 ; i--)  
  {
      sub_if_gte_128(&hi, &mid, a_hi, a_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
               );     // subtract multiples of q
      a_lo = (a_lo >> 1) | (a_hi << 63);    // slowly shift back until we have q again
      a_hi = a_hi >> 1;
#if (TRACE_KERNEL > 2)
      if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx:%llx, a=%llx:%llx\n", i, hi, mid, a_hi, a_lo);
#endif
  }
  sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
      if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx:%llx, a=%llx:%llx\n", i, hi, mid, q_hi, q_lo);
#endif

//  for (i=0; i<64; i++)  //process the 64 bits of lo.  PERF: unroll loop later ! unrolled: +10% speed
//  {
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1) + ( (lo & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    lo  = lo << 1;
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_192_96_s: i=%d: hi= %llx, mid=%llx, lo=%llx, q=%llx:%llx\n", i, hi, mid, lo, q_hi, q_lo);
#endif
//  }
  if (lshift)
  {
    hi  = (hi << 1) + ( (mid & mask) ? 1 : 0);  // PERF: mad(2,hi,(lo & mask) ? 1 : 0) faster? or lo >> 63?
    mid = (mid << 1);
    sub_if_gte_128(&hi, &mid, q_hi, q_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );     // subtract q
  }
  *r_hi = hi;
  *r_lo = mid;

#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mod_192_96_s: return %llx:%llx\n", hi, mid);
#endif
}

__kernel void mfakto_cl_64(__private uint exp, __private ulong k_base, __global uint *k_tab, __private ulong4 b_pre_shift, __private int bit_max64, __global uint *RES)
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  // __private long shiftcount = b_pre_shift.w;  // how many bits of exp are left to be processed
  __private ulong pp, hi, lo, k, q, r;
	__private uint tid, mask;
//	__private float qr; /* a little less  than 1/q */

	tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  pp = exp;             // 32 -> 64 bit
  k = k_tab[tid];       // 32 -> 64 bit
  k = k*4620 + k_base;  // NUM_CLASSES
	q = (pp<<1) * k + 1;  // q = 2*k*exp+1
	                      // the first bits of exp are processed on the host w/o modulo,
                        // as the result of the squaring was less than the FC anyway.
  /* now only to bit_max^2 of the kernel 
                        // preprocessing is now done as long as it fits into 192 bits w/o modulo

  mod_192_96_s(b_pre_shift.z, b_pre_shift.y, b_pre_shift.x, q, 0, 0, &hi, &lo   // 192 bit with q < 2^64 is more than the mod can handle
#if (TRACE_KERNEL > 1)                              // Do a mod (q<<64) first to bring b down to 128 bits
                   , tid
#endif
              );  // initial modulo of the precomputed residue  */

  r = mod_128_64_s(b_pre_shift.y, b_pre_shift.x, q, 0
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                  ); // r = hi:lo  % q
 // and now again the real modulo

	mask = 1<<(b_pre_shift.w); /* the 1 in mask now points to the bit-pos after the first modulo was necessary */
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_64: tid=%ld: p=%llx, k=%llx, q=%llx, mask=%llx, r=%llx\n", tid, pp, k, q, mask, r);
#endif
	while (mask)
	{
	  square_64_128(&hi, &lo, r
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 ); /*hi:lo = (r * r); */
    r = mod_128_64_s(hi, lo, q, ( pp&mask ) ? 1 : 0
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                    ); // r = hi:lo << 0 or 1   % q
	  mask = mask >> 1;   // next bit of p
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_64: q=%llx, mask=%llx, r=%llx\n", q, mask, r);
#endif
	}
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakto_cl_64: tid=%ld: q=%llx, k=%llx, r=%llx\n", tid, q, k, r);
#endif

/* finally check if we found a factor and write the factor to RES[] */
  if(r==1)
  {
#if (TRACE_KERNEL > 0)  // will trace for any thread
      printf("mfakto_cl_64: tid=%ld found factor: q=%llx, k=%llx, r=%llx\n", tid, q, k, r);
#endif
      tid=atomic_inc(&RES[0]);
      if(tid<10)				/* limit to 10 factors per class */
      {
        RES[tid*3 + 1]= 0;
        RES[tid*3 + 2]= (uint) (q >> 32);
        RES[tid*3 + 3]= (uint) q & 0xFFFFFFFF;
      }
  }
}

// this kernel is only used for a quick test at startup - no need to be correct ;-)
__kernel void mod_128_64_k(const ulong hi, const ulong lo, const ulong q, const float qr, __global uint *res
#if (TRACE_KERNEL > 1)
                  , __private uint tid
#endif
)
{
 // __local ulong f;
  __local uint i;
  __local uint4 j,k;
  k=1;
  //f = convert_ulong((convert_float(hi)*18446744073709551616.0f + convert_float(lo) ) * qr);
  /* if everything goes well, then mul_hi(q,f) == hi, so we could skip that */
  //res[0] = hi - mul_hi(q,f);
  //res[1] = lo - f * q;
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mod_128_64_k: q=%llx: res= %llx : %llx  f=%llx, hi=%llx, lo=%llx\n", q, res[0], res[1], f, hi, lo);
//  printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f);
#endif
  // tests for comparing different inplementations of the same ...

#define STMT(a) a+i

  for (i=0; i<0x80000; i++) // 2G loops
  { // cpu: core i7 2GHz | 3.3 GHz, gpu: HD5750
  // 0. baseline
    j=STMT(k);  // cpu: 7.61s | 4.69s
    k=STMT(i);  // gpu: 7.5 us
    j.x=STMT(k.y);
    k.z=STMT(k.x);
    j.w=STMT(j.y);
    k.x=STMT(j.z);
  // 1. (x & 0xffffff) vs.  ((x<<8)>>8)
//    j=(k+i) & 0xffffff; // cpu: 9.22s | 5.48s
//    k=(i+i) & 0xffffff;  // gpu: 7.5 us
//    j=((k+i)<<8)>>8;  // cpu: 9.22s | 5.49s
//    k=((i+i)<<8)>>8;
  // 2. different multiplications
//    j=k*i;   // cpu: 9.85s | 5.71s
//    k=i*i; 
//    j=mul24(k,i); // cpu: 9.93s | 5.71s
//    k=mul24(i,i); 
//    j=mul_hi(k,i); // cpu: 9.4s | 5.37s
//    k=mul_hi(i,i); 
  // 3. c? b: a  vs.  select(a,b,c)
//    j= (k&4)?1:0 + (i&8)?1:0; // cpu: 9.09s | 5.58s
//    k= (i&4)?1:0 + (i&2)?1:0;
//    j= select(0,1,(k&4)) + select(0,1,(i&8)); // cpu: 8.31s | 5.04s
//    k= select(0,1,(i&4)) + select(0,1,(i&2));
  }
  res[0]=j.x;
  res[1]=j.y;
  res[2]=j.z;
  res[3]=j.w;
  res[4]=k.x;
  res[5]=k.y;
  res[6]=k.z;
  res[7]=k.w;
  res[8]=i;
}

__kernel void mfakto_cl_95(__private uint exp, __private ulong k_base, __global uint *k_tab, __private ulong4 b_pre_shift, __private int bit_max64, __global uint *RES)
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  // __private long shiftcount = b_pre_shift.w;  // how many bits of exp are left to be processed
  __private ulong q_lo, q_hi, r_lo, r_hi, pp, hi, mid, lo, k;
	__private uint tid, mask;
//	__private float qr; /* a little less  than 1/q */

	tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  pp = exp;             // 32 -> 64 bit
  k = k_tab[tid];       // 32 -> 64 bit
  k = k*4620 + k_base;  // NUM_CLASSES
	q_lo = pp<<1;
  q_hi = mul_hi(q_lo, k);
  q_lo = q_lo * k + 1;  // q = 2*k*exp+1
	lo  = b_pre_shift.x;  // the first bits of exp are processed on the host w/o modulo,
  mid = b_pre_shift.y;  // as the result of the squaring was less than the FC anyway.
  hi  = b_pre_shift.z;  // preprocessing is done as long as it fits into 192 bits w/o modulo

  mod_192_96_s(hi, mid, lo, q_hi, q_lo, 0, &r_hi, &r_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);  // initial modulo of the precomputed residue

	mask = 1<<(b_pre_shift.w); /* the 1 on mask now points to the bit-pos after the first modulo was necessary */
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_95: tid=%ld: p=%llx, k=%llx, q=%llx:%llx, mask=%llx, r=%llx:%llx\n", tid, pp, k, q_hi, q_lo, mask, r_hi, r_lo);
#endif
    

//  qr = as_float(0x3f7ffffb) / convert_float(q);

	while (mask)
	{
	  square_96_192(&hi, &mid, &lo, r_hi, r_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                 ); /*hi:mid:lo = (r * r); */
    mod_192_96_s(hi, mid, lo, q_hi, q_lo, ( exp&mask ) ? 1 : 0, &r_hi, &r_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
                );
	  mask = mask >> 1;
#if (TRACE_KERNEL > 1)
    if (tid==TRACE_TID) printf("mfakto_cl_95: q=%llx:%llx, mask=%llx, r=%llx:%llx\n", q_hi, q_lo, mask, r_hi, r_lo);
#endif
	}
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakto_cl_95: tid=%ld: q=%llx:%llx, k=%llx, r=%llx:%llx\n", tid, q_hi, q_lo, k, r_hi, r_lo);
#endif

/* finally check if we found a factor and write the factor to RES[] */
  if((r_hi==0) && (r_lo==1))
  {
#if (TRACE_KERNEL > 0)  // trace this for any thread
      printf("mfakto_cl_95: tid=%ld found factor: q=%llx:%llx, k=%llx, r=%llx:%llx\n", tid, q_hi, q_lo, k, r_hi, r_lo);
#endif
      tid=atomic_inc(&RES[0]);
      if(tid<10)				/* limit to 10 factors per class */
      {
        RES[tid*3 + 1]= (uint) q_hi & 0xFFFFFFFF;
        RES[tid*3 + 2]= (uint) (q_lo >> 32);
        RES[tid*3 + 3]= (uint) q_lo & 0xFFFFFFFF;
      }
  }
}
/*==========================================*/
void div_192_96(ulong *res_hi, ulong *res_lo, ulong q_hi, ulong q_mid, ulong q_lo, ulong n_hi, ulong n_lo, float nf)
{  // *res = floor(q / n)   
  *res_hi = q_hi / n_hi + q_mid;   // Dummy to have some non-zero values ...
  *res_lo = q_mid / n_lo + q_lo;
}

/*==========================================*/
void mul_96_192_no_low2(ulong *res_hi, ulong *res_mid, ulong *res_lo, ulong a_hi, ulong a_lo, ulong b_hi, ulong b_lo)
{  // *res = (approx.) a * b
  ulong t1;                // PERF: faster when using local copies for res_mid and ho?

  *res_lo = a_lo * b_lo;   // PERF: not needed
  t1 = mul_hi(a_lo, b_lo);
  *res_mid = a_hi * b_lo + t1;
  *res_hi  = a_hi * b_hi + mul_hi(a_hi, b_lo) + mul_hi(a_lo, b_hi) + ((*res_mid < t1) ? 1 : 0);
  t1 = a_lo * b_hi;
  *res_mid += t1;
  *res_hi += ((*res_mid < t1) ? 1 : 0);
}

/*==========================================*/
void mul_96(ulong *res_hi, ulong *res_lo, ulong a_hi, ulong a_lo, ulong b_hi, ulong b_lo)
{ // *res = (lower half of) a * b
  *res_lo = a_lo * b_lo;   
  *res_hi = a_hi * b_lo + a_lo * b_hi + mul_hi(a_lo, b_lo);
}

void mod_simple_96(ulong *res_hi, ulong * res_lo, ulong a_hi, ulong a_lo, ulong f_hi, ulong f_lo, float ff
#if (TRACE_KERNEL > 1)
                  , __private uint tid
#endif
)
{ // *res = a mod f, a < 6f
  sub_if_gte_128(&a_hi, &a_lo, f_hi, f_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );
  sub_if_gte_128(&a_hi, &a_lo, f_hi, f_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );
  sub_if_gte_128(&a_hi, &a_lo, f_hi, f_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );
  sub_if_gte_128(&a_hi, &a_lo, f_hi, f_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );
  sub_if_gte_128(&a_hi, &a_lo, f_hi, f_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );
  sub_if_gte_128(&a_hi, &a_lo, f_hi, f_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
             );
  *res_hi = a_hi;
  *res_lo = a_lo;
}

/*===========uint exp, ulong k_base, __global uint *k_tab, ulong4 b_pre_shift, __global uint *RES=========================================================*/
__kernel void mfakto_cl_barrett92(__private uint exp, __private ulong k_base, __global uint *k_tab, __private ulong4 b_pre_shift, __private int bit_max64, __global uint *RES)
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  __private int shiftcount = b_pre_shift.w;  // how many bits of exp are left to be processed
  __private ulong f_lo, f_hi, a_lo, a_hi, pp, b_hi, b_mid, b_lo, k, u_hi, u_lo;
	__private uint tid;

	tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  pp = exp;             // 32 -> 64 bit
  pp = pp<<1;
  k = k_tab[tid];       // 32 -> 64 bit
  k = k*4620 + k_base;  // NUM_CLASSES
  f_hi = mul_hi(pp, k);
  f_lo = pp * k + 1;    // f = 2*k*exp+1
	b_lo  = b_pre_shift.x;  // the first bits of exp are processed on the host w/o modulo,
  b_mid = b_pre_shift.y;  // as the result of the squaring was less than the FC anyway.
  b_hi  = b_pre_shift.z;  // preprocessing is done as long as it fits into 192 bits w/o modulo

  ulong bigtmp_hi, bigtmp_mid, bigtmp_lo;
  ulong tmp_hi, tmp_lo;
  float ff;

#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakt_cl_barrett92: tid=%ld, f=%llx:%llx, k=%llx, b=%llx:%llx:%llx\n", tid, f_hi, f_lo, k, b_hi, b_mid, b_lo);
#endif

/*
ff = f as float, needed in mod_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= convert_float(f_hi);
  ff= ff * 4294967296.0f + convert_float(f_lo>>32);		// low 32 bits ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff=as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we allways underestimate the quotient
        
  bigtmp_hi  = 1ULL << (bit_max64 << 1);			// bigtmp = 2^(2*bit_max)
  bigtmp_mid = 0;
  bigtmp_lo  = 0;

  div_192_96(&u_hi, &u_lo, bigtmp_hi, bigtmp_mid, bigtmp_lo, f_hi, f_lo, ff);		// u = floor(bigtmp / f) = floor(2^(2*bit_max) / f)

  a_hi = (b_hi  >> bit_max64);
  a_lo = (b_mid >> bit_max64) + (b_hi << bit_max64);          // a = b / (2^bit_max)

  // a.d0 = (b.d2 >> bit_max64) + (b.d3 << bit_max64_32);			// a = b / (2^bit_max)
  // a.d1 = (b.d3 >> bit_max64) + (b.d4 << bit_max64_32);
  // a.d2 = (b.d4 >> bit_max64) + (b.d5 << bit_max64_32);

  mul_96_192_no_low2(&bigtmp_hi, &bigtmp_mid, &bigtmp_lo, a_hi, a_lo, u_hi, u_lo);	// bigtmp = (b / (2^bit_max)) * u 
                                                                                    // approx.=  b * (2^bit_max) / f
  a_hi = (bigtmp_hi  >> bit_max64);
  a_lo = (bigtmp_mid >> bit_max64) + (bigtmp_hi  << bit_max64); // a = ((b / (2^bit_max)) * u) / (2^bit_max) = b / f (approx.)

  // a.d0 = (tmp192.d2 >> bit_max64) + (tmp192.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  // a.d1 = (tmp192.d3 >> bit_max64) + (tmp192.d4 << bit_max64_32);
  // a.d2 = (tmp192.d4 >> bit_max64) + (tmp192.d5 << bit_max64_32);

  mul_96(&tmp_hi, &tmp_lo, a_hi, a_lo, f_hi, f_lo);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

  tmp_hi = b_mid - tmp_hi - ((b_lo < tmp_lo) ? 1 : 0);
  tmp_lo = b_lo - tmp_lo;
  
  // tmp96.d0 = __sub_cc (b.d0, tmp96.d0);					// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  // tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
  // tmp96.d2 = __subc   (b.d2, tmp96.d2);

  mod_simple_96(&a_hi, &a_lo, tmp_hi, tmp_lo, f_hi, f_lo, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
  
  exp<<= 32 - shiftcount;
  while(exp)
  {
	  square_96_192(&b_hi, &b_mid, &b_lo, a_hi, a_lo
#if (TRACE_KERNEL > 1)
                   , tid
#endif
       ); // b = a^2

    a_hi = (b_hi  >> bit_max64);
    a_lo = (b_mid >> bit_max64) + (b_hi << bit_max64);          // a = b / (2^bit_max)


  //  a.d0 = (b.d2 >> bit_max64) + (b.d3 << bit_max64_32);		// a = b / (2^bit_max)
  //  a.d1 = (b.d3 >> bit_max64) + (b.d4 << bit_max64_32);
  //  a.d2 = (b.d4 >> bit_max64) + (b.d5 << bit_max64_32);

    mul_96_192_no_low2(&bigtmp_hi, &bigtmp_mid, &bigtmp_lo, a_hi, a_lo, u_hi, u_lo);	// bigtmp = (b / (2^bit_max)) * u 
                                                                                    // approx.=  b * (2^bit_max) / f

  //  mul_96_192_no_low2(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)

    a_hi = (bigtmp_hi  >> bit_max64);
    a_lo = (bigtmp_mid >> bit_max64) + (bigtmp_hi  << bit_max64); // a = ((b / (2^bit_max)) * u) / (2^bit_max) = b / f (approx.)

  //  a.d0 = (tmp192.d2 >> bit_max64) + (tmp192.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  //  a.d1 = (tmp192.d3 >> bit_max64) + (tmp192.d4 << bit_max64_32);
  //  a.d2 = (tmp192.d4 >> bit_max64) + (tmp192.d5 << bit_max64_32);

    mul_96(&tmp_hi, &tmp_lo, a_hi, a_lo, f_hi, f_lo);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

  //  mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

    tmp_hi = b_mid - tmp_hi - ((b_lo < tmp_lo) ? 1 : 0);
    tmp_lo = b_lo - tmp_lo;

  //  tmp96.d0 = __sub_cc (b.d0, tmp96.d0);				// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  //  tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
  //  tmp96.d2 = __subc   (b.d2, tmp96.d2);
    
    if (exp&0x80000000)
    {
      tmp_hi = (tmp_hi << 1) | (tmp_lo >> 63);
      tmp_lo = tmp_lo << 1;
    }

  //  if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

    mod_simple_96(&a_hi, &a_lo, tmp_hi, tmp_lo, f_hi, f_lo, ff
#if (TRACE_KERNEL > 1)
                  , tid
#endif
);					// adjustment, plain barrett returns N = AB mod M where N < 3M!

  //  mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!

    exp<<=1;
  }

  if (gte_128(a_hi, a_lo, f_hi, f_lo))
  {
    a_hi = a_hi - f_hi - ((a_lo < f_lo) ? 1 : 0);
    a_lo = a_lo - f_lo;
  }
  // if(cmp_96(a,f)>0)							// final adjustment
  // {
  //  sub_96(&a, a, f);
  // }
  
/* finally check if we found a factor and write the factor to RES[] */
  if( (a_hi==0) && (a_lo==1) )
  {
#if (TRACE_KERNEL > 0)
      printf("mfakt_cl_barrett92: tid=%ld found factor: f=%llx:%llx, k=%llx, a=%llx:%llx\n", tid, f_hi, f_lo, k, a_hi, a_lo);
#endif
      tid=atomic_inc(&RES[0]);
      if(tid<10)				/* limit to 10 factors per class */
      {
        RES[tid*3 + 1]= (uint) f_hi & 0xFFFFFFFF;
        RES[tid*3 + 2]= (uint) (f_lo >> 32);
        RES[tid*3 + 3]= (uint) f_lo & 0xFFFFFFFF;
      }
  }
}

/*
 * 24-bit-stuff for the 71-bit-kernel
 *
 */

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct
{
  uint d0,d1,d2;
}int72_t;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct
{
  uint d0,d1,d2,d3,d4,d5;
}int144_t;


void mul_24_48(uint *res_hi, uint *res_lo, uint a, uint b)
/* res_hi*(2^24) + res_lo = a * b */
{ // PERF: inline its use
/* thats how it should be, but the mul24_hi is missing ...
  *res_lo = mul24(a,b) & 0xFFFFFF;
  *res_hi = mul24_hi(a,b) >> 8;       // PERF: check for mul24_hi
  */
  *res_lo  = mul24(a,b);
  *res_hi  = (mul_hi(a,b) << 8) | (*res_lo >> 24);       // PERF: check for mul24_hi
  *res_lo &= 0xFFFFFF;
}


void copy_72(int72_t *a, int72_t b)
/* a = b */
{
  a->d0 = b.d0;
  a->d1 = b.d1;
  a->d2 = b.d2;
}

int gte_72(int72_t a, int72_t b)
/* returns
0  if a < b
1  if a >= b */
{
if (a.d2 == b.d2)
  if (a.d1 == b.d1)
    return (a.d0 > b.d0);
  else
    return (a.d1 > b.d1);
else
  return (a.d2 > b.d2);
}

int cmp_72(int72_t a, int72_t b)
/* returns
-1 if a < b
0  if a = b
1  if a > b */
{
  if(a.d2 < b.d2)return -1;
  if(a.d2 > b.d2)return 1;
  if(a.d1 < b.d1)return -1;
  if(a.d1 > b.d1)return 1;
  if(a.d0 < b.d0)return -1;
  if(a.d0 > b.d0)return 1;
  return 0;
}


void sub_72(int72_t *res, int72_t a, int72_t b)
/* a must be greater or equal b!
res = a - b */
{
  /*
  res->d0 = __sub_cc (a.d0, b.d0) & 0xFFFFFF;
  res->d1 = __subc_cc(a.d1, b.d1) & 0xFFFFFF;
  res->d2 = __subc   (a.d2, b.d2) & 0xFFFFFF;
  */
  res->d0 = (a.d0 - b.d0) & 0xFFFFFF;
  res->d1 = (a.d1 - b.d1 - ((b.d0 > a.d0) ? 1 : 0)) & 0xFFFFFF;
  res->d2 = (a.d2 - b.d2 - ((res->d1 > a.d1) ? 1 : 0)) & 0xFFFFFF;
}

int72_t sub_if_gte_72(int72_t a, int72_t b)
/* return (a>b)?a-b:a */
{
  int72_t tmp;
  /* do the subtraction and use tmp.d2 to decide if the result is valid (if a was > b) */

  tmp.d0 = (a.d0 - b.d0) & 0xFFFFFF;
  tmp.d1 = (a.d1 - b.d1 - ((b.d0 > a.d0) ? 1 : 0)) & 0xFFFFFF;
  tmp.d2 = (a.d2 - b.d2 - ((tmp.d1 > a.d1) ? 1 : 0)) & 0xFFFFFF;

  return (tmp.d2 > a.d2) ? a : tmp;
}

void mul_72(int72_t *res, int72_t a, int72_t b)
/* res = (a * b) mod (2^72) */
{
  uint hi,lo; // PERF: inline mul_24_48

  mul_24_48(&hi, &lo, a.d0, b.d0);
  res->d0 = lo;
  res->d1 = hi;

  mul_24_48(&hi, &lo, a.d1, b.d0);
  res->d1 += lo;
  res->d2 = hi;

  mul_24_48(&hi, &lo, a.d0, b.d1);
  res->d1 += lo;
  res->d2 += hi;

  res->d2 += mul24(a.d2,b.d0);

  res->d2 += mul24(a.d1,b.d1);

  res->d2 += mul24(a.d0,b.d2);

//  no need to carry res->d0

  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d2 &= 0xFFFFFF;
}


void square_72_144(int144_t *res, int72_t a)
/* res = a^2 */
{ // PERF: use local copy for intermediate res->...?
  uint tmp;

  tmp      =  mul24(a.d0, a.d0);
  res->d1  = (mul_hi(a.d0, a.d0) << 8) | (tmp >> 24);
  res->d0  =  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
  res->d2  = (mul_hi(a.d1, a.d0) << 9) | (tmp >> 23);
  res->d1 += (tmp << 1) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d0);
  res->d3  = (mul_hi(a.d2, a.d0) << 9) | (tmp >> 23);
  res->d2 += (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 += (mul_hi(a.d1, a.d1) << 8) | (tmp >> 24);
  res->d2 +=  tmp       & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  = (mul_hi(a.d2, a.d1) << 9) | (tmp >> 23);
  res->d3 += (tmp << 1) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d2);
  res->d5  = (mul_hi(a.d2, a.d2) << 8) | (tmp >> 24);
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


void square_72_144_shl(int144_t *res, int72_t a)
/* res = 2* a^2 */
{ // PERF: use local copy for intermediate res->...?
  uint tmp;

  tmp      =  mul24(a.d0, a.d0);
  res->d1  = (mul_hi(a.d0, a.d0) << 9) | (tmp >> 23);
  res->d0  = (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d0);
  res->d2  = (mul_hi(a.d1, a.d0) << 10)| (tmp >> 22);
  res->d1 += (tmp << 2) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d0);
  res->d3  = (mul_hi(a.d2, a.d0) << 10)| (tmp >> 22);
  res->d2 += (tmp << 2) & 0xFFFFFF;
  
  tmp      =  mul24(a.d1, a.d1);
  res->d3 += (mul_hi(a.d1, a.d1) << 9) | (tmp >> 23);
  res->d2 += (tmp << 1) & 0xFFFFFF;
  
  tmp      =  mul24(a.d2, a.d1);
  res->d4  = (mul_hi(a.d2, a.d1) << 10)| (tmp >> 22);
  res->d3 += (tmp << 2) & 0xFFFFFF;

  tmp      =  mul24(a.d2, a.d2);
  res->d5  = (mul_hi(a.d2, a.d2) << 9) | (tmp >> 23);
  res->d4 += (tmp << 1) & 0xFFFFFF;

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


void mod_144_72(int72_t *res, int144_t q, int72_t n, float nf
#if (TRACE_KERNEL > 1)
                   , __private uint tid
#endif
)
/* res = q mod n */
{
  float qf;
  uint  qi, tmp;
  int144_t nn={0};

/********** Step 1, Offset 2^51 (2*24 + 3) **********/
  qf= convert_float_rte(q.d5);
  qf= qf * 16777216.0f + convert_float_rte(q.d4);
  qf= qf * 16777216.0f + convert_float_rte(q.d3);
  qf*= 2097152.0f;

  qi=convert_uint(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#1: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
//    if (tid==TRACE_TID) printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", qf, qf, qf, qf, qf, qf, qf, qf, qf, qf, qf, qf);
 //   if (tid==TRACE_TID) printf("g: %g, G: %G, #g %#g, #G %#G, f %f, F %F, #f %#f, #F %#F, e %e, E %E, #e %#e, #E %#E\n", nf, nf, nf, nf, nf, nf, nf, nf, nf, nf, nf, nf);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#1: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, qi);
#endif

//  nn.d0=0;
//  nn.d1=0;
// nn = n * qi AND shiftleft 3 bits at once, carry is done later
  tmp    =  mul24(n.d0, qi);
  nn.d3  = (mul_hi(n.d0, qi) << 11) | (tmp >> 21);
  nn.d2  = (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d1, qi);
  nn.d4  = (mul_hi(n.d1, qi) << 11) | (tmp >> 21);
  nn.d3 += (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d2, qi);
  nn.d5  = (mul_hi(n.d2, qi) << 11) | (tmp >> 21);
  nn.d4 += (tmp << 3) & 0xFFFFFF;
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#1.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
#endif


/* do carry */
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#1: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
#endif

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions
  q.d2 = __sub_cc (q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5); */
  q.d2 = q.d2 - nn.d2;
  q.d3 = q.d3 - nn.d3 - ((q.d2 > 0xFFFFFF)?1:0);
  q.d4 = q.d4 - nn.d4 - ((q.d3 > 0xFFFFFF)?1:0);
  q.d5 = q.d5 - nn.d5 - ((q.d4 > 0xFFFFFF)?1:0);
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 2, Offset 2^31 (1*24 + 7) **********/
  qf= convert_float_rte(q.d5);
  qf= qf * 16777216.0f + convert_float_rte(q.d4);
  qf= qf * 16777216.0f + convert_float_rte(q.d3);
  qf= qf * 16777216.0f + convert_float_rte(q.d2);
  qf*= 131072.0f;

  qi=convert_uint(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#2: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
    //if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", 0.0f, 1.0f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#2: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, qi);
#endif

//  nn.d0=0;
// nn = n * qi AND shiftleft 7 bits at once, carry is done later

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d0, qi);
  nn.d2  = (mul_hi(n.d0, qi) << 15) | (tmp >> 17);
  nn.d1  = (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d1, qi);
  nn.d3  = (mul_hi(n.d1, qi) << 15) | (tmp >> 17);
  nn.d2 += (tmp << 7) & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp    =  mul24(n.d2, qi);
  nn.d4  = (mul_hi(n.d2, qi) << 15) | (tmp >> 17);
  nn.d3 += (tmp << 7) & 0xFFFFFF;
#if (TRACE_KERNEL > 2)
  nn.d5=0;
#endif
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#2.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

/* do carry */
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
#if (TRACE_KERNEL > 2)
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;
#endif

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#2: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
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
  q.d2 = q.d2 - nn.d2 - ((q.d1 > 0xFFFFFF)?1:0);
  q.d3 = q.d3 - nn.d3 - ((q.d2 > 0xFFFFFF)?1:0);
  q.d4 = q.d4 - nn.d4 - ((q.d3 > 0xFFFFFF)?1:0);
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;
  q.d4 &= 0xFFFFFF;

/********** Step 3, Offset 2^11 (0*24 + 11) **********/
  qf= convert_float_rte(q.d4);
  qf= qf * 16777216.0f + convert_float_rte(q.d3);
  qf= qf * 16777216.0f + convert_float_rte(q.d2);
  qf= qf * 16777216.0f + convert_float_rte(q.d1);
  qf*= 8192.0f;

  qi=convert_uint(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#3: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
    // if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", -1.0e10f, 3.2e8f, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#3: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, qi);
#endif

//nn = n * qi, shiftleft is done later
/*  nn.d0 =                                  mul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (mul_hi(n.d0, qi) >> 8, mul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d2 = __addc_cc(mul_hi(n.d1, qi) >> 8, mul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (mul_hi(n.d2, qi) >> 8, 0); */

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d0, qi);
  nn.d1 = (mul_hi(n.d0, qi) << 8) | (tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d1, qi);
  nn.d2 = (mul_hi(n.d1, qi) << 8) | (tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;
 
#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d2, qi);
  nn.d3 = (mul_hi(n.d2, qi) << 8) | (tmp >> 24);
  nn.d2 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#3.3: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  /* do carry */
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#3: before shl(11): nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif
// shiftleft 11 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                           nn.d3>>13;
  nn.d3 = ((nn.d3 & 0x1FFF)<<11) + (nn.d2>>13);
#else  
  nn.d3 = ( nn.d3          <<11) + (nn.d2>>13);	// we don't need to clear top bits here, this is done during q = q - nn
#endif  
  nn.d2 = ((nn.d2 & 0x1FFF)<<11) + (nn.d1>>13);
  nn.d1 = ((nn.d1 & 0x1FFF)<<11) + (nn.d0>>13);
  nn.d0 = ((nn.d0 & 0x1FFF)<<11);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
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
  q.d1 = q.d1 - nn.d1 - ((q.d0 > 0xFFFFFF)?1:0);
  q.d2 = q.d2 - nn.d2 - ((q.d1 > 0xFFFFFF)?1:0);
  q.d3 = q.d3 - nn.d3 - ((q.d2 > 0xFFFFFF)?1:0);
  q.d0 &= 0xFFFFFF;
  q.d1 &= 0xFFFFFF;
  q.d2 &= 0xFFFFFF;
  q.d3 &= 0xFFFFFF;

/********** Step 4, Offset 2^0 (0*24 + 0) **********/

  qf= convert_float_rte(q.d3);
  qf= qf * 16777216.0f + convert_float_rte(q.d2);
  qf= qf * 16777216.0f + convert_float_rte(q.d1);
  qf= qf * 16777216.0f + convert_float_rte(q.d0);

  qi=convert_uint(qf*nf);

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_%d%d_%d#4: qf=%#G, nf=%#G, *=%#G, qi=%d\n", 1, 44, 72, qf, nf, qf*nf, qi);
    //if (tid==TRACE_TID) printf("mod_144_72: qf=%#G, nf=%#G, qi=%d\n", qf, nf, qi);
#endif

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, q.d2, q.d1, q.d0, n.d2, n.d1, n.d0, qi);
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
  if (tid==TRACE_TID) printf("mod_144_72#4.0: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d0, qi);
  nn.d1 = (mul_hi(n.d0, qi) << 8) | (tmp >> 24);
  nn.d0 = tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.1: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  tmp   = mul24(n.d1, qi);
  nn.d2 = (mul_hi(n.d1, qi) << 8) | (tmp >> 24);
  nn.d1 += tmp & 0xFFFFFF;

#if (TRACE_KERNEL > 4)
  if (tid==TRACE_TID) printf("mod_144_72#4.2: nn=%x:%x:%x:%x:%x:%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0);
#endif

  nn.d2 += mul24(n.d2, qi);
  nn.d2 += nn.d1 >> 24; nn.d1 &= 0xFFFFFF;

#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mod_144_72#4.3: nn=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        nn.d5, nn.d4, nn.d3, nn.d2, nn.d1, nn.d0, n.d2, n.d1, n.d0, qi);
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
  q.d1 = q.d1 - nn.d1 - ((q.d0 > 0xFFFFFF)?1:0);
  q.d2 = q.d2 - nn.d2 - ((q.d1 > 0xFFFFFF)?1:0);

  res->d0 = q.d0 & 0xFFFFFF;
  res->d1 = q.d1 & 0xFFFFFF;
  res->d2 = q.d2 & 0xFFFFFF;

#if (TRACE_KERNEL > 2)
    if (tid==TRACE_TID) printf("mod_144_72#4: q=%x:%x:%x:%x:%x:%x, n=%x:%x:%x, qi=%x\n",
        q.d5, q.d4, q.d3, res->d2, res->d1, res->d0, n.d2, n.d1, n.d0, qi);
#endif

/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
/*  if(cmp_72(*res,n)>0)
  {
    sub_72(&tmp72,*res,n);
    copy_72(res,tmp72);
  }*/
}


__kernel void mfakto_cl_71(__private uint exp, __private int72_t k, __global uint *k_tab, __private int shiftcount, __private int144_t b, __global uint *RES)
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  int72_t exp72,f;
  int72_t a;
  int tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  float ff;

  exp72.d2=0;exp72.d1=exp>>23;exp72.d0=(exp&0x7FFFFF)<<1;	// exp72 = 2 * exp

  mul_24_48(&(a.d1),&(a.d0),k_tab[tid],4620); // NUM_CLASSES
  k.d0 += a.d0;
  k.d1 += a.d1;
  k.d1 += k.d0 >> 24; k.d0 &= 0xFFFFFF;
  k.d2 += k.d1 >> 24; k.d1 &= 0xFFFFFF;		// k = k + k_tab[tid] * NUM_CLASSES

  mul_72(&f,k,exp72);				// f = 2 * k * exp
  f.d0 += 1;				      	// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_144_72().
Precalculated here since it is the same for all steps in the following loop */
  ff= convert_float(f.d2);
  ff= ff * 16777216.0f + convert_float(f.d1);
  ff= ff * 16777216.0f + convert_float(f.d0);

//  ff=0.9999997f/ff;
//  ff=__int_as_float(0x3f7ffffc) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
  ff=as_float(0x3f7ffffb) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
 
#if (TRACE_KERNEL > 1)
  if (tid==TRACE_TID) printf("mfakto_cl_71: tid=%ld: p=%x, *2 =%x:%x, k=%x:%x:%x, f=%x:%x:%x, shift=%d, b=%x:%x:%x:%x:%x:%x\n",
                              tid, exp, exp72.d1, exp72.d0, k.d2, k.d1, k.d0, f.d2, f.d1, f.d0, shiftcount, b.d5, b.d4, b.d3, b.d2, b.d1, b.d0);
#endif

  mod_144_72(&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
);			// a = b mod f
  exp<<= 32 - shiftcount;
  while(exp)
  {
    if(exp&0x80000000)square_72_144_shl(&b,a);	// b = 2 * a^2 ("optional multiply by 2" in Prime 95 documentation)
    else              square_72_144(&b,a);	// b = a^2
#if (TRACE_KERNEL > 3)
  if (tid==TRACE_TID) printf("mfakto_cl_71: exp=%x,  %x:%x:%x ^2 (shl:%d) = %x:%x:%x:%x:%x:%x\n",
                              exp, a.d2, a.d1, a.d0, (exp&0x80000000?1:0), b.d5, b.d4, b.d3, b.d2, b.d1, b.d0);
#endif
    mod_144_72(&a,b,f,ff
#if (TRACE_KERNEL > 1)
                   , tid
#endif
      );			// a = b mod f
    exp<<=1;
  }
#if (TRACE_KERNEL > 0)
  if (tid==TRACE_TID) printf("mfakto_cl_71 result: f=%x:%x:%x, a=%x:%x:%x\n",
                              f.d2, f.d1, f.d0, a.d2, a.d1, a.d0);
#endif

  a=sub_if_gte_72(a,f);
/*  if(gte_72(a,f))
  {
    sub_72(&exp72,a,f);
    copy_72(&a,exp72);
  } */

/* finally check if we found a factor and write the factor to RES[] */
  if((a.d2|a.d1)==0 && a.d0==1)
  {
    if ((f.d2|f.d1)!=0 || f.d0 != 1)  // happens for k=0 and does not help us ;-)
    {
#if (TRACE_KERNEL > 0)  // trace this for any thread
      printf("mfakto_cl_71: tid=%ld found factor: q=%x:%x:%x, k=%x:%x:%x\n", tid, f.d2, f.d1, f.d0, k.d2, k.d1, k.d0);
#endif
      tid=atomic_inc(&RES[0]);
      if(tid<10)				/* limit to 10 factors per class */
      {
        RES[tid*3 + 1]=f.d2;
        RES[tid*3 + 2]=f.d1;
        RES[tid*3 + 3]=f.d0;
      }
    }
  }
}
