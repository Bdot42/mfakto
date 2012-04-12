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

Version 0.11pre3
*/

#define CL_SIEVE_DEBUG 0   

#if (CL_SIEVE_DEBUG > 0)
// available on all platforms so far ...
#pragma  OPENCL EXTENSION cl_amd_printf : enable
#endif

__kernel void mfakto_cl_sieve_init(__private uint exp,
                                   __private ulong k_base,
                                   __constant uint *primes,        // primes used for sieving, start with primes[0]=13
                                   __global  uint *next_multiple,  // out-array of k-offsets when the corresponding prime divides the factor candidate
                                   __private uint vector_size)     // not yet used
{
  __private uint ii, jj, k=0;
  __private uint tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  __private uint p = primes[tid];                  // p is up to 22 bits when sieving up to 200000 primes

  /* find k for  2*(k*4620+k_base)*exp+1 = 0 mod primes[tid] */

  /* second version of sieve.c as it has the fewest mod-operations */
  /* when sieving only the first ~3500 primes, the calculations below can be done in 32 bit: 25% speedup */

  ii=(2 * (exp%p) * (k_base%p))%p;
  jj=(9240 * convert_ulong(exp%p))%p;

  while(ii != (p-1))
  {
    ii+=jj;
    ii = (ii>=p) ? (ii-p) : ii;
    k++;
  }
  next_multiple[tid]=k;
#if (CL_SIEVE_DEBUG>0)
  printf("Th %d: prime=%d, k=%d, (2(k_base+4620*k)*p+1)%%prime = %lld\n", tid, p, k, (((k_base+4620*k)%p)*2*exp+1)%p);
#endif
} // kernelanalyzer: 513M/s (5770)
                                     
__kernel void mfakto_cl_sieve(__global   uint *k_tab,         // out-array, the sieved factor-candidates
                              __constant uint *primes,        // primes used for sieving, indexed by tid
                              __global   uint *next_multiple, // in-array, for each prime, the next k when the fc is a multiple of prime[tid]
                              __private  uint k_tab_size,     // number of entries in the sieve
                              __global   uint *savestate      // to remember where to continue
                             )
{
  __private uint tid = get_global_id(0)+get_global_size(0)*get_global_id(1);
  __private uint k, m = next_multiple[tid];
  __private uint p = primes[tid];
//  savestate[0] : return k for next round (caller should add that to its k_min)
//  index   = savestate[1]
//  isprime = savestate[2]

  if (tid == 0) {savestate[1]=0; savestate[2]=1;}
  //BARRIER - write inits
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (k=0; savestate[1] < k_tab_size; k++)
  {
    if (k == m)
    {
      savestate[2]=0;  // kill this FC and
      m += p;          // set for the next match
#if (CL_SIEVE_DEBUG>0)
 //     printf("Th %d: prime=%d, k=%d killed\n", tid, p, k);
#endif
    }
    //BARRIER - read isprime
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (tid == 0)
    {
      if (savestate[2] > 0)
      {
        k_tab[savestate[1]] = k;  // not killed?: save it.
#if (CL_SIEVE_DEBUG>0)
        printf("Th %d: k=%d saved to idx %d\n", tid, k, savestate[1]);
#endif
        savestate[1]++;
      }
      else
      {
        savestate[2] = 1;  // init for next loop
      }
    }
    //BARRIER - write isprime/index
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  if (tid == 0)
  {
    savestate[0]=k;  // let the main program add to k_min how many ks we checked
  }
  next_multiple[tid] = m - k;  // adjust for next kernel run: k will start at 0 again.
}  // kernelanalyzer: 370M/s (5770)

