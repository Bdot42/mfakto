/**********************************************************************
   This file file is part of GPUSIEVE
   Copyright (C) 2012 by Rocke Verser.  All rights reserved.
  
   GPUSIEVE is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **********************************************************************/

/**********************************************************************
   A personal note from the orignal author:

   GPUSIEVE was originally written during January through March
   of 2012 by Rocke Verser.  To the best of my (Rocke Verser's)
   knowledge, there was no similar GPU software available to the
   public at the time of this release.  My original goals were to
   learn (always my goal) and to determine whether or not an
   efficient sieve could be implemented for GPU-style architectures.

   As such, I consider this release to be a *prototype*.  I assure
   you it contains defects, and I assure you it will require
   modifications to run on any platform different from my own.

   This software was written specifically with GIMPS (The Great
   Internet Mersenne Prime Search) in mind.  Sieving primes has
   been an interest of mine since I was a boy.  And finding
   factors of Mersenne numbers (which all have a very specific
   and special form) is a modern example of how sieving primes
   remains an important algorithm.  By the way, if Eratosthenes
   owned the patent on sieving, one would hope it has expired,
   by now.

   [The Sieve of Eratosthenes is a classic algorithm.  It was one
   of those algorithms that sparked my interest in mathematical
   computing, and which I actually first implemented as a boy to
   run on an IBM 1130 computer.  I did, in fact, succeed in
   printing a table of prime numbers up to 10 million on the
   IBM 1130 -- A computer with 8192 16-bit words of main memory.
   Using the same computer, I also managed to print the value of
   pi to 100,000 places after the decimal.  Some of the same tricks
   used to achieve those feats in the early 1970's still show
   through in modern mathematical algorithms, including what
   follows.  Very little is *really new*.  Things are just
   rewrapped and reinvented.]
 **********************************************************************/

#include <stdio.h>
#include <malloc.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "my_intrinsics.h"
#include "gpusieve.h"

sievecontext context[MAX_STREAMS];         // Space for one context per Stream Handle


// The only allocation of memory is at program start.  No other alocation
// or deallocation is required while this software runs.  We *are* so very
// sloppy!  If this prototype code is used where dynamic memory management
// is required, this needs to be cleaned up.
unsigned int *h_m1;
unsigned int *h_m2;
unsigned int *h_m3;
unsigned int *h_m4;
unsigned int *h_m5;
// igned int *h_m6;
unsigned int *h_m7;
unsigned int *h_m8;
unsigned int *h_m9;
unsigned int *h_m10;

// Following are created once, and remain constant for the life of the program
unsigned int *h_primep;         /* Host copy of primep array */
unsigned int *h_kncount;        /* Maximum number of hits within SIEVE_PRIMES for each prime */
unsigned int *h_ktree;          /* Tree of sum of kncount values */

// Device and host each have a copy of these constant structures
unsigned int *d_primep;         /* Device copy of primep array */
unsigned int *d_kncount;        /* Maximum number of hits within SIEVE_PRIMES for each prime */
unsigned int *d_ktree;          /* Tree of sum of kncount values */

// These structures are dynamic, with each stream owning its own structures
unsigned int *d_bdelta[MAX_STREAMS];    /* Offset to first multiple of a prime */
unsigned int *d_bitmapw[MAX_STREAMS];   /* Bitmap for sieving */
unsigned int *d_karray[MAX_STREAMS];    /* List of k-offsets to be trial-factored */
unsigned int *d_xaindexes[MAX_STREAMS]; /* atomic indices into karray */

// Following are copied back from device, for debugging purposes
unsigned int *h_bdelta[MAX_STREAMS];    /* Offset to first multiple of a prime */
unsigned int *h_bitmapw[MAX_STREAMS];   /* Bitmap for sieving */
unsigned int *h_karray[MAX_STREAMS];    /* List k-offsets to be trial-factored */
unsigned int *h_xaindexes[MAX_STREAMS]; /* atomic indices into karray */


__global__ void rcv_build_prime_tree(
        unsigned int *d_plist,  /* In: pointer to list of primes */
        unsigned int pcount,    /* number of elements in list */
        unsigned int kcount,    /* maximum number of bits in sieve table */
        unsigned int *d_kncount,/* Out: pointer to list of counts of k-values per prime */
        unsigned int *d_ktree   /* Out: pointer to tree of counts of k-values per prime */
        );

__global__ void rcv_init_class(
        unsigned int exp,       /* Mersenne exponent, p, of M(p) = 2^p-1 */
        unsigned int nclass,    // Number of classes  (must = 4620) */
        unsigned int kclass,    /* Number of this class (kstart mod nclasses) */
        int96        kstart,    /* Starting k-value.  Must=class (mod nclass) */
        unsigned int *d_plist,  /* In: Pointer to list of primes for sieving */
        unsigned int pcount,    /* In: Number of primes in sieve list */
        unsigned int *d_bdelta  /* Out: First bit, s.t. q is a multiple of the prime */
        );

__global__ void rcv_set_sieve_bits(
        unsigned int kcount,    /* number of bits in upcoming sieve */
        unsigned int *d_bitmapw /* bitmap for the sieve */
        );

__global__ void rcv_sieve_small_13_61(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta13, /* list of deltas, starting with 13 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        );

__global__ void rcv_sieve_small_67_127(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta67, /* list of deltas, starting with 67 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        );

__global__ void rcv_sieve_small_131_251(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta131, /* list of 23 deltas, starting with 131 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        );

__global__ void rcv_sieve_small_257_509(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta257,  /* list of 43 deltas for primes 257 through 509 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        );

__global__ void rcv_sieve_small_521_1021(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta521,  /* list of 75 deltas for primes 521 through 1021 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        );

__global__ void rcv_sieve_small_1031_2039(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta1031,  /* list of 137 deltas for primes 1031 through 2039 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        );

__global__ void rcv_sieve_primes(
        unsigned int tidoffseta,/* Offset from tid to first tid's element in tree */
        unsigned int tidoffsetz,/* Offset from tid to just past last tid's element in tree */
        unsigned int *d_plist,  /* pointer to list of primes */
        unsigned int pcount,    /* number of elements in list */
        int96        kstart,    /* lowest k-value in upcoming sieve */
        unsigned int kcount,    /* number of bits in upcoming sieve */
        unsigned int *d_bdelta, /* pointer to starting deltas per prime */
        unsigned int *d_kncount,/* pointer to count of k-values per prime */
        unsigned int *d_ktree,  /* pointer to tree of count of k-values per prime */
        unsigned int *d_bitmapw /* bitmap for the sieve */
        );

__global__ void rcv_reset_atomic_indexes(
        unsigned int width,          /* width of atomic index array */
        unsigned int *d_xaindexes    /* atomic index into array */
        );

__global__ void rcv_linearize_sieve(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bitmapw,/* bitmap for the sieve, 32-bit words */
        unsigned int *d_karray, /* linear array of k-values */
        unsigned int kasize,    /* number of spots in output array */
        unsigned int *d_kaindex /* atomic allocation index into karray */
        );


////////////////////////////////////////////////////////////////////////
//
// fillprimep -- This functions fills a linear array with small primes
//
////////////////////////////////////////////////////////////////////////

void fillprimep(unsigned int nump)
{
  unsigned int i;
  unsigned int j;
  unsigned int n;

  // Class mechanism implicitly deals with primes 2, 3, 5, 7, and 11
  // So, we skip the first five primes when filling our array.

  for (i=0, n=13; i<nump; n+=2)    /* Fill an array, starting at index i=0 */
  {
    j=0;
    if ((n%3 != 0) && (n%5 != 0) && (n%7 != 0) && (n%11 != 0))
        for (j=0; j<i; j+=1)       /* Check divisibility by previous primes */
    {
      if (n%h_primep[j] == 0)      /* If divisible, then n is not prime */
        break;
      if (h_primep[j]*h_primep[j] > n) /* We only have to test to sqrt(n) */
        j=i;                       /* Force exit from loop, with j>=i */
    }
    if (j>=i)                      /* If j<i, then not prime */
      h_primep[i++] = n;           /* If j>=i, then add new prime to array */
  }
  printf("h_primep[%u] = %u\n", i-1, h_primep[i-1]);  /* Print last prime */
}


////////////////////////////////////////////////////////////////////////
//
// InitApplication -- This function performs one-time initialization
//                    for this application.
//
////////////////////////////////////////////////////////////////////////

bool bPinGenericMemory;

void InitApplication()
{
  bool b;               // true if we are printing verbose debug info

  bPinGenericMemory = true;      // pin memory to allow async transfers

#ifdef DEBUGMALLOC
  b = true;
#else
  b = false;
#endif

  {
    int devID;
    devID = gpuDeviceInit(-1);
    printf("gpuDeviceInit returns deviceID:  %u\n",devID);
  }

  if (b) printf("Size of h_primep element:  %u\n", sizeof(*h_primep));
  if (b) printf("Size of malloc:  %u\n", sizeof(*h_primep)*MAX_SIEVE_PRIMES);

  AllocateHostMemory(bPinGenericMemory, &h_m1, &h_primep,  sizeof(*h_primep)*MAX_SIEVE_PRIMES);
  if (b) printf("h_primep      = %16.16lX\n", h_primep);

  AllocateHostMemory(bPinGenericMemory, &h_m3, &h_kncount, sizeof(*h_kncount)*MAX_SIEVE_PRIMES);
  if (b) printf("h_kncount     = %16.16lX\n", h_kncount);

  AllocateHostMemory(bPinGenericMemory, &h_m4, &h_ktree,   sizeof(*h_ktree)*(2*MAX_SIEVE_PRIMES));
  if (b) printf("h_ktree       = %16.16lX\n", h_ktree);

  for (int i=0; i<MAX_STREAMS; i+=1)
  {
    AllocateHostMemory(bPinGenericMemory, &h_m2, &h_bdelta[i], sizeof(*h_bdelta[i])*MAX_SIEVE_PRIMES);
    if (b) printf("h_bdelta[%d]   = %16.16lX\n", i, h_bdelta[i]);

    AllocateHostMemory(bPinGenericMemory, &h_m5, &h_bitmapw[i], sizeof(*h_bitmapw[i])*((SIEVE_BITS+31)/32));
    if (b) printf("h_bitmapw[%d]  = %16.16lX\n", i, h_bitmapw[i]);

    AllocateHostMemory(bPinGenericMemory, &h_m7, &h_karray[i],  sizeof(*h_karray[i] )*MAX_TF_IN_SIEVE);
    if (b) printf("h_karray[%d]   = %16.16lX\n", i, h_karray[i] );

    AllocateHostMemory(bPinGenericMemory, &h_m8, &h_xaindexes[i], sizeof(*h_xaindexes[i])*AX_COLUMNS);
    if (b) printf("h_xaindexes[%d]= %16.16lX\n", i, h_xaindexes[i] );
  }

  checkCudaErrors(cudaMalloc(&d_primep, sizeof(*d_primep)*MAX_SIEVE_PRIMES));
  if (b) printf("d_primep      = %16.16lX\n", d_primep);

  checkCudaErrors(cudaMalloc(&d_kncount, sizeof(*d_kncount)*MAX_SIEVE_PRIMES));
  if (b) printf("d_kncount     = %16.16lX\n", d_kncount);

  checkCudaErrors(cudaMalloc(&d_ktree, sizeof(*d_ktree)*(2*MAX_SIEVE_PRIMES)));
  if (b) printf("d_ktree       = %16.16lx\n", d_ktree);

  for (int i=0; i<MAX_STREAMS; i+=1)
  {
    checkCudaErrors(cudaMalloc(&d_bdelta[i], sizeof(*d_bdelta[i])*MAX_SIEVE_PRIMES));
    if (b) printf("d_bdelta[%d]   = %16.16lX\n", i, d_bdelta[i]);

    checkCudaErrors(cudaMalloc(&d_bitmapw[i], 4*((SIEVE_BITS+31)/32)));
    if (b) printf("d_bitmapw[%d]  = %16.16lx\n", i, d_bitmapw[i]);

    checkCudaErrors(cudaMalloc(&d_karray[i] , sizeof(*d_karray[i])*MAX_TF_IN_SIEVE));
    if (b) printf("d_karray[%d]   = %16.16lx\n", i, d_karray[i] );

    checkCudaErrors(cudaMalloc(&d_xaindexes[i] , sizeof(*d_xaindexes[i])*AX_COLUMNS));
    if (b) printf("d_xaindexes[%d]= %16.16lx\n", i, d_xaindexes[i] );
  }

  // Compute a table of primes for sieving
  {
    fillprimep(MAX_SIEVE_PRIMES);     // once per program run.  These primes do not change.

    checkCudaErrors(cudaMemcpyAsync(
            d_primep,
            h_primep,
            sizeof(*h_primep)*MAX_SIEVE_PRIMES,
            cudaMemcpyHostToDevice));  // copy table to the device.
  }

  // We let a single thread block generate the tree
  int threadsPerBlock = 256;

  // printf("Calling rcv_build_prime_tree<<<%u,%u>>>\n",1, threadsPerBlock);
  rcv_build_prime_tree<<<1, threadsPerBlock>>> (
        d_primep,  /* in: Pointer to list of primes */
        MAX_SIEVE_PRIMES, /* number of elements in list */
        SIEVE_BITS, /* maximum number of bits in upcoming sieves */
        d_kncount, /* out: pointer to list of counts of k-values per prime */
        d_ktree);  /* out: pointer to tree of counts of k-values per prime */

  // Get a copy of the list of counts
  checkCudaErrors(cudaMemcpyAsync(
          h_kncount,
          d_kncount,
          sizeof(*h_kncount)*MAX_SIEVE_PRIMES,
          cudaMemcpyDeviceToHost));

  // Get a copy of the tree of counts
  checkCudaErrors(cudaMemcpyAsync(
          h_ktree,
          d_ktree,
          sizeof(*h_kncount)*(2*MAX_SIEVE_PRIMES),
          cudaMemcpyDeviceToHost));

  // Ensure the list of counts and the tree have been fetched
  checkCudaErrors(cudaThreadSynchronize());  // Gotta make sure the tree has been fetched
                                             // CUDA drivers spin.  But it's only once.  Sigh.

#ifdef DEBUGTREE
  {
    bool bShowListSamples = true;
    bool bShowTreeSamples = true;

    if (bShowListSamples)  // Show first and last element of list?
    {
      printf("h_primep[0] = %u, h_kncount[0] = %u\n", h_primep[0], h_kncount[0]);
      printf("h_primep[%u] = %u, h_kncount[%u] = %u\n", MAX_SIEVE_PRIMES-1, h_primep[MAX_SIEVE_PRIMES-1], MAX_SIEVE_PRIMES-1, h_kncount[MAX_SIEVE_PRIMES-1]);
    }

    if (bShowTreeSamples)  // Show samples from the tree?
    {
      for (int i=0; i<40; i+=1)                       // Show some elements near the root of the tree
        printf("h_ktree[%u] = %u\n", i, h_ktree[i]);

      for (int i=MAX_SIEVE_PRIMES; i<MAX_SIEVE_PRIMES+64; i+=1)  // Show some of the leaves of the tree
        printf("h_ktree[%u] = %u\n", i, h_ktree[i]);

      for (int i=MAX_SIEVE_PRIMES-1; i>0; i=i>>1)     // Navigate up from the last two leaves of the tree
        printf("h_ktree[%u] = h_ktree[%u] + h_ktree[%u] : %u = %u + %u\n",
            i, 2*i, 2*i+1, h_ktree[i], h_ktree[2*i], h_ktree[2*i+1]);
    }

    // Perform a very basic check that tree was built correctly
    for (int i=1; i<MAX_SIEVE_PRIMES; i+=1)
      if (h_ktree[i] != h_ktree[2*i] + h_ktree[2*i+1])
        printf("ERROR IN TREE:  [%u] != [%u]+[%u] : %u != %u + %u\n", i, 2*i, 2*i+1,
                h_ktree[i], h_ktree[2*i], h_ktree[2*i+1]);
  }
#endif

}


////////////////////////////////////////////////////////////////////////
//
// InitMersenne -- This function performs initialization once for
//                 each new Mersenne exponent we receive.
//
//    exp -- The prime Mersenne exponent.  Such as 1000099 or 323000323
//    qlowpower
//    qhighpower -- These specify the range of candidates sought.
//                    2^qlowpoer < q < 2^qhighpower
//
////////////////////////////////////////////////////////////////////////

int96 mul96by32(int96 x96, unsigned int y32)
{
  int96 t96;
  unsigned long long t0,t1,t2;
  t2 = (0llu+x96.d2)*y32;
  t96.d2 = t2&0x00000000ffffffffllu;
  t1 = (0llu+x96.d1)*y32;
  t96.d2 += t1>>32;
  t96.d1 = t1&0x00000000ffffffffllu;
  t0 = (0llu+x96.d0)*y32;
  t96.d1 += t0>>32;
  t96.d0 = t0&0x00000000ffffffffllu;
  if (t96.d1 < t0>>32)	// If sum is smaller than what we added, then we should carry
    t96.d2 += 1;
  return(t96);
}

int96 div96by32(int96 x96, unsigned int y32)
{
  // Does not handle divide-by-zero.
  int96 t96;
  unsigned long long t0,t1,t2;
  t2 = (0llu+x96.d2);     // 64-bit version of high-order word of dividend
  t96.d2 = t2/y32;        // Set h/o word of quotient.  Overflow not possible.
  t1 = (t2 % y32) << 32;  // Remainder becomes h/o part of next partial divide.
  t1 += (0llu+x96.d1);    // Add in middle word of dividend
  t96.d1 = t1/y32;        // Set middle word of quotient.  Overflow not possible.
  t0 = (t1 % y32) << 32;  // Remainder becomes h/o part of next partial divide.
  t0 += (0llu+x96.d0);    // Add in low-order word of dividend.
  t96.d0 = t0/y32;        // Set l/o word of quotient.  Overflow not possible.
                          // Remainder is discarded.
  return(t96);
}

unsigned int mod96by32(int96 x96, unsigned int y32)
{
  // Does not handle divide-by-zero.
  unsigned int t;
  unsigned long long t0,t1,t2;
  t2 = (0llu+x96.d2);     // 64-bit version of high-order word of dividend
  t1 = (t2 % y32) << 32;  // Remainder becomes h/o part of next partial divide.
  t1 += (0llu+x96.d1);    // Add in middle word of dividend
  t0 = (t1 % y32) << 32;  // Remainder becomes h/o part of next partial divide.
  t0 += (0llu+x96.d0);    // Add in low-order word of dividend.
  t  = (t0 % y32);        // Remainder is returned
  return(t);
}

int96 add96by32(int96 x96, unsigned int y32)
{
  int96 t96;
  t96 = x96;
  t96.d0 += y32;
  if (t96.d0 < y32)
  {
    t96.d1 += 1;
    if (t96.d1 < 1)
      t96.d2 += 1;
  }
  return(t96);
}

int96 add96by96(int96 x96, int96 y96)
{
  int96 t96;
  unsigned long long t;
  t = (0llu + x96.d0) + y96.d0;
  t96.d0 = t & 0x00000000ffffffff;
  t >>= 32;
  t += (0llu + x96.d1) + y96.d1;
  t96.d1 = t & 0x00000000ffffffff;
  t >>= 32;
  t += (0llu + x96.d2) + y96.d2;
  t96.d2 = t & 0x00000000ffffffff;
  return(t96);
}

int96 sub96by96(int96 x96, int96 y96)
{
  int96 t96;
  t96 = y96;
  t96.d0 ^= 0xffffffff;
  t96.d1 ^= 0xffffffff;
  t96.d2 ^= 0xffffffff;
  t96 = add96by32(t96, 1);
  t96 = add96by96(x96, t96);
  return(t96);
}

int96 sub96by32(int96 x96, unsigned int y32)
{
  int96 t96;
  t96 = x96;
  if (t96.d0 >= y32) {t96.d0 -= y32; return(t96);} // no borrow from l/o word
  t96.d0 -= y32;

  if (t96.d1 >= 1) {t96.d1 -= 1; return(t96);}     // no borrow from middle word
  t96.d1 -= 1;

  t96.d2 -= 1;
  return(t96);
}

int cmp96(int96 x96, int96 y96)
{
  int96 t96;
  int t;
  t96 = sub96by96(x96, y96);
  t = t96.d2;
  if (t == 0)
    t = t96.d1;
  if (t == 0)
    t = t96.d0;
  return (t);
}

char *cvt96hex24(int96 x96, char *s, int slength)
{
  snprintf(s, slength, "%8.8X%8.8X%8.8X", x96.d2, x96.d1, x96.d0);
  return (s);
}

int96 mqstart;        // 2^minbit
int96 mkstart;        // Lowest k, s.t. 2*k*p+1 >= mqstart

int96 mqend;          // 2^maxbit
int96 mkend;          // Highest k, s.t. 2*k*p+1 <= mqend

int96 mbstart;        // Lowest b, s.t.  4620*b >= mkstart
int96 mbend;          // Highest b, s.t.  4620*b <= mkend
unsigned int mbstartrem;  // 4620*mbstart + mbstartrem === mkstart.  (-4620 < mbstartrem <= 0)
unsigned int mbendrem;    // 4620*mbend + mbendrem === mkend.  (0 <= mbendrem < 4620)


void InitMersenne(unsigned int exp, unsigned int qminbits, unsigned int qmaxbits)
{

  unsigned int f;       // Factor of our Mersenne number
  bool bprime;
  bool b;               // true if we are printing verbose debug info

#ifdef DEBUGBOUNDS
  b = true;
#else
  b = false;
#endif

  // Rudimentary range check.
  // Can't handle exponents that are inherently sieved by classes.
  // Also can't handle exponents that overflow to 32 bits.
  if (exp < 13 || exp > 0x7fffffff)
  {
    printf("InitMersenne: exp=%u is outside supported range\n", exp);
    exit(1);
  }

  bprime = true;        // Assume caller has provided a prime exponent

  // Check for small divisors, which aren't in our list of primes for sieving
  if (bprime) {f=2; if (exp%f == 0) bprime=false;}
  if (bprime) {f=3; if (exp%f == 0) bprime=false;}
  if (bprime) {f=5; if (exp%f == 0) bprime=false;}
  if (bprime) {f=7; if (exp%f == 0) bprime=false;}
  if (bprime) {f=11; if (exp%f == 0) bprime=false;}

  // Check prime divisors, which we happen to have handy
  for (int i=0; bprime && i<MAX_SIEVE_PRIMES; i+=1)
    {
      f = h_primep[i];
      if (f*f > exp)  // if clear of all divisors to sqrt(exp), then it's prime
        break;
      if (exp%f == 0) bprime=false;
    }

  // Check higher odd divisors
  for (unsigned int j=h_primep[MAX_SIEVE_PRIMES-1]+2; bprime; j+=2)
    {
      f = j;
      if (f*f > exp)              // if clear of all divisors to sqrt(exp), then it's prime
        break;
      if (exp%f == 0) bprime=false;
    }

  if (!bprime)
  {
    printf("InitMersenne: exp=%u is not prime.  It is divisible by %u\n", exp, f);
    exit(1);
  }

  if (qminbits >= qmaxbits)
  {
    printf("InitMersenne: qminbits=%u >= qmaxbits=%u\n",
            qminbits, qmaxbits);
    exit(1);
  }

  if (qmaxbits > 95)
  {
    printf("InitMersenne: qmaxbits=%u, exceeds limit of 95 bits\n", qmaxbits);
    exit(1);
  }

  if (qminbits < 24)
  {
    printf("InitMersenne: qminbits=%u is below limit of 24 bits\n", qminbits);
    exit(1);
  }

  if (b) printf("    exp=%8.8X%8.8X%8.8X = %20llu\n", 0, 0, exp, exp);

  mqstart.d2 = 0;
  mqstart.d1 = 0;
  mqstart.d0 = 0;

  if (qminbits >= 64 && qminbits <= 95)
    mqstart.d2 = 1<<(qminbits-64);

  if (qminbits >= 32 && qminbits <= 63)
    mqstart.d1 = 1<<(qminbits-32);

  if (                  qminbits <= 31)
    mqstart.d0 = 1<<(qminbits   );

  if (b) if (mqstart.d2 == 0)
    printf("mqstart=%8.8X%8.8X%8.8X = %20llu\n", mqstart.d2, mqstart.d1, mqstart.d0, ((0llu+mqstart.d1)<<32)+mqstart.d0);
  else
    printf("mqstart=%8.8X%8.8X%8.8X\n", mqstart.d2, mqstart.d1, mqstart.d0);

  mqend.d2 = 0;
  mqend.d1 = 0;
  mqend.d0 = 0;

  if (qmaxbits >= 64 && qmaxbits <= 95)
    mqend.d2 = 1<<(qmaxbits-64);

  if (qmaxbits >= 32 && qmaxbits <= 63)
    mqend.d1 = 1<<(qmaxbits-32);

  if (                  qmaxbits <= 31)
    mqend.d0 = 1<<(qmaxbits   );

  if (b) if (mqend.d2 == 0)
    printf("mqend  =%8.8X%8.8X%8.8X = %20llu\n", mqend.d2, mqend.d1, mqend.d0, ((0llu+mqend.d1)<<32)+mqend.d0);
  else
    printf("mqend  =%8.8X%8.8X%8.8X\n", mqend.d2, mqend.d1, mqend.d0);


  // Compute lowest possible k, s.t. mqstart <= q=2*k*p+1
  {
    unsigned long long t;
    mkstart = div96by32(mqstart, 2);
    mkstart = div96by32(mkstart, exp);
    mkstart = add96by32(mkstart, 1);  // Safe to always round up because divisor (exp) is prime
    t = 0;
    if (mkstart.d2 == 0)
      t = ((0llu+mkstart.d1) << 32) + mkstart.d0;
    if (b) printf("mkstart=%8.8X%8.8X%8.8X = %20llu\n", mkstart.d2, mkstart.d1, mkstart.d0, t);
  }

  // Compute highest possible k, s.t. q=2*k*p+1 <= mqend
  {
    unsigned long long t;
    mkend = div96by32(mqend, 2);
    mkend = div96by32(mkend, exp);
    t = 0;
    if (mkend.d2 == 0)
      t = ((0llu+mkend.d1) << 32) + mkend.d0;
    if (b) printf("mkend  =%8.8X%8.8X%8.8X = %20llu\n", mkend.d2, mkend.d1, mkend.d0, t);
  }


  // Compute lowest possible b, s.t. mkstart <= 4960*b
  {
    unsigned long long t2;
    mbstart = div96by32(mkstart, 4620);
    mbstartrem = mkstart.d0 - mbstart.d0*4620;     // Result guaranteed to be 0 <= t1 < 4620
    t2 = 0;
    if (mbstart.d2 == 0)
      t2 = ((0llu+mbstart.d1)<< 32) + mbstart.d0;
    if (b) printf("mbstart=%8.8X%8.8X%8.8X = %20llu | %4d\n", mbstart.d2, mbstart.d1, mbstart.d0, t2, mbstartrem);
  }

  // Compute highest possible b, s.t. 4960*b <= mkend
  {
    unsigned long long t2;
    mbend = div96by32(mkend, 4620);
    mbendrem = mkend.d0 - mbend.d0*4620;     // Result guaranteed to be 0 <= t1 < 4620
    t2 = 0;
    if (mbend.d2 == 0)
      t2 = ((0llu+mbend.d1)<< 32) + mbend.d0;
    if (b) printf("mbend  =%8.8X%8.8X%8.8X = %20llu | %4llu\n", mbend.d2, mbend.d1, mbend.d0, t2, mbendrem);
  }

  // Perform a quick test that our computed values are sensible
  {
    int96 tb1,                tb5;
    int96 tk1, tk2,      tk4, tk5;
    int96 tq1, tq2, tq3, tq4, tq5;

    char s1[25];
    char s2[25];
    char s3[25];

    // When scaled to corresponding values, we want the following relationships:
    //   bstart-1 <  kstart-1 <  qstart <= kstart   <= bstart
    //   bend     <= kend     <= qend   <  kend+1   <  bend+1

    // First, we handle the low end of the range, near minbits
    //   bstart-1 <  kstart-1 <  qstart <= kstart   <= bstart

    // In terms of b, please
    tb1 = mbstart;           // Compute 1 less than starting value
    tb1 = sub96by32(tb1, 1);
    tb5 = mbstart;           // Grab a copy of starting value

    // In terms of k, please
    tk1 = mul96by32(tb1, 4620);
    tk1 = add96by32(tk1, mbstartrem);
    tk2 = mkstart;           // Compute 1 less than starting value
    tk2 = sub96by32(tk2, 1);
    tk4 = mkstart;           // Grab a copy of starting value
    tk5 = mul96by32(tb5, 4620);
    tk5 = add96by32(tk5, mbstartrem);

    // In terms of q, please
    tq1 = mul96by32(tk1, exp);
    tq1 = mul96by32(tq1, 2);
    tq1 = add96by32(tq1, 1);
    tq2 = mul96by32(tk2, exp);
    tq2 = mul96by32(tq2, 2);
    tq2 = add96by32(tq2, 1);
    tq3 = mqstart;
    tq4 = mul96by32(tk4, exp);
    tq4 = mul96by32(tq4, 2);
    tq4 = add96by32(tq4, 1);
    tq5 = mul96by32(tk5, exp);
    tq5 = mul96by32(tq5, 2);
    tq5 = add96by32(tq5, 1);

    if (b) printf("     q1=%s  %s  %s  R %4u  = mbstart-1\n", cvt96hex24(tq1, s1, 25), cvt96hex24(tk1, s2, 25), cvt96hex24(tb1, s3, 25)+8, mbstartrem);
    if (b) printf("     q2=%s  %s  %24s  = mkstart-1\n", cvt96hex24(tq2, s1, 25), cvt96hex24(tk2, s2, 25), "");
    if (b) printf("     q3=%s  %24s  %24s  = mqstart\n",     cvt96hex24(tq3, s1, 25), "", "");
    if (b) printf("     q4=%s  %s  %24s  = mkstart\n", cvt96hex24(tq4, s1, 25), cvt96hex24(tk4, s2, 25), "");
    if (b) printf("     q5=%s  %s  %s  R %4u  = mbstart\n", cvt96hex24(tq5, s1, 25), cvt96hex24(tk5, s2, 25), cvt96hex24(tb5, s3, 25)+8, mbstartrem);

    if (cmp96(tq2,tq3) > 0 || cmp96(tq2,tq3) > 0)
    {
      printf("Computed mkstart is out of bounds.\n");
      exit(1);
    }

    if (cmp96(tq1,tq2) > 0 || cmp96(tq4,tq5) != 0)
    {
      printf("Computed mbstart is out of bounds.\n");
      exit(1);
    }


    // Next, we handle the high end of the range, near qmaxbits
    //   bend     <= kend     <= qend   <  kend+1   <  bend+1

    // In terms of b, please
    tb1 = mbend;             // Grab a copy of starting value
    tb5 = mbend;             // Compute 1 more than starting value
    tb5 = add96by32(tb5, 1);

    // In terms of k, please
    tk1 = mul96by32(tb1, 4620);
    tk1 = add96by32(tk1, mbendrem);
    tk2 = mkend;             // Grab a copy of starting value
    tk4 = mkend;             // Compute 1 more than starting value
    tk4 = add96by32(tk4, 1);
    tk5 = mul96by32(tb5, 4620);
    tk5 = add96by32(tk5, mbendrem);

    // In terms of q, please
    tq1 = mul96by32(tk1, exp);
    tq1 = mul96by32(tq1, 2);
    tq1 = add96by32(tq1, 1);
    tq2 = mul96by32(tk2, exp);
    tq2 = mul96by32(tq2, 2);
    tq2 = add96by32(tq2, 1);
    tq3 = mqend;  
    tq4 = mul96by32(tk4, exp);
    tq4 = mul96by32(tq4, 2);
    tq4 = add96by32(tq4, 1);
    tq5 = mul96by32(tk5, exp);
    tq5 = mul96by32(tq5, 2);
    tq5 = add96by32(tq5, 1);

    if (b) printf("\n");
    if (b) printf("     q1=%s  %s  %s  R %4u  = mbend\n", cvt96hex24(tq1, s1, 25), cvt96hex24(tk1, s2, 25), cvt96hex24(tb1, s3, 25)+8, mbendrem);
    if (b) printf("     q2=%s  %s  %24s  = mkend\n", cvt96hex24(tq2, s1, 25), cvt96hex24(tk2, s2, 25), "");
    if (b) printf("     q3=%s  %24s  %24s  = mqend\n",     cvt96hex24(tq3, s1, 25), "", "");
    if (b) printf("     q4=%s  %s  %24s  = mkend+1\n", cvt96hex24(tq4, s1, 25), cvt96hex24(tk4, s2, 25), "");
    if (b) printf("     q5=%s  %s  %s  R %4u  = mbend+1\n", cvt96hex24(tq5, s1, 25), cvt96hex24(tk5, s2, 25), cvt96hex24(tb5, s3, 25)+8, mbendrem);

    if (cmp96(tq2,tq3) > 0 || cmp96(tq2,tq3) > 0)
    {
      printf("Computed mkend is out of bounds.\n");
      exit(1);
    }

    if (cmp96(tq1,tq2) != 0 || cmp96(tq4,tq5) > 0)
    {
      printf("Computed mbend is out of bounds.\n");
      exit(1);
    }
  }
}

////////////////////////////////////////////////////////////////////////
//
// TestClass -- tests whether or not this class is composite
//
//    exp -- The prime Mersenne exponent.  Such as 1000099 or 56753239
//    class -- The proposed class (0 <= class < 4620)
//
////////////////////////////////////////////////////////////////////////

bool TestClass(unsigned int exp, unsigned int kclass)
{
  // q = 2*k*p+1
  unsigned int qclass;

  qclass = (((2 * kclass) * (exp % 9240)) + 1) % 9240;

  if ((qclass%8 != 1) && (qclass%8 != 7))
    return (false);     // qclass mod 8 not in {1,7}

  if (qclass%3 == 0)
    return (false);     // qclass mod 3 == 0

  if (qclass%5 == 0)
    return (false);     // qclass mod 5 == 0

  if (qclass%7 == 0)
    return (false);     // qclass mod 7 == 0

  if (qclass%11 == 0)
    return (false);     // qclass mod 11 == 0

  printf("TestClass: kclass=%4u --> qclass=%4u, qmod3=%u, qmod5=%u, qmod7=%u, qmod11=%2u, qmod8=%u\n",
            kclass, qclass, qclass%3, qclass%5, qclass%7, qclass%11, qclass%8);
  return (true);
}

////////////////////////////////////////////////////////////////////////
//
// InitClass -- This function performs initialization for a new class
//
//    exp -- The prime Mersenne exponent.  Such as 1000099 or 56753239
//    class -- The upcoming class (0 <= class < 4620)
//
////////////////////////////////////////////////////////////////////////

//    cqstart;        // 2^minbit
int96 ckstart;        // Lowest k, s.t. 2*k*p+1 >= qstart
int96 cbstart;        // Lowest b, s.t.  4620*b >= kstart

//    cqend;          // 2^maxbit
int96 ckend;          // Highest k, s.t. 2*k*p+1 <= qend
int96 cbend;          // Highest b, s.t.  4620*b <= kend

unsigned int csieve_bits;      // Total number of sieve bits to be processed for this class

void InitClass(unsigned int exp, unsigned int kclass)
{
  cbstart = mbstart;  // Our class bit-offset will start near the others
  if (kclass < mbstartrem)
    cbstart = add96by32(cbstart, 1);   // The low class numbers start one bit later

  ckstart = mul96by32(cbstart, 4620);
  ckstart = add96by32(ckstart, kclass);

  cbend = mbend;      // Our class bit-offset will end near the others
  if (kclass > mbendrem)
    cbend = sub96by32(cbend, 1);     // The high class numbers end one bit earlier

  int96 tq1, tq2, tq3, tq4, tq5;
  int96 tk1, tk2,      tk4, tk5;
  int96 tb1,                tb5;

  {
    tb1 = sub96by32(cbstart, 1);
    tb5 = cbstart;

    tk1 = mul96by32(tb1, 4620);
    tk1 = add96by32(tk1, kclass);
    tk4 = mkstart;
    tk5 = mul96by32(tb5, 4620);
    tk5 = add96by32(tk5, kclass);

    tq1 = mul96by32(tk1, exp);
    tq1 = mul96by32(tq1, 2);
    tq1 = add96by32(tq1, 1);
    tq3 = mqstart; 
    tq4 = mul96by32(tk4, exp);
    tq4 = mul96by32(tq4, 2);
    tq4 = add96by32(tq4, 1);
    tq5 = mul96by32(tk5, exp);
    tq5 = mul96by32(tq5, 2);
    tq5 = add96by32(tq5, 1);
#if 0
    {
      char s1[25], s2[25], s3[25];
      printf("\n");
      printf("     q1=%s  %s  %s  R %4u  = cbstart-1\n", cvt96hex24(tq1, s1, 25), cvt96hex24(tk1, s2, 25), cvt96hex24(tb1, s3, 25)+8, kclass);
      printf("     q3=%s  %24s  %24s  = mqstart\n",     cvt96hex24(tq3, s1, 25), "", "");
      printf("     q4=%s  %s  %24s  = mkstart\n", cvt96hex24(tq4, s1, 25), cvt96hex24(tk4, s2, 25), "");
      printf("     q5=%s  %s  %s  R %4u  = cbstart\n", cvt96hex24(tq5, s1, 25), cvt96hex24(tk5, s2, 25), cvt96hex24(tb5, s3, 25)+8, kclass);
    }
#endif

    if (cmp96(tq1,tq3) >= 0 || cmp96(tq4,tq5) > 0)
    {
      printf("Computed cbstart is out of bounds.\n");
      exit(1);
    }


    tb1 = cbend;
    tb5 = add96by32(cbend, 1);

    tk1 = mul96by32(tb1, 4620);
    tk1 = add96by32(tk1, kclass);
    tk2 = mkend;
    tk5 = mul96by32(tb5, 4620);
    tk5 = add96by32(tk5, kclass);

    tq1 = mul96by32(tk1, exp);
    tq1 = mul96by32(tq1, 2);
    tq1 = add96by32(tq1, 1);
    tq2 = mul96by32(tk2, exp);
    tq2 = mul96by32(tq2, 2);
    tq2 = add96by32(tq2, 1);
    tq3 = mqend; 
    tq5 = mul96by32(tk5, exp);
    tq5 = mul96by32(tq5, 2);
    tq5 = add96by32(tq5, 1);
#if 0
    {
      char s1[25], s2[25], s3[25];
      printf("\n");
      printf("     q1=%s  %s  %s  R %4u  = cbend\n", cvt96hex24(tq1, s1, 25), cvt96hex24(tk1, s2, 25), cvt96hex24(tb1, s3, 25)+8, kclass);
      printf("     q2=%s  %s  %24s  = mkend\n", cvt96hex24(tq2, s1, 25), cvt96hex24(tk2, s2, 25), "");
      printf("     q3=%s  %24s  %24s  = mqend\n",     cvt96hex24(tq3, s1, 25), "", "");
      printf("     q5=%s  %s  %s  R %4u  = cbend+1\n", cvt96hex24(tq5, s1, 25), cvt96hex24(tk5, s2, 25), cvt96hex24(tb5, s3, 25)+8, kclass);
    }
#endif

    if (cmp96(tq1,tq2) > 0 || cmp96(tq3,tq5) >= 0)
    {
      printf("Computed cbend is out of bounds.\n");
      exit(1);
    }
  }
}


////////////////////////////////////////////////////////////////////////
//
// PrepClassBatch -- This function prepares for a batch of sieving for
//                   the class last InitClass'ed
//
//    exp -- The prime Mersenne exponent.  Such as 1000099 or 56753239
//    class -- The upcoming class (0 <= class < 4620)
//
////////////////////////////////////////////////////////////////////////

void PrepClassBatch(unsigned int exp, unsigned int kclass, int streamix)
{
  // We need to cooperate with the "sieve_small" and "linearize"
  // kernels on the trailing blocks.  Those kernels expect a minimum
  // power-of-2 candidates per block.  We must always give them a
  // multiple of their candidates per block.

  const unsigned int LCPB = 32*1024;  // Largest Candidates per Block
                                      // Several of the rcv_sieve_small_xxx kernels
                                      // are configured for 32K candidates per block.

  int96 t96;
  t96 = sub96by96(cbend, cbstart);
  t96 = add96by32(t96, 1);

  if (t96.d2 != 0 || t96.d1 != 0 || (t96.d0/2) >= SIEVE_BITS)
    csieve_bits = SIEVE_BITS;    // With plenty of bits to go, process the maximum possible
  else                           // We even up the last 2 sieves of this class, hoping for better latency
    if (t96.d0 > SIEVE_BITS)     // Can we finish up on this sieve?
      csieve_bits = ((t96.d0+1)/2+LCPB-1) & ~(LCPB-1);// No.  Do half, rounded up to multiple of 2^k bits.
    else
      csieve_bits = (t96.d0+LCPB-1) & ~(LCPB-1);  // Get 'er done, rounding up to multiple of 2^k bits.
                                 // ***Caution*** We may exceed qmaxbits.  We may
                                 // test some candidates beyond caller-specified range.
                                 // Consider shifting the last block down so it ends at qmaxbits.  A
                                 // very little bit of retesting, rather than overrunning qmaxbits.

  int threadsPerBlock = 256;
  int blocksPerGrid;
  blocksPerGrid = (MAX_SIEVE_PRIMES + threadsPerBlock - 1) / threadsPerBlock;

  // printf("Calling rcv_init_class<<<%u,%u>>>\n", blocksPerGrid, threadsPerBlock);
  rcv_init_class <<<blocksPerGrid, threadsPerBlock, 0, context[streamix].stream>>> (
      exp,                      /* p of M(p)=2^p-1 */
      4620,                     /* Number of classes.  4620 = 3 * 5 * 7 * 11 * 4 */
      kclass,                   /* Class number of this run */
      ckstart,                  /* First 96-bit k-value of this run.  Must be congruent to class number. */
      d_primep,                 /* In: Pointer to list of prime numbers for sieving */
      MAX_SIEVE_PRIMES,         /* Number of primes in the sieve-list */
      d_bdelta[streamix]);      /* Out: per-prime.  Returns first bit, s.t. q is a multiple of the prime */

  // Since we are running multiple (possibly overlapped) streams, we remember
  // the important context information about what our stream is running.
  context[streamix].exp         = exp;
  context[streamix].kclass      = kclass;
  context[streamix].ckstart     = ckstart;
  context[streamix].cbstart     = cbstart;
  context[streamix].cbend       = cbend;
  context[streamix].csieve_bits = csieve_bits;
}


////////////////////////////////////////////////////////////////////////
//
// NextClassBatch -- This function prepares for an additional batch
//                   of sieving for the class last InitClass'ed
//
//    exp -- The prime Mersenne exponent.  Such as 1000099 or 56753239
//    class -- The upcoming class (0 <= class < 4620)
//
//    Returns true if more work is required to complete this class.
//    Returns false if the class is complete.
//
////////////////////////////////////////////////////////////////////////

bool NextClassBatch(unsigned int exp, unsigned int kclass)
{

  cbstart = add96by32(cbstart, csieve_bits);  // Next sieve should begin this many bits ahead
  if (cmp96(cbstart, cbend) > 0)    // Is our work done?
    return (false);                 // Yes, return to caller

  // Recompute ckstart in preparation for next batch
  ckstart = mul96by32(cbstart, 4620);
  ckstart = add96by32(ckstart, kclass);

  return (true);                    // Tell our caller to go around again
}


////////////////////////////////////////////////////////////////////////
//
// SieveCandidates -- This function sieves a full block of candidates
//
////////////////////////////////////////////////////////////////////////

void SieveCandidates(int streamix)
{
  int threadsPerBlock = 256;
  int blocksPerGrid;

  // Initialize the sieve bit-map.  One thread per 32-bit word in the bitmap.
  blocksPerGrid = ((csieve_bits+31)/32 + threadsPerBlock - 1) / threadsPerBlock;

  // printf("Calling rcv_set_sieve_bits<<<%u,%u,%u,%u>>>(%u)\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream, csieve_bits);
  rcv_set_sieve_bits<<<blocksPerGrid, threadsPerBlock, 0, context[streamix].stream>>> (
        csieve_bits,          /* number of bits in bit-map */
        d_bitmapw[streamix]); /* bitmap for the sieve, 32-bit words */

  // tidoffseta and tidoffsetz range across the total represented in the root of the tree.
  unsigned int tidoffseta;    // offset (from threadId 0) to the first thread we plan to launch
  unsigned int tidoffsetz;    // offset (from threadId 0) just past the last thread we plan to launch
                              // (tidoffsetz-tidoffseta is the number of threads we plan to launch)
  unsigned int i;

  // Start sieving the primes...
  for(tidoffseta = 0, i=0; tidoffseta < h_ktree[1]; tidoffseta = tidoffsetz, i+=1)
  {
    tidoffsetz = tidoffseta + h_kncount[i];  // Assume we're going to sieve one prime.


    if (h_primep[i] == 61*1)  // Very special handling when we hit this prime
    {
      // One thread per 32-bit word in the bitmap.
      blocksPerGrid = ((csieve_bits+31)/32 + threadsPerBlock - 1) / threadsPerBlock;
      // printf("Calling rcv_sieve_small_13_61<<<%u,%u,%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
      rcv_sieve_small_13_61<<<blocksPerGrid, threadsPerBlock, 0, context[streamix].stream>>> (
            ckstart,           /* lowest k-value in current sieve */
            csieve_bits,       /* number of bits in current sieve */
            &d_bdelta[streamix][i-12],   /* pass pointer to 13 consecutive deltas from p=13 to p=61 */
            d_bitmapw[streamix]);        /* bitmap for the sieve, 32-bit words */
    }

    if (h_primep[i] == 127*1)  // Very special handling when we hit this prime
    {
      // One thread per 64-bit word in the bitmap.
      blocksPerGrid = ((csieve_bits+63)/64 + threadsPerBlock - 1) / threadsPerBlock;
      // printf("Calling rcv_sieve_small_67_127<<<%u,%u,%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
      rcv_sieve_small_67_127<<<blocksPerGrid, threadsPerBlock, 0, context[streamix].stream>>> (
            ckstart,           /* lowest k-value in current sieve */
            csieve_bits,       /* number of bits in current sieve */
            &d_bdelta[streamix][i-12],   /* pass pointer to 13 consecutive deltas from p=67 to p=127 */
            d_bitmapw[streamix]);        /* bitmap for the sieve, 32-bit words */
    }

    if (h_primep[i] == 251*1)  // Very special handling when we hit this prime
    {
      // One thread per 128-bit word in the bitmap.
      blocksPerGrid = ((csieve_bits+127)/128 + threadsPerBlock - 1) / threadsPerBlock;
      // printf("Calling rcv_sieve_small_131_251<<<%u,%u,%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
      rcv_sieve_small_131_251<<<blocksPerGrid, threadsPerBlock, 0, context[streamix].stream>>> (
            ckstart,           /* lowest k-value in current sieve */
            csieve_bits,       /* number of bits in current sieve */
            &d_bdelta[streamix][i-22],   /* pass pointer to 23 consecutive deltas from p=131 to p=251 */
            d_bitmapw[streamix]);        /* bitmap for the sieve, 32-bit words */
    }

    if (h_primep[i] == 509*1)  // Very special handling when we hit this prime
    {
      // One thread per 256-bit word in the bitmap.
      // Here, we use half the normal threadsPerBlock, due to large shared memory usage
      blocksPerGrid = ((csieve_bits+255)/256 + (threadsPerBlock/2) - 1) / (threadsPerBlock/2);
      // printf("Calling rcv_sieve_small_257_509<<<%u,%u,%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
      rcv_sieve_small_257_509<<<blocksPerGrid, threadsPerBlock/2, 0, context[streamix].stream>>> (
            ckstart,           /* lowest k-value in current sieve */
            csieve_bits,       /* number of bits in current sieve */
            &d_bdelta[streamix][i-42],   /* pass pointer to 43 consecutive deltas from p=257 to p=509 */
            d_bitmapw[streamix]);        /* bitmap for the sieve, 32-bit words */
    }

    if (h_primep[i] == 1021*1)  // Very special handling when we hit this prime
    {
      // One thread per 512-bit word in the bitmap.
      // Here, we use half the normal threadsPerBlock, due to large shared memory usage
      blocksPerGrid = ((csieve_bits+255)/256 + (threadsPerBlock/2) - 1) / (threadsPerBlock/2);
      // printf("Calling rcv_sieve_small_521_1021<<<%u,%u,%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
      rcv_sieve_small_521_1021<<<blocksPerGrid, threadsPerBlock/2, 0, context[streamix].stream>>> (
            ckstart,           /* lowest k-value in current sieve */
            csieve_bits,       /* number of bits in current sieve */
            &d_bdelta[streamix][i-74],   /* pass pointer to 75 consecutive deltas from p=521 to p=1021 */
            d_bitmapw[streamix]);        /* bitmap for the sieve, 32-bit words */
      }

    if (h_primep[i] == 2039*1)  // Very special handling when we hit this prime
    {
      // One thread per 1024-bit word in the bitmap.
      // Here, we use 1/8 the normal threadsPerBlock, due to large shared memory usage
      blocksPerGrid = ((csieve_bits+255)/256 + (threadsPerBlock/8) - 1) / (threadsPerBlock/8);
      // printf("Calling rcv_sieve_small_1031_2039<<<%u,%u,%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
      rcv_sieve_small_1031_2039<<<blocksPerGrid, threadsPerBlock/8, 0, context[streamix].stream>>> (
            ckstart,           /* lowest k-value in current sieve */
            csieve_bits,       /* number of bits in current sieve */
            &d_bdelta[streamix][i-136],   /* pass pointer to 137 consecutive deltas from p=1031 to p=2039 */
            d_bitmapw[streamix]);         /* bitmap for the sieve, 32-bit words */
      }

    if (h_primep[i] > 2048)                          // Use the general-purpose siever?
    {
      // Once we're to the general-purpose siever, we have a choice whether to sieve
      // one prime at a time, in which case atomic accesses to the sieve bits is not
      // required.  Or we can sieve everything that's left.  Since the above special-
      // purpose kernels handled all of the very small primes, our best strategy is
      // usually to just sieve all that remains via a single kernel.
     
      if ((tidoffsetz-tidoffseta) < threadsPerBlock*6)  // Small primes, with many threads, sieve prime-by-prime
        tidoffsetz = h_ktree[1];  // Large primes, with few threads, run helter-skelter.

      blocksPerGrid = (tidoffsetz-tidoffseta + threadsPerBlock - 1) / threadsPerBlock;
      // printf("Calling rcv_sieve_primes<<<%u,%u,%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
      rcv_sieve_primes<<<blocksPerGrid, threadsPerBlock, 0, context[streamix].stream>>> (
          tidoffseta,         /* Offset from tid to first tid's element in tree */
          tidoffsetz,         /* Offset from tid to just past last tid's element in tree */
          d_primep,           /* pointer to list of primes */
          MAX_SIEVE_PRIMES,   /* number of elements in list */
          ckstart,            /* lowest k-value in this sieve */
          csieve_bits,        /* number of bits in this sieve */
          d_bdelta[streamix], /* pointer to starting deltas per prime */
          d_kncount,          /* pointer to count of k-values per prime */
          d_ktree,            /* pointer to tree of count of k-values per prime */
          d_bitmapw[streamix]);/* bitmap for the sieve, 32-bit words */
    }

  }
}

////////////////////////////////////////////////////////////////////////
//
// LinearizeCandidates -- This function converts the sieved bits to a linear list
//
////////////////////////////////////////////////////////////////////////

void LinearizeCandidates(int streamix)
{
  int width;
  int threadsPerBlock;
  int blocksPerGrid;


  // Reset atomic index into linear array
  width = AX_COLUMNS;
  threadsPerBlock=(width+31) & ~31;  /* round up to warp size of 32 */
  blocksPerGrid = (width + threadsPerBlock - 1) / threadsPerBlock;

  // printf("Calling rcv_reset_atomic_indexes<<<%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
  rcv_reset_atomic_indexes<<<blocksPerGrid, threadsPerBlock, 0, context[streamix].stream>>> (
        width,                   /* width of atomic index array */
        d_xaindexes[streamix]);  /* pointer to our atomic index array */
  

  // Convert bitmap to a list of candidates.  One thread per 32-bit word in the bitmap.
  threadsPerBlock = 256;
  blocksPerGrid = ((csieve_bits+31)/32 + threadsPerBlock - 1) / threadsPerBlock;

  // One thread per 32-bit word in the bitmap.
  // printf("Calling rcv_linearize_sieve<<<%u,%u>>>\n", blocksPerGrid, threadsPerBlock, 0, context[streamix].stream);
  rcv_linearize_sieve<<<blocksPerGrid, threadsPerBlock, 0, context[streamix].stream>>> (
        ckstart,                /* lowest k-value in current sieve */
        csieve_bits,            /* number of bits in current sieve */
        d_bitmapw[streamix],    /* bitmap for the sieve, 32-bit words */
        d_karray[streamix],     /* linear array of k-values */
        MAX_TF_IN_SIEVE,        /* number of spots in output array */
        d_xaindexes[streamix]); /* atomic index into karray */

#if 1
  // Copy final count of candidates back to local storage
  checkCudaErrors(cudaMemcpyAsync(
          h_xaindexes[streamix],
          d_xaindexes[streamix],
          sizeof(*h_xaindexes[streamix])*AX_COLUMNS,
          cudaMemcpyDeviceToHost,
          context[streamix].stream));
#endif
}

////////////////////////////////////////////////////////////////////////
//
// TrialFactorCandidates -- This function trial factors a list of candidates
//
////////////////////////////////////////////////////////////////////////

void TrialFactorCandidates(unsigned int exp, unsigned int kclass, int96 ckstart, unsigned int ncand, int streamix)
{
  unsigned long cbstart;

  cbstart = ((0llu + context[streamix].cbstart.d1)<<32) + context[streamix].cbstart.d0;

  // This is just a stub.  Call the actual trial factoring kernel, here!

}

////////////////////////////////////////////////////////////////////////
//
// Fetch bdelta array for debugging
//
////////////////////////////////////////////////////////////////////////

void FetchBdelta(int streamix)
{
  checkCudaErrors(cudaMemcpyAsync(
          h_bdelta[streamix],
          d_bdelta[streamix],
          sizeof(*h_bdelta[streamix])*MAX_SIEVE_PRIMES,  // Specify the number of bytes in the array
          cudaMemcpyDeviceToHost,
          context[streamix].stream));    // Begin fetching delta array using specified stream
}


////////////////////////////////////////////////////////////////////////
//
// Debug the bdelta array
//
////////////////////////////////////////////////////////////////////////

void DebugBdelta(int streamix)
{
  bool bSanityCheck = true;    // When set, we sanity-check the array

  // Caller is responsible for synchronization to be certain our buffer is ready to go

  if (bSanityCheck)
  {
    unsigned long cbstart;     // Where the current window into the infinite bitmap begins
    cbstart = ((0llu + context[streamix].cbstart.d1)<<32) + context[streamix].cbstart.d0;

    // For each prime in our sieve, confirm that the bdelta value is correct
    for (unsigned int i=0; i<MAX_SIEVE_PRIMES; i+=1)
    {
      unsigned int f;
      unsigned int bdelta;

      f = h_primep[i];                 // Get the current prime
      bdelta = h_bdelta[streamix][i];  // Get delta (from the start of the bitmap) for current prime

      // Sieving is meaningless when the small prime is equal to the Mersenne exponent
      if (context[streamix].exp == f)  // Is the Mersenne exponent equal to this small prime?
        if (bdelta < SIEVE_BITS)       // bdelta must be large, so we don't do any sieving with this small prime
          printf("BDelta[%7u]:  prime=%7u == Mersenne exponent;  bdelta=%7u < MAX_SIEVE_PRIMES=%7u\n", i, f, bdelta, SIEVE_BITS);

      // For the other 99.99% of the cases, our sieve prime is relatively prime to the Mersenne exponent
      if (context[streamix].exp != f)  // Normally, the Mersenne exponent is not a small prime
      {
        if (bdelta >= f)               // delta to first sievable bit must be in the range [0,f)
          printf("BDelta[%7u]:  prime=%7u <= bdelta=%7u.  (Delta too large)\n", i, f, bdelta);

        unsigned long kmodf;
        unsigned long qmodf;

        kmodf  = 4620llu * ((cbstart + bdelta) % f) + context[streamix].kclass;
        kmodf %= f;
        qmodf  = 2llu * kmodf * (context[streamix].exp % f) + 1;
        qmodf %= f;
        if (qmodf != 0)
          printf("dbd:  prime=%7u; bdelta=%7u.  (At bdelta, q mod prime = %7u != 0.)\n", f, bdelta, qmodf);

      }
    }
  }
}


////////////////////////////////////////////////////////////////////////
//
// Fetch bitmap for debugging
//
////////////////////////////////////////////////////////////////////////

void FetchBitmap(int streamix)
{
  checkCudaErrors(cudaMemcpyAsync(
          h_bitmapw[streamix],
          d_bitmapw[streamix],
          (context[streamix].csieve_bits+31)/32 * 4,  // Specify the number of sieve bytes used
          cudaMemcpyDeviceToHost,
          context[streamix].stream));    // Begin fetching bitmap using specified stream
}


////////////////////////////////////////////////////////////////////////
//
// Debug the bitmap
//
////////////////////////////////////////////////////////////////////////

void DebugBitmap(int streamix)
{
  bool bDebug = false;             // When set, we show some of the bitmap
  bool bCountbits = true;          // When set, we count how many survivors are present

  // Caller is responsible for synchronization to be certain our buffer is ready to go
  if (bDebug)
  {
    // Display first 64 bytes of sieve bitmap
    for (int i=0; i<64; i+=1)
    {
      if (i%16 == 0)
        printf("+%6.6X  ", i);
      printf("%2.2X ", ((unsigned char *)h_bitmapw[streamix])[i] );
      if (i%16 == 15)
        printf("\n");
    }
    printf("...\n");

    // Display last 64 bytes of sieve bitmap
    for (int i=4*((context[streamix].csieve_bits+31)/32)-64; i< 4*((context[streamix].csieve_bits+31)/32); i+=1)
    {
      if (i%16 == 0)
        printf("+%6.6X  ", i);
      printf("%2.2X ", ((unsigned char *)h_bitmapw[streamix])[i] );
      if (i%16 == 15)
        printf("\n");
    }
  }

  if (bCountbits)
  {
    // Count the number of bits set in the bitmap
    int t = 0;
    for (unsigned int i=0; i<context[streamix].csieve_bits; i+=1)    // Count all bits still set
      if (h_bitmapw[streamix][i>>5]&(1<<(i&31)))
      {
        t +=1;
      }
    printf("%9u bits found\n", t);
  }

  //////////////////////////////////////////////////////////////////////////
  // Here, we can inspect the bitmap in h_bitmapw.  Perhaps for part of a
  // validation suite.  Perhaps for a specific debugging issue.  As long
  // as we use atomic updates in the rcv_sieve_bitmap kernel, the contents
  // of this bitmap should be repeatable.  If we don't use atomic updates
  // in the rcv_sieve_bitmap kernel, we may see some extraneous 1-bits.
  // We should *never* see any extraneous 0-bits!
  //////////////////////////////////////////////////////////////////////////

}

////////////////////////////////////////////////////////////////////////
//
// Fetch candidates for debugging
//
////////////////////////////////////////////////////////////////////////

void FetchCandidates(int streamix)
{
  // Get atomic index, which is essentially the number of candidates in the linearized list.
  // printf("h_xaindexes[i]=%16.16lX, d_xaindexes[i]=%16.16lX, size=%u, stream=%u\n",
  //         h_xaindexes[streamix], d_xaindexes[streamix], sizeof(*h_xaindexes[streamix])*AX_COLUMNS, context[streamix].stream);
  checkCudaErrors(cudaMemcpyAsync(
          h_xaindexes[streamix],
          d_xaindexes[streamix],
          sizeof(*h_xaindexes[streamix])*AX_COLUMNS,
          cudaMemcpyDeviceToHost,
          context[streamix].stream));  // Begin fetching the atomic index, which contains #candidates

  // Note.  If we wanted to wait for the above to complete, we could fetch
  // just the used portion of the list, below.  We don't want to wait, so we
  // fetch the entire list.

  // Get the full linear list.  [We even fetch beyond the limits given by h_xaindexes[0].]
  // printf("h_karray[i]=%16.16lX, d_karray[i]=%16.16lX, size=%u, stream=%u\n",
  //         h_karray[streamix], d_karray[streamix], sizeof(*h_karray[streamix])*MAX_TF_IN_SIEVE, context[streamix].stream);
  checkCudaErrors(cudaMemcpyAsync(
          h_karray[streamix],
          d_karray[streamix],
          sizeof(*h_karray[streamix])*MAX_TF_IN_SIEVE,
          cudaMemcpyDeviceToHost,
          context[streamix].stream));  // Begin fetching full candidate list
}


////////////////////////////////////////////////////////////////////////
//
// Debug candidates
//
////////////////////////////////////////////////////////////////////////

void DebugCandidates(int streamix)
{
  bool bShowstats = true;          // When set, we report how many candidates were allocated, and from what pool
  bool bShowsample = false;        // When set, we show one sample candidate
  bool bCompareBitmap = true;      // When set, each candidate will be checked against the bitmap and vice-versa
  bool bMissedSieve = false;       // When set, is trial factored with all of our factors.  [Very slow.]

#ifdef DEBUGBITMAP
#else
  bCompareBitmap = false;          // We *cannot* compare candidates to the bitmap if we don't have it!
#endif

  // Caller is responsible for synchronization to be certain our buffers are ready to go

  if (bShowstats)
  {
    printf("%9u candidates found, %9u candidates tested, %6.4f survived sieve of %u primes >= 13.\n",
            h_xaindexes[streamix][0], context[streamix].csieve_bits, (h_xaindexes[streamix][0]*1.0)/(context[streamix].csieve_bits*1.0), MAX_SIEVE_PRIMES);
  }

  ///////////////////////////////////////////////////////////////////////////////////
  // Here, we inspect the linearized list of candidates, in h_karray.  Perhaps
  // for part of a validation suite.  Perhaps for a specific debugging issue.
  // Remember, however, the list is not guaranteed to be in order, since kernel
  // blocks run independently of each other and CUDA does not guarantee their
  // execution order.
  ///////////////////////////////////////////////////////////////////////////////////

  if (bCompareBitmap)             // Extensive test that Candidate List is a perfect match to bitmap?
  {
    bool bError = false;          // Once an error occurs, the next phase is aborted.
    unsigned long cbstart;
    cbstart = ((0llu + context[streamix].cbstart.d1)<<32) + context[streamix].cbstart.d0;

    //
    // Phase I.  Check that every surviving Candidate is in the bitmap.
    //
    for (unsigned int i=0; i<h_xaindexes[streamix][0]; i+=1)
    {
      unsigned int b;

      b = h_karray[streamix][i];  // Get the next candidate

      if (b > context[streamix].csieve_bits)  // Is our b offset within bounds of bitmap?
      {
        printf("Candidate %u:  b=%u, B=%llu exceeds bounds of bitmap\n", i, b, cbstart+b);
        bError=true;
        continue;                            // Avoid blowing the array bounds
      }

      if (h_bitmapw[streamix][b>>5]&(1<<(b&31)))  // Is the corresponding bit set in the bitmap?
        ;
      else
      {
        printf("Candidate %u:  b=%u, B=%llu, k=4620*B+%u, q=2*k*%u+1 is in linear list, but not in bitmap\n",
                             i, b, cbstart+b, context[streamix].kclass, context[streamix].exp);
        bError=true;
      }
    }

    //
    // Phase II.  Check for duplicate candidates.
    //            *** WARNING *** THIS IS A DESTRUCTIVE TEST OF THE BITMAP
    //            If you need the bitmap after this, you better save a copy.
    //
    if (!bError)
        for (unsigned int i=0; i<h_xaindexes[streamix][0]; i+=1)
    {
      unsigned int b;

      b = h_karray[streamix][i];  // Get the next candidate

      if (h_bitmapw[streamix][b>>5]&(1<<(b&31)))  // Is the corresponding bit set in the bitmap?
        h_bitmapw[streamix][b>>5] &= ~(1<<(b&31));   // turn off one bit
      else
      {
        printf("Candidate %u:  b=%u, B=%llu, k=4620*B+%u, q=2*k*%u+1 is duplicated in linear list.\n",
                             i, b, cbstart+b, context[streamix].kclass, context[streamix].exp);
        bError=true;
      }
    }

    //
    // Phase III.  Check for candidates in the bitmap that weren't in the linear list
    //
    if (!bError)
    {
      // Count the number of bits remaining in the bitmap
      int t = 0;
      for (unsigned int b=0; b<context[streamix].csieve_bits; b+=1)    // Count all bits still set
        if (h_bitmapw[streamix][b>>5]&(1<<(b&31)))
        {
          t +=1;

          printf("Candidate:  b=%u, B=%llu, k=4620*B+%u, q=2*k*%u+1 is in bitmap, but not in linear list.\n",
                              b, cbstart+b, context[streamix].kclass, context[streamix].exp);
        }
      if (t != 0)
        printf("%u candidates are present in the bitmap, but not in linear list.\n", t);
    }
  }

  if (bShowsample)        // Display first candidate from this batch?
  {
    unsigned long cbstart;
    cbstart = ((0llu + context[streamix].cbstart.d1)<<32) + context[streamix].cbstart.d0;

    unsigned int i;
    unsigned int b;
    i = 0;                // First candidate is as good a sample as any.
    b = h_karray[streamix][i];  // Get the next candidate

    printf("Candidate %u:  b=%u, B=%llu, k=4620*B+%u, q=2*k*%u+1\n",
                       i, b, cbstart+b, context[streamix].kclass, context[streamix].exp);
  }

  // Following is very slow, as MAX_SIEVE_PRIMES increases
  if (bMissedSieve)       // Trial factor each candidate with all of our small primes?
  {
    unsigned long cbstart;
    cbstart = ((0llu + context[streamix].cbstart.d1)<<32) + context[streamix].cbstart.d0;

    for (unsigned int i=0; i<h_xaindexes[streamix][0]; i+=1)
    {
      unsigned int b;
      b = h_karray[streamix][i];  // Get the next candidate

      // Check for prime divisors the old-fashioned way
      for (int j=0; j<MAX_SIEVE_PRIMES; j+=1)
      {
        unsigned int f;
        unsigned long long k;
        unsigned long long q;
        f  = h_primep[j];
        k  = 4620llu * ((cbstart + b) % f) + context[streamix].kclass;
        k %= f;
        q  = 2llu * k * context[streamix].exp + 1;
        q %= f;
        if (q == 0llu)
        {
          printf("Candidate %u:  b=%u, B=%llu, k=4620*B+%u, q=2*k*%u+1, is divisible by %6u\n",
                           i, b, cbstart+b, context[streamix].kclass, context[streamix].exp, f);
          break;
        }
      }
    }
  }

}


int main(void)
{

  int streamix;                     // Current stream to schedule
  int laggingix;                    // Current stream to be debriefed
  int nLinearized;                  // Number of streams for which linearized event is scheduled

  bool bDebugBdelta;
  bool bDebugBitmap;
  bool bDebugCandidates;

#ifdef DEBUGBDELTA
  bDebugBdelta = true;    // When set, permits examination of the delta array
#else
  bDebugBdelta = false;   // When not set, saves copying large array from device
#endif

#ifdef DEBUGBITMAP
  bDebugBitmap = true;    // When set, permits examination of the sieve bitmap
#else
  bDebugBitmap = false;   // When not set, saves copying large bitmap from device
#endif

#ifdef DEBUGCANDIDATES
  bDebugCandidates = true;  // When set, permits examination of the candidate list
#else
  bDebugCandidates = false; // When not set, saves copying large candidate list from device
#endif

  InitApplication();      // Go perform application initialization

  // Initialize a set of stream-handles and events to keep things flowing
  for (int i=0; i<MAX_STREAMS; i+=1)
  {
    checkCudaErrors( cudaStreamCreate(&context[i].stream) );
    checkCudaErrors( cudaEventCreateWithFlags(&context[i].linearized_event,  cudaEventBlockingSync) );
  }
  streamix = 0;                 // First stream to be utilized
  nLinearized = 0;              // No streams have reached LinearizedCandidates, yet


  ////////////////////////////////////////////////////////////////////
  // This data should come from a "worktodo" file.  Hard-coded now,
  // just for prototyping purposes.
  ////////////////////////////////////////////////////////////////////
  
  InitMersenne(MERSENNE_EXPONENT, QBITMIN, QBITMAX);  // Go perform initialization for a new Mersenne number

  for (int kclass=MINKCLASS; kclass<=MAXKCLASS; kclass+=1)
  {
    bool bGoodClass;

    bGoodClass = TestClass(MERSENNE_EXPONENT, kclass); // If class contains no primes, returns false
    if (bGoodClass)
    {
      InitClass(MERSENNE_EXPONENT, kclass);  // Go perform initialization for a new class
      bool bMoreToSieve;
      bMoreToSieve = true;

      // Loop as long as we have more work to enqueue or we have work left to dequeue
      while (bMoreToSieve || nLinearized != 0)
      {
        if (bMoreToSieve)
        {
          PrepClassBatch(MERSENNE_EXPONENT, kclass, streamix);  // Setup for the next sieve block
          if (bDebugBdelta)                  // Do we want to fetch and examine the delta array?
            FetchBdelta(streamix);             // For debugging, go initiate fetching of the array
          SieveCandidates(streamix);         // Sieve the candidates in current sieve block
          if (bDebugBitmap)                  // Do we want to fetch and examine the bitmap?
            FetchBitmap(streamix);             // For debugging, go initiate fetching of the bitmap
          LinearizeCandidates(streamix);     // Extract candidates from bitmap to a list
          if (bDebugCandidates)              // Are we configured to fetch and examine the candidate list?
            FetchCandidates(streamix);         // For debugging, go initiate fetching of the candidates
          checkCudaErrors( cudaEventRecord(context[streamix].linearized_event, context[streamix].stream) );
                                             // record event in this stream
          if (nLinearized == 0)              // Have we scheduled our first EventRecord?
            laggingix = streamix;            // Yes.  Remember oldest stream scheduled
          nLinearized += 1;                  // Bump the number of linearized_event kernels running
        }

        // Wait 'til all streams are performing sieving to begin factoring.
        // But if this class is out of sieving, empty the pipeline.
        if (nLinearized == MAX_STREAMS ||
            (!bMoreToSieve && nLinearized != 0))    // Wait 'til all streams are performing sieving to begin factoring
        {
          checkCudaErrors( cudaEventSynchronize(context[laggingix].linearized_event) );
                                           // Wait for oldest kernel's stream to finish linearizing a block
          nLinearized -= 1;                // One stream has caught up

          if (bDebugBdelta)                // Are we configured to fetch and examine the delta array?
            DebugBdelta(laggingix);          // For debugging, go analyze the delta array
          if (bDebugBitmap)                // Are we configured to fetch and examine the bitmap?
            DebugBitmap(laggingix);          // For debugging, go analyze the bitmap
          if (bDebugCandidates)            // Are we configured to fetch and examine the candidate list?
            DebugCandidates(laggingix);      // For debugging, go analyze the candidate list

          TrialFactorCandidates(
                      context[laggingix].exp,
                      context[laggingix].kclass,
                      context[laggingix].ckstart,
                      h_xaindexes[laggingix][0],
                      laggingix);   // Go trial factor this batch of candidates

          laggingix = (laggingix+1)%MAX_STREAMS; // Next stream to be dequeued
        }

        streamix = (streamix+1)%MAX_STREAMS;     // Next batch of sieves switches to the next buffer
        if (bMoreToSieve)
          bMoreToSieve = NextClassBatch(MERSENNE_EXPONENT, kclass);
                                   // If more work at this class, get it setup
      }
    }
  }

  ////////////////////////////////////////////////////////////////////
  // Here, we should loop back for more work from a "worktodo"
  // file.  For prototyping, we just run a single Factor=
  ////////////////////////////////////////////////////////////////////
  
  // Clean up our stream-handles and events
  for (int i=0; i<MAX_STREAMS; i+=1)
  {
    checkCudaErrors( cudaStreamSynchronize(context[i].stream) );
    checkCudaErrors( cudaEventDestroy(context[i].linearized_event) );
    checkCudaErrors( cudaStreamDestroy(context[i].stream) );
  }

  cudaThreadSynchronize();


  FreeHostMemory(bPinGenericMemory, &h_m1, &h_primep,  sizeof(*h_primep)*MAX_SIEVE_PRIMES);
  h_primep = NULL;

  FreeHostMemory(bPinGenericMemory, &h_m3, &h_kncount, sizeof(*h_kncount)*MAX_SIEVE_PRIMES);
  h_kncount = NULL;

  FreeHostMemory(bPinGenericMemory, &h_m4, &h_ktree,   sizeof(*h_ktree)*(2*MAX_SIEVE_PRIMES));
  h_ktree = NULL;

  for (int i=0; i<MAX_STREAMS; i+=1)
  {
    FreeHostMemory(bPinGenericMemory, &h_m5, &h_bitmapw[i], sizeof(*h_bitmapw[i])*((SIEVE_BITS+31)/32));
    h_bitmapw[i] = NULL;

    FreeHostMemory(bPinGenericMemory, &h_m7, &h_karray[i],  sizeof(*h_karray[i] )*MAX_TF_IN_SIEVE);
    h_karray[i]  = NULL;

    FreeHostMemory(bPinGenericMemory, &h_m8, &h_xaindexes[i], sizeof(*h_xaindexes[i])*AX_COLUMNS);
    h_xaindexes[i]  = NULL;
  }

  cudaFree(d_primep);
  d_primep = NULL;

  cudaFree(d_kncount);
  d_kncount = NULL;

  cudaFree(d_ktree);
  d_ktree = NULL;

  for (int i=0; i<MAX_STREAMS; i+=1)
  {
    cudaFree(d_bdelta[i]);
    d_bdelta[i] = NULL;

    cudaFree(d_bitmapw[i]);
    d_bitmapw[i] = NULL;

    cudaFree(d_karray[i] );
    d_karray[i]  = NULL;

    cudaFree(d_xaindexes[i] );
    d_xaindexes[i]  = NULL;
  }

  exit (0);
}

