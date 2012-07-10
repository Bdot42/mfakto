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

// #define MAX_SIEVE_PRIMES      304 // Can't go smaller without code changes
// #define MAX_SIEVE_PRIMES      559
   #define MAX_SIEVE_PRIMES     1500 // Good starting value for tuning
// #define MAX_SIEVE_PRIMES   250000
// #define MAX_SIEVE_PRIMES 10000000 // Tested to this level, but slow

#define SIEVE_BITS (3*1024*1024)   // Size of the bitmap used for sieving
                                   // Must be a multiple of 32K

// Mersenne number should be an input.  But this is prototype code.
// #define MERSENNE_EXPONENT       2053  // Can't go smaller without code changes
// #define MERSENNE_EXPONENT    1000099
// #define MERSENNE_EXPONENT   56753239
// #define MERSENNE_EXPONENT   56755087
// #define MERSENNE_EXPONENT   56759389
   #define MERSENNE_EXPONENT   53785969
// #define MERSENNE_EXPONENT   15485959
// #define MERSENNE_EXPONENT  999999937
// #define MERSENNE_EXPONENT 0x7FFFFFFF  // Tested to this level

// Bit limits should be an input.  But this is prototype code.
#define QBITMIN 71            // Sieve for factors greater than 2^71
#define QBITMAX 72            // Sieve for factors less than 2^72

// #define or #undef the following to enable or disable running a subset of the classes
#define DEBUGSUBSET

#ifdef DEBUGSUBSET
#define MINKCLASS 4611        // Fewer classes permit faster debug runs
#define MAXKCLASS 4619        // Fewer classes permit faster debug runs
#else
#define MINKCLASS 0           // For production, must test all classes
#define MAXKCLASS (4620-1)    // For production, must test all classses
#endif


// Note:  At larger values of MAX_SIEVE_PRIMES, observed average number
// of candidates passing is slightly higher than theoretical shown.
// [This is when using atomic operations, with no missed sieve bits.]
// Not sure why.  When doing "normal" sieving, there's no point sieving
// for 11's at any level below 121.  I conjecture a similar effect may be
// happening here, but I haven't analyzed how it fits sieving for prime
// q=2*k*p+1.  (When sieving a 26-bit Mersenne number, for factors at
// the 72-bit level, at MAX_SIEVE_PRIMES=10000000, we tend to see about
// 14.3+% of the candidates passing.)

// Using non-atomic operations also results in values larger than shown,
// but for the entirely different reason that some sieved bits are
// simply lost.  If code is modified to use non-atomic operations,
// consideration must be given to how that will change these passing
// percentages, and to whether those changes will impact any decisions
// based on the "law of large numbers".

// at MAX_SIEVE_PRIMES =      304, 35.35% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES =      559, 32.42% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES =     1000, 30.04% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES =     1500, 28.59% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES =     2500, 26.96% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES =    25000, 21.50% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES =   250000, 17.93% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES =  1000000, 16.32% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES =  9000000, 14.30% of SIEVE_BITS should pass
// at MAX_SIEVE_PRIMES = 10000000, 14.22% of SIEVE_BITS should pass (works, but slow)

   #define MAX_TF_IN_SIEVE (SIEVE_BITS*37/100)    // Should be fine, based on the law of large numbers.
// #define MAX_TF_IN_SIEVE (SIEVE_BITS*31/100)    // Should be fine, at MAX_SIEVE_PRIMES >= 1500.
// #define MAX_TF_IN_SIEVE (SIEVE_BITS*100/100)   // Guaranteed safe.  Handles complete failure of sieving.
                                                  // If you decrease SIEVE_BITS, recalculate safety margin.

// #define MAX_STREAMS 1     // Works fine.  Least memory footprint.
   #define MAX_STREAMS 2     // Classic "double buffering".  Generally works well.
// #define MAX_STREAMS 3     // Good starting point for wringing out bugs.
// #define MAX_STREAMS 6
// #define MAX_STREAMS 10    // Upper limit is when you run out of memory
                             // Be cautious of video performance at high values

// #define or #undef the following to enable or disable major classes of debugging
#undef  DEBUGMALLOC        // Define this to verbosely show memory allocations
#undef  DEBUGTREE          // Define this to examine the tree
#undef  DEBUGBOUNDS        // Define this to verbosely examine the bounds on the sieve
#undef  DEBUGBDELTA        // Define this to fetch and examine the delta array
#undef  DEBUGBITMAP        // Define this to fetch and examine the bitmap
#undef  DEBUGCANDIDATES    // Define this to fetch and examine the list of candidates


#define AX_COLUMNS 1       // Number of entries in atomic index array
                           // Code only uses 1 atomic index at the moment.

typedef struct
{
  unsigned int d0, d1, d2;
} int96;

#include "cudahelper.h"         // Define CUDA "helper" functions

typedef struct                  // Context information for each stream's work
{
  cudaStream_t stream;          // Space for cuda Stream Handles
  cudaEvent_t linearized_event; // Event placed after linearization function
  unsigned int   exp;           // Mersenne exponent
  unsigned int   kclass;        // Class number
  unsigned int   csieve_bits;   // Number of bits in current sieve
           int96 ckstart;       // Lowest k-value for current sieve
           int96 cbstart;       // Lowest b-value for current sieve
           int96 cbend;         // Highest b-value for current sieve
} sievecontext;

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



///////////////////////////////////////////////////////////////////
//
// rcv_init_class
//
//   Background:
//
//     Let  M(p) = 2^p-1  be a Mersenne number.
//     By theorem, all prime factors of M(P) are of the form
//          q = 2*k*p+1,
//     where k is an integer.
//
//     Since we are interested in finding the smallest
//     factors of Mersenne numbers, we are interested
//     in finding prime factors of Mersenne numbers.  No
//     such prime factor, q, will contain any small prime
//     factors.  Furthermore, by theorem, any prime
//     factor of a Mersenne number will be congruent to
//     +/-1 (mod 8).
//
//     What is a "class"?  A class is a set of numbers that have a
//     common remainder, modulo the product of a set of small primes.
//     In our implementation, there are 4620 = 2^2 * 3 * 5 * 7 * 11
//     possible classes.
//
//     All of the candidates we generate in a batch are of the same
//     class, so all those candidates we generate will have the same
//     remainder, mod 4620.
//
//     The reason is simply for performance.  If we never even
//     attempt to generate candidates that are a multiple of 3,
//     then we save 1/3 of the work we might otherwise perform.
//     If we never generate candidates that are a multiple of 5,
//     then we save another 1/5 of the work we might otherwise
//     have performed.  Ditto for multiples of 7 and 11.
//
//     Furthermore, all candidates must be equal to 1 or 7,
//     modulo 8 to be a factor of a Mersenne number.  This lets
//     us reduce by another factor of 2 the work we might
//     otherwise have performed.  [It might appear we should be
//     reducing the work by a factor of 4, but we are inherently
//     only considering odd candidates, so we can only
//     eliminate candidates that are equal to 3 or 5, modulo 8.]
//
//     By theorem, all factors of a mersenne number are of the
//     form, 2*k*p+1, where k is a positive integer and p is the
//     exponent of the Mersenne number.
//
//     The following table contains (2*k*p) mod 8 for each
//     possible combination of a k-value with an odd Mersenne exponent.
//
//                   (2*k) mod 8
//       p mod 8   0   -   2   -   4   -   6   -
//                -------------------------------
//          1   |  0   -   2   -   4   -   6   -
//          3   |  0   -   6   -   4   -   2   -
//          5   |  0   -   2   -   4   -   6   -
//          7   |  0   -   6   -   4   -   2   -
//
//     Next, we look at the candidates (2*k*p+1) mod 8.  The valid
//     candidates are equal to 1 or 7.  Others need not be generated.
//
//                   (2*k) mod 8
//       p mod 8   0   1   2   3   4   5   6   7
//                -------------------------------
//          1   |  1   -   -   -   -   -   7   -
//          3   |  1   -   7   -   -   -   -   -
//          5   |  1   -   -   -   -   -   7   -
//          7   |  1   -   7   -   -   -   -   -
//
//     This shows us that if (p mod 4) == 1, then k must equal 0 or 3 (modulo 4).
//                    and if (p mod 4) == 3, then k must equal 0 or 1 (modulo 4).
//
//     For a given Mersenne exponent, we only generate Candidates from among
//     960 of the possible 4620 classes:
//
//          2 * 4 * 6 * 10 * 2    960
//          ------------------ = ----
//          3 * 5 * 7 * 11 * 4   4620
//
//     Within each class, we sieve out Candidates that are multiples of higher
//     primes, 13, 17, 19, 23, etc.
//
//
//     By our own internal convention, we number the classes based on k (mod 4620),
//     as used in the formula q = 2*k*p+1.  And all candidates, q, generated with a batch
//     of a given class will be congruent to each other modulo 9240.  Note that
//     we will consider only 960 of the 9240 possible candidate residues.  Also
//     note that the candidate residues will normally not be our class number.
//
//
//
//
//   Input parameters:
//
//     exp:  The exponent of the Mersenne Number being tested.
//           E.g., to trial-factor 2^100099-1, exp=100099
//
//     class:  The "class" of factors about to be tested.  There
//             are 4620 = 2^2 * 3 * 5 * 7 * 11 possible classes.
//             All factors tested within a given class will have
//             the same remainder, mod 4620.  Among the possible
//             classes, 960 classes may contain prime factors of
//             the Mersenne number under test.
//
//             Among the 4620 classes, only 960 classes are not
//             divisible by one of the 4 small odd primes and
//             congruent to +/- 1 (modulo 8).
//
//              960 = 2 * 2 * 4 * 6 * 10
//             4620 = 8 * 3 * 5 * 7 * 11 / 2.
//             (Even numbers are inherently excluded by q=2kp+1,
//             and are not a member of any class.)
//
//     kdiv4620:  This is the starting value of k, divided by
//             4620, and truncated to an integer.  The starting
//             value for k is therefore equal to 4620*k + kdelta,
//             where 0 <= kdelta < 4620, such that k mod 4620 = class.
//
//       kdelta:  The offset (bits) from the first bit of the sieve
//                to the first bit to be sieved out.  [Thereafter,
//                every prime-th bit is sieved out.]  (For each prime,
//                0 <= kdelta_prime < prime.)  (For very large primes,
//                in which no sieving will occur in this set of bits,
//                this may be beyond the bits available for sieving.)
//
// q  = 2*k*p + 1
// q0 = 2*k0*p + 1
// k0 = class + 4620*c0       k0 = kstart.  Corresponds to first bit of bitmap.
// kn = class + 4620*(c0+i)   0 <= i < SIEVE_BITS.  Corresponds to successive bits of bitmap.
// qn = 2*kn*p + 1
// qn = 2*(class + 4620*(c0+i))*p + 1
// qn = 2*class*p + 9240*c0*p + 9240*i*p + 1
// qn = q0 + 9240*i*p
//                            delta is chosen as the smallest i such that qn is divisible by the
//                              corresponding sieve-prime and qn >= q0.
//                            ndelta is chosen as the smallest i such that qn is divisible by the
//
///////////////////////////////////////////////////////////////////

__device__ unsigned int mod32bit(unsigned int n, unsigned int d)
{
  return n%d;
}

__device__ unsigned int modularinverse(unsigned int n, unsigned int d)
{
  int x, y, lastx, lasty, q, t;
  x=0; y=1; lastx=1; lasty=0;
  while (d != 0)
  {
    q=n/d;                      // Floor(n/d)
    t=d; d=n%d; n=t;            // d=n mod d; n=lastd;
    t=x; x=lastx-q*x; lastx=t;
    t=y; y=lasty-q*y; lasty=t;
  }
  return(lastx);
}

__global__ void __launch_bounds__(256, 4) rcv_init_class(
        unsigned int exp,       // Mersenne exponent, p, of M(p) = 2^p-1 */
        unsigned int nclass,    // Number of classes  (must = 4620) */
        unsigned int kclass,    // Number of this class */
        int96        kstart,    // Starting k-value.  Must=class (mod nclass) */
        unsigned int *d_plist,  // In: Pointer to list of primes for sieving */
        unsigned int pcount,    // In: Number of primes in sieve list */
        unsigned int *d_bdelta  // Out: First bit, s.t. q is a multiple of the prime */
        )
{

  unsigned int i;
  i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < pcount)               // Excess threads do not participate
  {
    // Compute lowest possible k_d, s.t. q=2(k+4620*k_d)p+1 is divisible by our prime
    unsigned int q0mp;
    if (d_plist[i] > 46341)  // Do we have to use long arithmetic?
    {
      unsigned long long ksmp;  // kstart mod (current small prime)
      unsigned long long qsmp;  // qstart mod (current small prime)
      ksmp = (             kstart.d2) % d_plist[i];
      ksmp = ((ksmp<<32) + kstart.d1) % d_plist[i];
      ksmp = ((ksmp<<32) + kstart.d0) % d_plist[i];

      qsmp = (2*ksmp*(exp % d_plist[i]) + 1) % d_plist[i];
      q0mp = qsmp;
    }
    else
    {
      unsigned int ksmp;  // kstart mod (current small prime)
      ksmp = (              kstart.d2)             % d_plist[i];
      ksmp = ((ksmp<<16) + (kstart.d1>>16       )) % d_plist[i];
      ksmp = ((ksmp<<16) + (kstart.d1&0x0000ffff)) % d_plist[i];
      ksmp = ((ksmp<<16) + (kstart.d0>>16       )) % d_plist[i];
      ksmp = ((ksmp<<16) + (kstart.d0&0x0000ffff)) % d_plist[i];

      q0mp = (2*ksmp*(exp % d_plist[i]) + 1) % d_plist[i];  // 2*46340*46340+1 < 2^32
    }

    unsigned int p9240mp;    // (2*4620*exponent) mod (current small prime)
    p9240mp = (9240llu * (exp % d_plist[i])) % d_plist[i];

    if (p9240mp == 0)        // Did our thread's prime equal the Mersenne exponent?
      d_bdelta[i] = SIEVE_BITS;    // Yes, don't attempt to sieve this prime
    else
    {
      unsigned int j;
               int p9240mpinv;
      p9240mpinv = modularinverse(p9240mp, d_plist[i]);
      if (p9240mpinv < 0)
        p9240mpinv += d_plist[i];

      // primes can exceed 65535, so use 64-bit multiply
      j = (0llu+d_plist[i]-q0mp)*(0llu+p9240mpinv) % d_plist[i];
      d_bdelta[i] = j;
    }
  }
}


///////////////////////////////////////////////////////////////////
//
// rcv_build_prime_tree
//
//   This function receives the set of primes for which sieving
//   will be performed.
//
//   Input parameters:
//
//       kcount:  Number of bits in sieve table.
//
//       plist:   Pointer to a list of primes to sieve the k-values.
//
//       pcount:  Number of primes in the sieve-list.
//
//       kncount: The maximum number of sievable bits, per prime.
//
//                A---V  Above and below are the same.  One is a list.  One is a tree.
//
//       ktree:   ktree is an array of 2*pcount+1 elements.
//                Element [0] is unused.
//                Element [n] is the sum of [2n] and [2n+1].
//                Elements [pcount+1 to 2*pcount] are the
//                same as the kncount list.
//
//   Example:
//   kcount = 1048576
//   kstart = First k-value in upcoming sieve
//   plist->     13,   17,   19,   23,   29,   31,   ...,   3497743
//   kdelta->     7,   15,   16,    1,    0,   15,   ...,   1589450
//   knn->    80660,61681,55189,45591,36158,33826,   ...,         1
//
///////////////////////////////////////////////////////////////////

__global__ void rcv_build_prime_tree(
        unsigned int *d_plist,  /* pointer to list of primes */
        unsigned int pcount,    /* number of elements in list */
        unsigned int kcount,    /* maximum number of bits in upcoming sieves */
        unsigned int *d_kncount,/* pointer to list of counts of k-values per prime */
        unsigned int *d_ktree   /* pointer to tree of counts of k-values per prime */
        )
{
  unsigned int i;

  for (i=threadIdx.x; i<2*pcount; i+=blockDim.x)
    d_ktree[i] = 0;
  __syncthreads();      // Does this speed things?

  unsigned int pcountpow2;      /* next power of 2 >= pcount */

  for (pcountpow2=1; pcountpow2 < pcount; pcountpow2 = pcountpow2+pcountpow2)
    ;

  unsigned int ndeeper;         /* number of elements 1 level deeper in tree */
  ndeeper = pcount+pcount - pcountpow2;

#ifdef DEBUGTREE
#if (__CUDA_ARCH__ >= 200) && (CUDART_VERSION >= 4010)
  if (threadIdx.x == 0)
    printf("pcount = %u; pcountpow2 = %u; ndeeper = %u\n", pcount, pcountpow2, ndeeper);
#endif
#endif

  // When building the tree, we must work within a single block,
  // because blocks cannot synchronize with each other.  Since
  // this data isn't massively parallel, that's OK.  But we do
  // let multiple threads compute the differences.

  for (i=threadIdx.x; i<ndeeper; i+=blockDim.x)
  {
    d_kncount[i] = (kcount+d_plist[i]-1)/d_plist[i];  // Maximum bits to clear for this prime
    d_ktree[pcountpow2+i] = d_kncount[i];  // Copy from list to tree
  }

  __syncthreads();      // Does this speed things?

  for ( ; i<pcount; i+=blockDim.x)
  {
    d_kncount[i] = (kcount+d_plist[i]-1)/d_plist[i];  // Maximum bits to clear for this prime
    d_ktree[pcount+i-ndeeper] = d_kncount[i];  // Copy from list to tree
  }

  __syncthreads();      // Ensure all differences are tallied.

  // Let a single block build the tree
  if (blockIdx.x == 0)
  {
    i = pcount-1;
    while (blockDim.x < i/2)         // Can we put all threads to work?
    {
      unsigned int myi;
      myi = i - (blockDim.x-1) + threadIdx.x;
      d_ktree[myi] = d_ktree[2*myi] + d_ktree[2*myi+1];
      i = i - blockDim.x;
      __syncthreads();	// Warps may not be in lock-step, so sync up
    }

    // Let a single thread finish the tree
    if (threadIdx.x == 0)
      for ( ; i>0; i-=1)
        d_ktree[i] = d_ktree[2*i] + d_ktree[2*i+1];
  }
}


__global__ void rcv_set_sieve_bits(
        unsigned int kcount,    /* number of bits in upcoming sieve */
        unsigned int *d_bitmapw /* pointer to bitmap of 32-bit words */
        )
{
  /* One thread, per 32-bit word of bitmap, please */
  unsigned int i;
  i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < (kcount>>5))	// Excess threads don't participate
    d_bitmapw[i] = 0xffffffff;
}


// For your visualization pleasure, These tables show how patterns of consecutive
// factors of "13" and "23" appear through consecutive 32-bit words.

// Patterns of locations of factors of "13" in a 32-bit word
// .....1............1............1  [0] = 0, 13, 26
// ...........1............1.......  39, 52
// ....1............1............1.  65, 78, 91
// ..........1............1........  104, 117
// ...1............1............1..  130, 143, 156
// .........1............1.........  169, 182
// ..1............1............1...  195, 208, 221
// ........1............1..........  234, 247
// .1............1............1....  260, 273, 286
// .......1............1...........  299, 312
// 1............1............1.....  325, 338, 351
// ......1............1............  364, 377
// ............1............1......  390, 403,
// .....1............1............1  [13] = 416, 429, 442     32*13 = 416

// Patterns of locations of factors of "23" in a 32-bit word
// ........1......................1  [0] = 0, 23
// .................1..............  46
// ...1......................1.....  69, 92
// ............1...................  115
// .....................1..........  138
// .......1......................1.  161, 184
// ................1...............  207
// ..1......................1......  230, 253
// ...........1....................  276
// ....................1...........  299
// ......1......................1..  322, 345
// ...............1................  368
// .1......................1.......  391, 414
// ..........1.....................  437
// ...................1............  460
// .....1......................1...  483, 506
// ..............1.................  529
// 1......................1........  552, 575
// .........1......................  598
// ..................1.............  621
// ....1......................1....  644, 667
// .............1..................  690
// ......................1.........  713
// ........1......................1  [23] = 736, 759   32*23 = 736


__global__ void __launch_bounds__(256, 4) rcv_sieve_small_13_61(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta13,  /* pointer to list of 13 deltas for primes 13 through 61 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        )
{
  unsigned int i;
  __shared__ unsigned int s_bdelta[13]; // Space for 13 k-values in shared (fast) memory
#define kdelta13  s_bdelta[0]
#define kdelta17  s_bdelta[1]
#define kdelta19  s_bdelta[2]
#define kdelta23  s_bdelta[3]
#define kdelta29  s_bdelta[4]
#define kdelta31  s_bdelta[5]
#define kdelta37  s_bdelta[6]
#define kdelta41  s_bdelta[7]
#define kdelta43  s_bdelta[8]
#define kdelta47  s_bdelta[9]
#define kdelta53  s_bdelta[10]
#define kdelta59  s_bdelta[11]
#define kdelta61  s_bdelta[12]

  // We let the first 13 threads of each thread block simultaneously transfer
  // kdelta values from global memory to shared memory
  if (threadIdx.x < 13)
    s_bdelta[threadIdx.x] = d_bdelta13[threadIdx.x];

  __syncthreads();

// BITSLL:  13: 04002001  17: 00020001
// BITSRR:  13: 00080040  17: 00008000

#define BITSLL13 (1 | 1<<(   13) | 1<<(   13+13))
#define BITSRR13 (    1<<(32-13) | 1<<(32-13-13))

#define BITSLL17 (1 | 1<<(   17))
#define BITSRR17 (    1<<(32-17))

#define BITSLL19 (1 | 1<<(   19))
#define BITSRR19 (    1<<(32-19))

#define BITSLL23 (1 | 1<<(   23))
#define BITSRR23 (    1<<(32-23))

#define BITSLL29 (1 | 1<<(   29))
#define BITSRR29 (    1<<(32-29))

#define BITSLL31 (1 | 1<<(   31))
#define BITSRR31 (    1<<(32-31))

  // One thread, per 32-bit word of bitmap should be launched for this kernel, please.)
  i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < ((kcount+31)>>5))  // Excess threads don't participate.
  {
    unsigned int j;
    unsigned int k;
    unsigned int mask;

    // The following handles primes 13 < p < 32.  These are the primes
    // that fit within one 32-bit word, that haven't been inherently
    // sieved out via the "classes" mechanism.
    // Since we are executing with one thread per 32-bit word, we are
    // guaranteed to find sieve bits in our word for each and every prime.
    // With p = 13, we will always find 2 or 3 bits to sieve.
    // With 16 < p < 32, we will always find 1 or 2 bits to sieve.

    j = (i * 32 + 13-1 - kdelta13) / 13;        // Lowest bit this thread can reach with 32-bit accesses
    k = kdelta13 + j*13;
    mask  = (BITSLL13<<(k&31));

    j = (i * 32 + 17-1 - kdelta17) / 17;
    k = kdelta17 + j*17;
    mask |= (BITSLL17<<(k&31));

    j = (i * 32 + 19-1 - kdelta19) / 19;
    k = kdelta19 + j*19;
    mask |= (BITSLL19<<(k&31));

    j = (i * 32 + 23-1 - kdelta23) / 23;
    k = kdelta23+ j*23;
    mask |= (BITSLL23<<(k&31));

    j = (i * 32 + 29-1 - kdelta29) / 29;
    k = kdelta29 + j*29;
    mask |= (BITSLL29<<(k&31));

    j = (i * 32 + 31-1 - kdelta31) / 31;
    k = kdelta31 + j*31;
    mask |= (BITSLL31<<(k&31));


    // The following handles primes, 32 < p < 64.
    // Since we are executing with one thread per 32-bit word, we will either
    // find one or zero bits to sieve.  There is some inherent inefficiency,
    // since we may go through the motions, without sieving anything.
    // Theoretically, this technique could be used for even larger primes,
    // but the inefficiencies rapidly grow.

    j = (i * 32 + 37-1 - kdelta37) / 37;
    k = kdelta37 + j*37;
    mask |= (k>>5)==i ? (1<<(k&31)) : 0;

    j = (i * 32 + 41-1 - kdelta41) / 41;
    k = kdelta41 + j*41;
    mask |= (k>>5)==i ? (1<<(k&31)) : 0;

    j = (i * 32 + 43-1 - kdelta43) / 43;
    k = kdelta43 + j*43;
    mask |= (k>>5)==i ? (1<<(k&31)) : 0;

    j = (i * 32 + 47-1 - kdelta47) / 47;
    k = kdelta47 + j*47;
    mask |= (k>>5)==i ? (1<<(k&31)) : 0;

    j = (i * 32 + 53-1 - kdelta53) / 53;
    k = kdelta53 + j*53;
    mask |= (k>>5)==i ? (1<<(k&31)) : 0;

    j = (i * 32 + 59-1 - kdelta59) / 59;
    k = kdelta59 + j*59;
    mask |= (k>>5)==i ? (1<<(k&31)) : 0;

    j = (i * 32 + 61-1 - kdelta61) / 61;
    k = kdelta61 + j*61;
    mask |= (k>>5)==i ? (1<<(k&31)) : 0;

    if (d_bitmapw[i] & mask)    // Are any of our bits still on?
    {
      d_bitmapw[i] &= ~mask;    // If yes, turn them off.  If no, save a memory write.
    }
  }
}


__global__ void __launch_bounds__(256, 4) rcv_sieve_small_67_127(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta67,  /* pointer to list of 13 deltas for primes 67 through 127 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        )
{
  unsigned int i;
  __shared__ unsigned int smap[512];  // Two 32-bit words per thread per block
  __shared__ unsigned int s_bdelta[13]; // Space for 13 k-values in shared (fast) memory
#define kdelta67  s_bdelta[0]
#define kdelta71  s_bdelta[1]
#define kdelta73  s_bdelta[2]
#define kdelta79  s_bdelta[3]
#define kdelta83  s_bdelta[4]
#define kdelta89  s_bdelta[5]
#define kdelta97  s_bdelta[6]
#define kdelta101 s_bdelta[7]
#define kdelta103 s_bdelta[8]
#define kdelta107 s_bdelta[9]
#define kdelta109 s_bdelta[10]
#define kdelta113 s_bdelta[11]
#define kdelta127 s_bdelta[12]

  // We let the first 13 threads of each thread block simultaneously transfer
  // kdelta values from global memory to shared memory
  if (threadIdx.x < 13)
    s_bdelta[threadIdx.x] = d_bdelta67[threadIdx.x];

  __syncthreads();

  // One thread, per 64-bit word of bitmap should be launched for this kernel, please.)
  i = blockDim.x * blockIdx.x + threadIdx.x;

  // All threads *must* participate, since they write each other's results to global memory
  {
    unsigned int j;
    unsigned int k;

    // The following handles primes, 64 < p < 128.
    // Since we are executing with one thread per 64-bit word, we will either
    // find one or zero bits to sieve.

    // The bits we sieve will be ORed into one of these two 32-bit words.
    smap[2*threadIdx.x  ] = 0;
    smap[2*threadIdx.x+1] = 0;

#define SIEVE_64_BIT(p, kdeltap) { \
    j = (i * 64 + p-1 - kdeltap) / p; \
    k = kdeltap + j*p; \
    if ((k>>6) == i) \
      smap[2*threadIdx.x+((k>>5)&1)] |= 1<<(k&31); \
    }

    SIEVE_64_BIT( 67, kdelta67);
    SIEVE_64_BIT( 71, kdelta71);
    SIEVE_64_BIT( 73, kdelta73);
    SIEVE_64_BIT( 79, kdelta79);
    SIEVE_64_BIT( 83, kdelta83);
    SIEVE_64_BIT( 89, kdelta89);
    SIEVE_64_BIT( 97, kdelta97);
    SIEVE_64_BIT(101, kdelta101);
    SIEVE_64_BIT(103, kdelta103);
    SIEVE_64_BIT(107, kdelta107);
    SIEVE_64_BIT(109, kdelta109);
    SIEVE_64_BIT(113, kdelta113);
    SIEVE_64_BIT(127, kdelta127);

    __syncthreads();                    // Make sure everybody has stored their results
    d_bitmapw[2*i-2*threadIdx.x+           threadIdx.x] &= ~smap[           threadIdx.x];
    d_bitmapw[2*i-2*threadIdx.x+blockDim.x+threadIdx.x] &= ~smap[blockDim.x+threadIdx.x];
  }
}


__global__ void __launch_bounds__(256, 4) rcv_sieve_small_131_251(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta131,  /* pointer to list of 23 deltas for primes 131 through 251 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        )
{
  unsigned int i;
  __shared__ unsigned int smap[4*256];  // Four 32-bit words per thread per block
  __shared__ unsigned int s_bdelta[23]; // Space for 23 k-values in shared (fast) memory
#define kdelta131 s_bdelta[0]
#define kdelta137 s_bdelta[1]
#define kdelta139 s_bdelta[2]
#define kdelta149 s_bdelta[3]
#define kdelta151 s_bdelta[4]
#define kdelta157 s_bdelta[5]
#define kdelta163 s_bdelta[6]
#define kdelta167 s_bdelta[7]
#define kdelta173 s_bdelta[8]
#define kdelta179 s_bdelta[9]
#define kdelta181 s_bdelta[10]
#define kdelta191 s_bdelta[11]
#define kdelta193 s_bdelta[12]
#define kdelta197 s_bdelta[13]
#define kdelta199 s_bdelta[14]
#define kdelta211 s_bdelta[15]
#define kdelta223 s_bdelta[16]
#define kdelta227 s_bdelta[17]
#define kdelta229 s_bdelta[18]
#define kdelta233 s_bdelta[19]
#define kdelta239 s_bdelta[20]
#define kdelta241 s_bdelta[21]
#define kdelta251 s_bdelta[22]

  // We let the first 23 threads of each thread block simultaneously transfer
  // kdelta values from global memory to shared memory
  if (threadIdx.x < 23)
    s_bdelta[threadIdx.x] = d_bdelta131[threadIdx.x];

  __syncthreads();

  // One thread, per 128-bit word of bitmap should be launched for this kernel, please.)
  i = blockDim.x * blockIdx.x + threadIdx.x;

  // All threads *must* participate, since they write each other's results to global memory
  {
    unsigned int j;
    unsigned int k;

    // The following handles primes, 128 < p < 256.
    // Since we are executing with one thread per 128-bit word, we will either
    // find one or zero bits to sieve per thread.

    // The bits we sieve will be ORed into one of these four 32-bit words.
    smap[4*threadIdx.x  ] = 0;
    smap[4*threadIdx.x+1] = 0;
    smap[4*threadIdx.x+2] = 0;
    smap[4*threadIdx.x+3] = 0;

#define SIEVE_128_BIT(p, kdeltap) { \
    j = (i * 128 + p-1 - kdeltap) / p; \
    k = kdeltap + j*p; \
    if ((k>>7) == i) \
      smap[4*threadIdx.x+((k>>5)&3)] |= 1<<(k&31); \
    }

    SIEVE_128_BIT(131, kdelta131);
    SIEVE_128_BIT(137, kdelta137);
    SIEVE_128_BIT(139, kdelta139);
    SIEVE_128_BIT(149, kdelta149);
    SIEVE_128_BIT(151, kdelta151);
    SIEVE_128_BIT(157, kdelta157);
    SIEVE_128_BIT(163, kdelta163);
    SIEVE_128_BIT(167, kdelta167);
    SIEVE_128_BIT(173, kdelta173);
    SIEVE_128_BIT(179, kdelta179);
    SIEVE_128_BIT(181, kdelta181);
    SIEVE_128_BIT(191, kdelta191);
    SIEVE_128_BIT(193, kdelta193);
    SIEVE_128_BIT(197, kdelta197);
    SIEVE_128_BIT(199, kdelta199);
    SIEVE_128_BIT(211, kdelta211);
    SIEVE_128_BIT(223, kdelta223);
    SIEVE_128_BIT(227, kdelta227);
    SIEVE_128_BIT(229, kdelta229);
    SIEVE_128_BIT(233, kdelta233);
    SIEVE_128_BIT(239, kdelta239);
    SIEVE_128_BIT(241, kdelta241);
    SIEVE_128_BIT(251, kdelta251);

// Which strategy to write results to global memory?  atomicAnd isn't really necessary, as long
// as all kernel's prior to this one have finished before our kernel runs.
// Scrambling the copies from shared memory to global memory yields ~100% global load/store efficiency,
// but doesn't actually have much performance gain.
// Straightforward copy reports 25% global load/store efficiency, but isn't much slower.

    __syncthreads();    // Make sure everybody has stored their results
    d_bitmapw[4*i-4*threadIdx.x+             threadIdx.x] &= ~smap[             threadIdx.x];
    d_bitmapw[4*i-4*threadIdx.x+  blockDim.x+threadIdx.x] &= ~smap[  blockDim.x+threadIdx.x];
    d_bitmapw[4*i-4*threadIdx.x+2*blockDim.x+threadIdx.x] &= ~smap[2*blockDim.x+threadIdx.x];
    d_bitmapw[4*i-4*threadIdx.x+3*blockDim.x+threadIdx.x] &= ~smap[3*blockDim.x+threadIdx.x];
  }
}

__global__ void __launch_bounds__(128, 4) rcv_sieve_small_257_509(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta257,  /* pointer to list of 43 deltas for primes 257 through 509 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        )
{
  unsigned int i;
  __shared__ unsigned int smap[8*128];  // Eight 32-bit words per thread per block
  __shared__ unsigned int s_bdelta[43]; // Space for 43 k-values in shared (fast) memory
#define kdelta257 s_bdelta[0]
#define kdelta263 s_bdelta[1]
#define kdelta269 s_bdelta[2]
#define kdelta271 s_bdelta[3]
#define kdelta277 s_bdelta[4]
#define kdelta281 s_bdelta[5]
#define kdelta283 s_bdelta[6]
#define kdelta293 s_bdelta[7]
#define kdelta307 s_bdelta[8]
#define kdelta311 s_bdelta[9]
#define kdelta313 s_bdelta[10]
#define kdelta317 s_bdelta[11]
#define kdelta331 s_bdelta[12]
#define kdelta337 s_bdelta[13]
#define kdelta347 s_bdelta[14]
#define kdelta349 s_bdelta[15]
#define kdelta353 s_bdelta[16]
#define kdelta359 s_bdelta[17]
#define kdelta367 s_bdelta[18]
#define kdelta373 s_bdelta[19]
#define kdelta379 s_bdelta[20]
#define kdelta383 s_bdelta[21]
#define kdelta389 s_bdelta[22]
#define kdelta397 s_bdelta[23]
#define kdelta401 s_bdelta[24]
#define kdelta409 s_bdelta[25]
#define kdelta419 s_bdelta[26]
#define kdelta421 s_bdelta[27]
#define kdelta431 s_bdelta[28]
#define kdelta433 s_bdelta[29]
#define kdelta439 s_bdelta[30]
#define kdelta443 s_bdelta[31]
#define kdelta449 s_bdelta[32]
#define kdelta457 s_bdelta[33]
#define kdelta461 s_bdelta[34]
#define kdelta463 s_bdelta[35]
#define kdelta467 s_bdelta[36]
#define kdelta479 s_bdelta[37]
#define kdelta487 s_bdelta[38]
#define kdelta491 s_bdelta[39]
#define kdelta499 s_bdelta[40]
#define kdelta503 s_bdelta[41]
#define kdelta509 s_bdelta[42]

  // We let the first 43 threads of each thread block simultaneously transfer
  // kdelta values from global memory to shared memory
  if (threadIdx.x < 43)
    s_bdelta[threadIdx.x] = d_bdelta257[threadIdx.x];

  __syncthreads();

  // One thread, per 256-bit word of bitmap should be launched for this kernel, please.)
  i = blockDim.x * blockIdx.x + threadIdx.x;

  // All threads *must* participate, since they write each other's results to global memory
  {
    unsigned int j;
    unsigned int k;

    // The following handles primes, 256 < p < 512.
    // Since we are executing with one thread per 256-bit word, we will either
    // find one or zero bits to sieve per thread per prime.

    // The bits we sieve will be ORed into one of these eight 32-bit words.
    smap[8*threadIdx.x  ] = 0;
    smap[8*threadIdx.x+1] = 0;
    smap[8*threadIdx.x+2] = 0;
    smap[8*threadIdx.x+3] = 0;
    smap[8*threadIdx.x+4] = 0;
    smap[8*threadIdx.x+5] = 0;
    smap[8*threadIdx.x+6] = 0;
    smap[8*threadIdx.x+7] = 0;

#define SIEVE_256_BIT(p, kdeltap) { \
    j = (i * 256 + p-1 - kdeltap) / p; \
    k = kdeltap + j*p; \
    if ((k>>8) == i) \
      smap[8*threadIdx.x+((k>>5)&7)] |= 1<<(k&31); \
    }

    SIEVE_256_BIT(257, kdelta257);
    SIEVE_256_BIT(263, kdelta263);
    SIEVE_256_BIT(269, kdelta269);
    SIEVE_256_BIT(271, kdelta271);
    SIEVE_256_BIT(277, kdelta277);
    SIEVE_256_BIT(281, kdelta281);
    SIEVE_256_BIT(283, kdelta283);
    SIEVE_256_BIT(293, kdelta293);
    SIEVE_256_BIT(307, kdelta307);
    SIEVE_256_BIT(311, kdelta311);
    SIEVE_256_BIT(313, kdelta313);
    SIEVE_256_BIT(317, kdelta317);
    SIEVE_256_BIT(331, kdelta331);
    SIEVE_256_BIT(337, kdelta337);
    SIEVE_256_BIT(347, kdelta347);
    SIEVE_256_BIT(349, kdelta349);
    SIEVE_256_BIT(353, kdelta353);
    SIEVE_256_BIT(359, kdelta359);
    SIEVE_256_BIT(367, kdelta367);
    SIEVE_256_BIT(373, kdelta373);
    SIEVE_256_BIT(379, kdelta379);
    SIEVE_256_BIT(383, kdelta383);
    SIEVE_256_BIT(389, kdelta389);
    SIEVE_256_BIT(397, kdelta397);
    SIEVE_256_BIT(401, kdelta401);
    SIEVE_256_BIT(409, kdelta409);
    SIEVE_256_BIT(419, kdelta419);
    SIEVE_256_BIT(421, kdelta421);
    SIEVE_256_BIT(431, kdelta431);
    SIEVE_256_BIT(433, kdelta433);
    SIEVE_256_BIT(439, kdelta439);
    SIEVE_256_BIT(443, kdelta443);
    SIEVE_256_BIT(449, kdelta449);
    SIEVE_256_BIT(457, kdelta457);
    SIEVE_256_BIT(461, kdelta461);
    SIEVE_256_BIT(463, kdelta463);
    SIEVE_256_BIT(467, kdelta467);
    SIEVE_256_BIT(479, kdelta479);
    SIEVE_256_BIT(487, kdelta487);
    SIEVE_256_BIT(491, kdelta491);
    SIEVE_256_BIT(499, kdelta499);
    SIEVE_256_BIT(503, kdelta503);
    SIEVE_256_BIT(509, kdelta509);

// Which strategy to write results to global memory?  atomicAnd isn't really necessary, as long
// as all kernel's prior to this one have finished before out kernel runs.
// Scrambling the copies from shared memory to global memory yields ~100% global load/store efficiency,
// but doesn't actually have much performance gain.
// Straightforward copy reports 25% global load/store efficiency, and is only a little slower.

    __syncthreads();    // Make sure everybody has stored their results
    d_bitmapw[8*i-8*threadIdx.x+             threadIdx.x] &= ~smap[             threadIdx.x];
    d_bitmapw[8*i-8*threadIdx.x+  blockDim.x+threadIdx.x] &= ~smap[  blockDim.x+threadIdx.x];
    d_bitmapw[8*i-8*threadIdx.x+2*blockDim.x+threadIdx.x] &= ~smap[2*blockDim.x+threadIdx.x];
    d_bitmapw[8*i-8*threadIdx.x+3*blockDim.x+threadIdx.x] &= ~smap[3*blockDim.x+threadIdx.x];
    d_bitmapw[8*i-8*threadIdx.x+4*blockDim.x+threadIdx.x] &= ~smap[4*blockDim.x+threadIdx.x];
    d_bitmapw[8*i-8*threadIdx.x+5*blockDim.x+threadIdx.x] &= ~smap[5*blockDim.x+threadIdx.x];
    d_bitmapw[8*i-8*threadIdx.x+6*blockDim.x+threadIdx.x] &= ~smap[6*blockDim.x+threadIdx.x];
    d_bitmapw[8*i-8*threadIdx.x+7*blockDim.x+threadIdx.x] &= ~smap[7*blockDim.x+threadIdx.x];
  }
}


__global__ void __launch_bounds__(128, 4) rcv_sieve_small_521_1021(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bdelta521,  /* pointer to list of 75 deltas for primes 521 through 1021 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        )
{
  unsigned int i;
  __shared__ unsigned int smap[16*128]; // Sixteen 32-bit words per thread per block
  __shared__ unsigned int s_bdelta[75]; // Space for 75 k-values in shared (fast) memory
#define kdelta521 s_bdelta[0]
#define kdelta523 s_bdelta[1]
#define kdelta541 s_bdelta[2]
#define kdelta547 s_bdelta[3]
#define kdelta557 s_bdelta[4]
#define kdelta563 s_bdelta[5]
#define kdelta569 s_bdelta[6]
#define kdelta571 s_bdelta[7]
#define kdelta577 s_bdelta[8]
#define kdelta587 s_bdelta[9]
#define kdelta593 s_bdelta[10]
#define kdelta599 s_bdelta[11]
#define kdelta601 s_bdelta[12]
#define kdelta607 s_bdelta[13]
#define kdelta613 s_bdelta[14]
#define kdelta617 s_bdelta[15]
#define kdelta619 s_bdelta[16]
#define kdelta631 s_bdelta[17]
#define kdelta641 s_bdelta[18]
#define kdelta643 s_bdelta[19]
#define kdelta647 s_bdelta[20]
#define kdelta653 s_bdelta[21]
#define kdelta659 s_bdelta[22]
#define kdelta661 s_bdelta[23]
#define kdelta673 s_bdelta[24]
#define kdelta677 s_bdelta[25]
#define kdelta683 s_bdelta[26]
#define kdelta691 s_bdelta[27]
#define kdelta701 s_bdelta[28]
#define kdelta709 s_bdelta[29]
#define kdelta719 s_bdelta[30]
#define kdelta727 s_bdelta[31]
#define kdelta733 s_bdelta[32]
#define kdelta739 s_bdelta[33]
#define kdelta743 s_bdelta[34]
#define kdelta751 s_bdelta[35]
#define kdelta757 s_bdelta[36]
#define kdelta761 s_bdelta[37]
#define kdelta769 s_bdelta[38]
#define kdelta773 s_bdelta[39]
#define kdelta787 s_bdelta[40]
#define kdelta797 s_bdelta[41]
#define kdelta809 s_bdelta[42]
#define kdelta811 s_bdelta[43]
#define kdelta821 s_bdelta[44]
#define kdelta823 s_bdelta[45]
#define kdelta827 s_bdelta[46]
#define kdelta829 s_bdelta[47]
#define kdelta839 s_bdelta[48]
#define kdelta853 s_bdelta[49]
#define kdelta857 s_bdelta[50]
#define kdelta859 s_bdelta[51]
#define kdelta863 s_bdelta[52]
#define kdelta877 s_bdelta[53]
#define kdelta881 s_bdelta[54]
#define kdelta883 s_bdelta[55]
#define kdelta887 s_bdelta[56]
#define kdelta907 s_bdelta[57]
#define kdelta911 s_bdelta[58]
#define kdelta919 s_bdelta[59]
#define kdelta929 s_bdelta[60]
#define kdelta937 s_bdelta[61]
#define kdelta941 s_bdelta[62]
#define kdelta947 s_bdelta[63]
#define kdelta953 s_bdelta[64]
#define kdelta967 s_bdelta[65]
#define kdelta971 s_bdelta[66]
#define kdelta977 s_bdelta[67]
#define kdelta983 s_bdelta[68]
#define kdelta991 s_bdelta[69]
#define kdelta997 s_bdelta[70]
#define kdelta1009 s_bdelta[71]
#define kdelta1013 s_bdelta[72]
#define kdelta1019 s_bdelta[73]
#define kdelta1021 s_bdelta[74]

  // CAUTION:  Following code will not work if threadsPerBlock is less than 64

  // Simultaneously transfer maximum number of kdelta values
  if (threadIdx.x < 64)
    s_bdelta[threadIdx.x   ] = d_bdelta521[threadIdx.x   ];
  if (threadIdx.x < 75-64)
    s_bdelta[threadIdx.x+64] = d_bdelta521[threadIdx.x+64];

  __syncthreads();

  // One thread, per 512-bit word of bitmap should be launched for this kernel, please.)
  i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < ((kcount+511)>>9))  // Excess threads don't participate.
  // All threads *must* participate, since they write each other's results to global memory
  {
    unsigned int j;
    unsigned int k;

    // The following handles primes, 512 < p < 1024.
    // Since we are executing with one thread per 512-bit word, we will either
    // find one or zero bits to sieve per thread per prime.

    // The bits we sieve will be ORed into one of these sixteen 32-bit words.
    smap[16*threadIdx.x   ] = 0;
    smap[16*threadIdx.x+ 1] = 0;
    smap[16*threadIdx.x+ 2] = 0;
    smap[16*threadIdx.x+ 3] = 0;
    smap[16*threadIdx.x+ 4] = 0;
    smap[16*threadIdx.x+ 5] = 0;
    smap[16*threadIdx.x+ 6] = 0;
    smap[16*threadIdx.x+ 7] = 0;
    smap[16*threadIdx.x+ 8] = 0;
    smap[16*threadIdx.x+ 9] = 0;
    smap[16*threadIdx.x+10] = 0;
    smap[16*threadIdx.x+11] = 0;
    smap[16*threadIdx.x+12] = 0;
    smap[16*threadIdx.x+13] = 0;
    smap[16*threadIdx.x+14] = 0;
    smap[16*threadIdx.x+15] = 0;

#define SIEVE_512_BIT(p, kdeltap) { \
    j = (i * 512 + p-1 - kdeltap) / p; \
    k = kdeltap + j*p; \
    if ((k>>9) == i) \
      smap[16*threadIdx.x+((k>>5)&15)] |= 1<<(k&31); \
    }

    SIEVE_512_BIT( 521, kdelta521 );
    SIEVE_512_BIT( 523, kdelta523 );
    SIEVE_512_BIT( 541, kdelta541 );
    SIEVE_512_BIT( 547, kdelta547 );
    SIEVE_512_BIT( 557, kdelta557 );
    SIEVE_512_BIT( 563, kdelta563 );
    SIEVE_512_BIT( 569, kdelta569 );
    SIEVE_512_BIT( 571, kdelta571 );
    SIEVE_512_BIT( 577, kdelta577 );
    SIEVE_512_BIT( 587, kdelta587 );
    SIEVE_512_BIT( 593, kdelta593 );
    SIEVE_512_BIT( 599, kdelta599 );
    SIEVE_512_BIT( 601, kdelta601 );
    SIEVE_512_BIT( 607, kdelta607 );
    SIEVE_512_BIT( 613, kdelta613 );
    SIEVE_512_BIT( 617, kdelta617 );
    SIEVE_512_BIT( 619, kdelta619 );
    SIEVE_512_BIT( 631, kdelta631 );
    SIEVE_512_BIT( 641, kdelta641 );
    SIEVE_512_BIT( 643, kdelta643 );
    SIEVE_512_BIT( 647, kdelta647 );
    SIEVE_512_BIT( 653, kdelta653 );
    SIEVE_512_BIT( 659, kdelta659 );
    SIEVE_512_BIT( 661, kdelta661 );
    SIEVE_512_BIT( 673, kdelta673 );
    SIEVE_512_BIT( 677, kdelta677 );
    SIEVE_512_BIT( 683, kdelta683 );
    SIEVE_512_BIT( 691, kdelta691 );
    SIEVE_512_BIT( 701, kdelta701 );
    SIEVE_512_BIT( 709, kdelta709 );
    SIEVE_512_BIT( 719, kdelta719 );
    SIEVE_512_BIT( 727, kdelta727 );
    SIEVE_512_BIT( 733, kdelta733 );
    SIEVE_512_BIT( 739, kdelta739 );
    SIEVE_512_BIT( 743, kdelta743 );
    SIEVE_512_BIT( 751, kdelta751 );
    SIEVE_512_BIT( 757, kdelta757 );
    SIEVE_512_BIT( 761, kdelta761 );
    SIEVE_512_BIT( 769, kdelta769 );
    SIEVE_512_BIT( 773, kdelta773 );
    SIEVE_512_BIT( 787, kdelta787 );
    SIEVE_512_BIT( 797, kdelta797 );
    SIEVE_512_BIT( 809, kdelta809 );
    SIEVE_512_BIT( 811, kdelta811 );
    SIEVE_512_BIT( 821, kdelta821 );
    SIEVE_512_BIT( 823, kdelta823 );
    SIEVE_512_BIT( 827, kdelta827 );
    SIEVE_512_BIT( 829, kdelta829 );
    SIEVE_512_BIT( 839, kdelta839 );
    SIEVE_512_BIT( 853, kdelta853 );
    SIEVE_512_BIT( 857, kdelta857 );
    SIEVE_512_BIT( 859, kdelta859 );
    SIEVE_512_BIT( 863, kdelta863 );
    SIEVE_512_BIT( 877, kdelta877 );
    SIEVE_512_BIT( 881, kdelta881 );
    SIEVE_512_BIT( 883, kdelta883 );
    SIEVE_512_BIT( 887, kdelta887 );
    SIEVE_512_BIT( 907, kdelta907 );
    SIEVE_512_BIT( 911, kdelta911 );
    SIEVE_512_BIT( 919, kdelta919 );
    SIEVE_512_BIT( 929, kdelta929 );
    SIEVE_512_BIT( 937, kdelta937 );
    SIEVE_512_BIT( 941, kdelta941 );
    SIEVE_512_BIT( 947, kdelta947 );
    SIEVE_512_BIT( 953, kdelta953 );
    SIEVE_512_BIT( 967, kdelta967 );
    SIEVE_512_BIT( 971, kdelta971 );
    SIEVE_512_BIT( 977, kdelta977 );
    SIEVE_512_BIT( 983, kdelta983 );
    SIEVE_512_BIT( 991, kdelta991 );
    SIEVE_512_BIT( 997, kdelta997 );
    SIEVE_512_BIT(1009, kdelta1009);
    SIEVE_512_BIT(1013, kdelta1013);
    SIEVE_512_BIT(1019, kdelta1019);
    SIEVE_512_BIT(1021, kdelta1021);

// Which strategy to write results to global memory?  atomicAnd isn't really necessary, as long
// as all kernel's prior to this one have finished before our kernel runs.
// Scrambling the copies from shared memory to global memory yields ~100% global load/store efficiency,
// but doesn't actually have a huge performance gain.
// Straightforward copy reports 6.25% global load/store efficiency.

    __syncthreads();    // Make sure everybody has stored their results
    d_bitmapw[16*i-16*threadIdx.x+              threadIdx.x] &= ~smap[              threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+   blockDim.x+threadIdx.x] &= ~smap[   blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+ 2*blockDim.x+threadIdx.x] &= ~smap[ 2*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+ 3*blockDim.x+threadIdx.x] &= ~smap[ 3*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+ 4*blockDim.x+threadIdx.x] &= ~smap[ 4*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+ 5*blockDim.x+threadIdx.x] &= ~smap[ 5*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+ 6*blockDim.x+threadIdx.x] &= ~smap[ 6*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+ 7*blockDim.x+threadIdx.x] &= ~smap[ 7*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+ 8*blockDim.x+threadIdx.x] &= ~smap[ 8*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+ 9*blockDim.x+threadIdx.x] &= ~smap[ 9*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+10*blockDim.x+threadIdx.x] &= ~smap[10*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+11*blockDim.x+threadIdx.x] &= ~smap[11*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+12*blockDim.x+threadIdx.x] &= ~smap[12*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+13*blockDim.x+threadIdx.x] &= ~smap[13*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+14*blockDim.x+threadIdx.x] &= ~smap[14*blockDim.x+threadIdx.x];
    d_bitmapw[16*i-16*threadIdx.x+15*blockDim.x+threadIdx.x] &= ~smap[15*blockDim.x+threadIdx.x];
    return;
  }
}


// To limit our shared memory footprint, we launch with a small threads per block.
// Note that we tested with 32 TPB, 64 TPB, and 256 TPB.  In each case, the
// run-time of the kernel remained (at approximately 100 us), and the occupancy
// remained near 1/6.  Should investigate if something is limiting our occupancy.
__global__ void __launch_bounds__(32, 4) rcv_sieve_small_1031_2039(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_kdelta1031,  /* pointer to list of 137 deltas for primes 1031 through 2039 */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        )
{
  unsigned int i;
  __shared__ unsigned int smap[32*32];   // Thirty-two 32-bit words per thread per block
  __shared__ unsigned int s_kdelta[137]; // Space for 137 k-values in shared (fast) memory
#define kdelta1031 s_kdelta[0]
#define kdelta1033 s_kdelta[1]
#define kdelta1039 s_kdelta[2]
#define kdelta1049 s_kdelta[3]
#define kdelta1051 s_kdelta[4]
#define kdelta1061 s_kdelta[5]
#define kdelta1063 s_kdelta[6]
#define kdelta1069 s_kdelta[7]
#define kdelta1087 s_kdelta[8]
#define kdelta1091 s_kdelta[9]
#define kdelta1093 s_kdelta[10]
#define kdelta1097 s_kdelta[11]
#define kdelta1103 s_kdelta[12]
#define kdelta1109 s_kdelta[13]
#define kdelta1117 s_kdelta[14]
#define kdelta1123 s_kdelta[15]
#define kdelta1129 s_kdelta[16]
#define kdelta1151 s_kdelta[17]
#define kdelta1153 s_kdelta[18]
#define kdelta1163 s_kdelta[19]
#define kdelta1171 s_kdelta[20]
#define kdelta1181 s_kdelta[21]
#define kdelta1187 s_kdelta[22]
#define kdelta1193 s_kdelta[23]
#define kdelta1201 s_kdelta[24]
#define kdelta1213 s_kdelta[25]
#define kdelta1217 s_kdelta[26]
#define kdelta1223 s_kdelta[27]
#define kdelta1229 s_kdelta[28]
#define kdelta1231 s_kdelta[29]
#define kdelta1237 s_kdelta[30]
#define kdelta1249 s_kdelta[31]
#define kdelta1259 s_kdelta[32]
#define kdelta1277 s_kdelta[33]
#define kdelta1279 s_kdelta[34]
#define kdelta1283 s_kdelta[35]
#define kdelta1289 s_kdelta[36]
#define kdelta1291 s_kdelta[37]
#define kdelta1297 s_kdelta[38]
#define kdelta1301 s_kdelta[39]
#define kdelta1303 s_kdelta[40]
#define kdelta1307 s_kdelta[41]
#define kdelta1319 s_kdelta[42]
#define kdelta1321 s_kdelta[43]
#define kdelta1327 s_kdelta[44]
#define kdelta1361 s_kdelta[45]
#define kdelta1367 s_kdelta[46]
#define kdelta1373 s_kdelta[47]
#define kdelta1381 s_kdelta[48]
#define kdelta1399 s_kdelta[49]
#define kdelta1409 s_kdelta[50]
#define kdelta1423 s_kdelta[51]
#define kdelta1427 s_kdelta[52]
#define kdelta1429 s_kdelta[53]
#define kdelta1433 s_kdelta[54]
#define kdelta1439 s_kdelta[55]
#define kdelta1447 s_kdelta[56]
#define kdelta1451 s_kdelta[57]
#define kdelta1453 s_kdelta[58]
#define kdelta1459 s_kdelta[59]
#define kdelta1471 s_kdelta[60]
#define kdelta1481 s_kdelta[61]
#define kdelta1483 s_kdelta[62]
#define kdelta1487 s_kdelta[63]
#define kdelta1489 s_kdelta[64]
#define kdelta1493 s_kdelta[65]
#define kdelta1499 s_kdelta[66]
#define kdelta1511 s_kdelta[67]
#define kdelta1523 s_kdelta[68]
#define kdelta1531 s_kdelta[69]
#define kdelta1543 s_kdelta[70]
#define kdelta1549 s_kdelta[71]
#define kdelta1553 s_kdelta[72]
#define kdelta1559 s_kdelta[73]
#define kdelta1567 s_kdelta[74]
#define kdelta1571 s_kdelta[75]
#define kdelta1579 s_kdelta[76]
#define kdelta1583 s_kdelta[77]
#define kdelta1597 s_kdelta[78]
#define kdelta1601 s_kdelta[79]
#define kdelta1607 s_kdelta[80]
#define kdelta1609 s_kdelta[81]
#define kdelta1613 s_kdelta[82]
#define kdelta1619 s_kdelta[83]
#define kdelta1621 s_kdelta[84]
#define kdelta1627 s_kdelta[85]
#define kdelta1637 s_kdelta[86]
#define kdelta1657 s_kdelta[87]
#define kdelta1663 s_kdelta[88]
#define kdelta1667 s_kdelta[89]
#define kdelta1669 s_kdelta[90]
#define kdelta1693 s_kdelta[91]
#define kdelta1697 s_kdelta[92]
#define kdelta1699 s_kdelta[93]
#define kdelta1709 s_kdelta[94]
#define kdelta1721 s_kdelta[95]
#define kdelta1723 s_kdelta[96]
#define kdelta1733 s_kdelta[97]
#define kdelta1741 s_kdelta[98]
#define kdelta1747 s_kdelta[99]
#define kdelta1753 s_kdelta[100]
#define kdelta1759 s_kdelta[101]
#define kdelta1777 s_kdelta[102]
#define kdelta1783 s_kdelta[103]
#define kdelta1787 s_kdelta[104]
#define kdelta1789 s_kdelta[105]
#define kdelta1801 s_kdelta[106]
#define kdelta1811 s_kdelta[107]
#define kdelta1823 s_kdelta[108]
#define kdelta1831 s_kdelta[109]
#define kdelta1847 s_kdelta[110]
#define kdelta1861 s_kdelta[111]
#define kdelta1867 s_kdelta[112]
#define kdelta1871 s_kdelta[113]
#define kdelta1873 s_kdelta[114]
#define kdelta1877 s_kdelta[115]
#define kdelta1879 s_kdelta[116]
#define kdelta1889 s_kdelta[117]
#define kdelta1901 s_kdelta[118]
#define kdelta1907 s_kdelta[119]
#define kdelta1913 s_kdelta[120]
#define kdelta1931 s_kdelta[121]
#define kdelta1933 s_kdelta[122]
#define kdelta1949 s_kdelta[123]
#define kdelta1951 s_kdelta[124]
#define kdelta1973 s_kdelta[125]
#define kdelta1979 s_kdelta[126]
#define kdelta1987 s_kdelta[127]
#define kdelta1993 s_kdelta[128]
#define kdelta1997 s_kdelta[129]
#define kdelta1999 s_kdelta[130]
#define kdelta2003 s_kdelta[131]
#define kdelta2011 s_kdelta[132]
#define kdelta2017 s_kdelta[133]
#define kdelta2027 s_kdelta[134]
#define kdelta2029 s_kdelta[135]
#define kdelta2039 s_kdelta[136]

  // CAUTION:  Following code will not work if threadsPerBlock is less than 32

  // Simultaneously transfer maximum number of kdelta values
  if (threadIdx.x < 32)
  {
    s_kdelta[threadIdx.x    ] = d_kdelta1031[threadIdx.x    ];
    s_kdelta[threadIdx.x+ 32] = d_kdelta1031[threadIdx.x+ 32];
    s_kdelta[threadIdx.x+ 64] = d_kdelta1031[threadIdx.x+ 64];
    s_kdelta[threadIdx.x+ 96] = d_kdelta1031[threadIdx.x+ 96];
  }
  if (threadIdx.x < 137-128)
    s_kdelta[threadIdx.x+128] = d_kdelta1031[threadIdx.x+128];

  __syncthreads();              // Is this necessary?

  // One thread, per 1024-bit word of bitmap should be launched for this kernel, please.)
  i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < ((kcount+1023)>>10))  // Excess threads don't participate.
  // All threads participate.
  {
    unsigned int j;
    unsigned int k;

    // The following handles primes, 1024 < p < 2048.
    // Since we are executing with one thread per 1024-bit word, we will either
    // find one or zero bits to sieve per thread per prime.

    // The bits we sieve will be ORed into one of these thirty-two 32-bit words.
    smap[32*threadIdx.x   ] = 0;
    smap[32*threadIdx.x+ 1] = 0;
    smap[32*threadIdx.x+ 2] = 0;
    smap[32*threadIdx.x+ 3] = 0;
    smap[32*threadIdx.x+ 4] = 0;
    smap[32*threadIdx.x+ 5] = 0;
    smap[32*threadIdx.x+ 6] = 0;
    smap[32*threadIdx.x+ 7] = 0;
    smap[32*threadIdx.x+ 8] = 0;
    smap[32*threadIdx.x+ 9] = 0;
    smap[32*threadIdx.x+10] = 0;
    smap[32*threadIdx.x+11] = 0;
    smap[32*threadIdx.x+12] = 0;
    smap[32*threadIdx.x+13] = 0;
    smap[32*threadIdx.x+14] = 0;
    smap[32*threadIdx.x+15] = 0;
    smap[32*threadIdx.x+16] = 0;
    smap[32*threadIdx.x+17] = 0;
    smap[32*threadIdx.x+18] = 0;
    smap[32*threadIdx.x+19] = 0;
    smap[32*threadIdx.x+20] = 0;
    smap[32*threadIdx.x+21] = 0;
    smap[32*threadIdx.x+22] = 0;
    smap[32*threadIdx.x+23] = 0;
    smap[32*threadIdx.x+24] = 0;
    smap[32*threadIdx.x+25] = 0;
    smap[32*threadIdx.x+26] = 0;
    smap[32*threadIdx.x+27] = 0;
    smap[32*threadIdx.x+28] = 0;
    smap[32*threadIdx.x+29] = 0;
    smap[32*threadIdx.x+30] = 0;
    smap[32*threadIdx.x+31] = 0;

#define SIEVE_1024_BIT(p, kdeltap) { \
    j = (i * 1024 + p-1 - kdeltap) / p; \
    k = kdeltap + j*p; \
    if ((k>>10) == i) \
      smap[32*threadIdx.x+((k>>5)&31)] |= 1<<(k&31); \
    }

#define SIEVE_1024_BIT_DEBUG(p, kdeltap) { \
    j = (i * 1024 + p-1 - kdeltap) / p; \
    k = kdeltap + j*p; \
    if ((k>>10) == i) \
      { \
      smap[32*threadIdx.x+((k>>5)&31)] |= 1<<(k&31); \
      if (k<10000 || k>3140000) \
        printf("p=%u, kdeltap=%u, i=%u, j=%u, k=%u=%8.8X, smap[.]=%8.8X\n", \
                 p, kdeltap, i, j, k, k, \
                 smap[32*threadIdx.x+((k>>5)&31)]); \
      } \
    }

    SIEVE_1024_BIT(1031, kdelta1031);
    SIEVE_1024_BIT(1033, kdelta1033);
    SIEVE_1024_BIT(1039, kdelta1039);
    SIEVE_1024_BIT(1049, kdelta1049);
    SIEVE_1024_BIT(1051, kdelta1051);
    SIEVE_1024_BIT(1061, kdelta1061);
    SIEVE_1024_BIT(1063, kdelta1063);
    SIEVE_1024_BIT(1069, kdelta1069);
    SIEVE_1024_BIT(1087, kdelta1087);
    SIEVE_1024_BIT(1091, kdelta1091);
    SIEVE_1024_BIT(1093, kdelta1093);
    SIEVE_1024_BIT(1097, kdelta1097);
    SIEVE_1024_BIT(1103, kdelta1103);
    SIEVE_1024_BIT(1109, kdelta1109);
    SIEVE_1024_BIT(1117, kdelta1117);
    SIEVE_1024_BIT(1123, kdelta1123);
    SIEVE_1024_BIT(1129, kdelta1129);
    SIEVE_1024_BIT(1151, kdelta1151);
    SIEVE_1024_BIT(1153, kdelta1153);
    SIEVE_1024_BIT(1163, kdelta1163);
    SIEVE_1024_BIT(1171, kdelta1171);
    SIEVE_1024_BIT(1181, kdelta1181);
    SIEVE_1024_BIT(1187, kdelta1187);
    SIEVE_1024_BIT(1193, kdelta1193);
    SIEVE_1024_BIT(1201, kdelta1201);
    SIEVE_1024_BIT(1213, kdelta1213);
    SIEVE_1024_BIT(1217, kdelta1217);
    SIEVE_1024_BIT(1223, kdelta1223);
    SIEVE_1024_BIT(1229, kdelta1229);
    SIEVE_1024_BIT(1231, kdelta1231);
    SIEVE_1024_BIT(1237, kdelta1237);
    SIEVE_1024_BIT(1249, kdelta1249);
    SIEVE_1024_BIT(1259, kdelta1259);
    SIEVE_1024_BIT(1277, kdelta1277);
    SIEVE_1024_BIT(1279, kdelta1279);
    SIEVE_1024_BIT(1283, kdelta1283);
    SIEVE_1024_BIT(1289, kdelta1289);
    SIEVE_1024_BIT(1291, kdelta1291);
    SIEVE_1024_BIT(1297, kdelta1297);
    SIEVE_1024_BIT(1301, kdelta1301);
    SIEVE_1024_BIT(1303, kdelta1303);
    SIEVE_1024_BIT(1307, kdelta1307);
    SIEVE_1024_BIT(1319, kdelta1319);
    SIEVE_1024_BIT(1321, kdelta1321);
    SIEVE_1024_BIT(1327, kdelta1327);
    SIEVE_1024_BIT(1361, kdelta1361);
    SIEVE_1024_BIT(1367, kdelta1367);
    SIEVE_1024_BIT(1373, kdelta1373);
    SIEVE_1024_BIT(1381, kdelta1381);
    SIEVE_1024_BIT(1399, kdelta1399);
    SIEVE_1024_BIT(1409, kdelta1409);
    SIEVE_1024_BIT(1423, kdelta1423);
    SIEVE_1024_BIT(1427, kdelta1427);
    SIEVE_1024_BIT(1429, kdelta1429);
    SIEVE_1024_BIT(1433, kdelta1433);
    SIEVE_1024_BIT(1439, kdelta1439);
    SIEVE_1024_BIT(1447, kdelta1447);
    SIEVE_1024_BIT(1451, kdelta1451);
    SIEVE_1024_BIT(1453, kdelta1453);
    SIEVE_1024_BIT(1459, kdelta1459);
    SIEVE_1024_BIT(1471, kdelta1471);
    SIEVE_1024_BIT(1481, kdelta1481);
    SIEVE_1024_BIT(1483, kdelta1483);
    SIEVE_1024_BIT(1487, kdelta1487);
    SIEVE_1024_BIT(1489, kdelta1489);
    SIEVE_1024_BIT(1493, kdelta1493);
    SIEVE_1024_BIT(1499, kdelta1499);
    SIEVE_1024_BIT(1511, kdelta1511);
    SIEVE_1024_BIT(1523, kdelta1523);
    SIEVE_1024_BIT(1531, kdelta1531);
    SIEVE_1024_BIT(1543, kdelta1543);
    SIEVE_1024_BIT(1549, kdelta1549);
    SIEVE_1024_BIT(1553, kdelta1553);
    SIEVE_1024_BIT(1559, kdelta1559);
    SIEVE_1024_BIT(1567, kdelta1567);
    SIEVE_1024_BIT(1571, kdelta1571);
    SIEVE_1024_BIT(1579, kdelta1579);
    SIEVE_1024_BIT(1583, kdelta1583);
    SIEVE_1024_BIT(1597, kdelta1597);
    SIEVE_1024_BIT(1601, kdelta1601);
    SIEVE_1024_BIT(1607, kdelta1607);
    SIEVE_1024_BIT(1609, kdelta1609);
    SIEVE_1024_BIT(1613, kdelta1613);
    SIEVE_1024_BIT(1619, kdelta1619);
    SIEVE_1024_BIT(1621, kdelta1621);
    SIEVE_1024_BIT(1627, kdelta1627);
    SIEVE_1024_BIT(1637, kdelta1637);
    SIEVE_1024_BIT(1657, kdelta1657);
    SIEVE_1024_BIT(1663, kdelta1663);
    SIEVE_1024_BIT(1667, kdelta1667);
    SIEVE_1024_BIT(1669, kdelta1669);
    SIEVE_1024_BIT(1693, kdelta1693);
    SIEVE_1024_BIT(1697, kdelta1697);
    SIEVE_1024_BIT(1699, kdelta1699);
    SIEVE_1024_BIT(1709, kdelta1709);
    SIEVE_1024_BIT(1721, kdelta1721);
    SIEVE_1024_BIT(1723, kdelta1723);
    SIEVE_1024_BIT(1733, kdelta1733);
    SIEVE_1024_BIT(1741, kdelta1741);
    SIEVE_1024_BIT(1747, kdelta1747);
    SIEVE_1024_BIT(1753, kdelta1753);
    SIEVE_1024_BIT(1759, kdelta1759);
    SIEVE_1024_BIT(1777, kdelta1777);
    SIEVE_1024_BIT(1783, kdelta1783);
    SIEVE_1024_BIT(1787, kdelta1787);
    SIEVE_1024_BIT(1789, kdelta1789);
    SIEVE_1024_BIT(1801, kdelta1801);
    SIEVE_1024_BIT(1811, kdelta1811);
    SIEVE_1024_BIT(1823, kdelta1823);
    SIEVE_1024_BIT(1831, kdelta1831);
    SIEVE_1024_BIT(1847, kdelta1847);
    SIEVE_1024_BIT(1861, kdelta1861);
    SIEVE_1024_BIT(1867, kdelta1867);
    SIEVE_1024_BIT(1871, kdelta1871);
    SIEVE_1024_BIT(1873, kdelta1873);
    SIEVE_1024_BIT(1877, kdelta1877);
    SIEVE_1024_BIT(1879, kdelta1879);
    SIEVE_1024_BIT(1889, kdelta1889);
    SIEVE_1024_BIT(1901, kdelta1901);
    SIEVE_1024_BIT(1907, kdelta1907);
    SIEVE_1024_BIT(1913, kdelta1913);
    SIEVE_1024_BIT(1931, kdelta1931);
    SIEVE_1024_BIT(1933, kdelta1933);
    SIEVE_1024_BIT(1949, kdelta1949);
    SIEVE_1024_BIT(1951, kdelta1951);
    SIEVE_1024_BIT(1973, kdelta1973);
    SIEVE_1024_BIT(1979, kdelta1979);
    SIEVE_1024_BIT(1987, kdelta1987);
    SIEVE_1024_BIT(1993, kdelta1993);
    SIEVE_1024_BIT(1997, kdelta1997);
    SIEVE_1024_BIT(1999, kdelta1999);
    SIEVE_1024_BIT(2003, kdelta2003);
    SIEVE_1024_BIT(2011, kdelta2011);
    SIEVE_1024_BIT(2017, kdelta2017);
    SIEVE_1024_BIT(2027, kdelta2027);
    SIEVE_1024_BIT(2029, kdelta2029);
    SIEVE_1024_BIT(2039, kdelta2039);

// Which strategy to write results to global memory?  atomicAnd isn't really necessary, as long
// as all kernel's prior to this one have finished before our kernel runs.
// Scrambling the copies from shared memory to global memory yields ~100% global load/store efficiency,
// but doesn't actually have a huge performance gain.
// Straightforward copy reports 6.25% global load/store efficiency.

#if 0                                            // NVVP reported 211.773 us, 4.6%/12.5% global load/store efficiency
    d_bitmapw[32*i   ] &= ~smap[32*threadIdx.x   ];
    d_bitmapw[32*i+ 1] &= ~smap[32*threadIdx.x+ 1];
    d_bitmapw[32*i+ 2] &= ~smap[32*threadIdx.x+ 2];
    d_bitmapw[32*i+ 3] &= ~smap[32*threadIdx.x+ 3];
    d_bitmapw[32*i+ 4] &= ~smap[32*threadIdx.x+ 4];
    d_bitmapw[32*i+ 5] &= ~smap[32*threadIdx.x+ 5];
    d_bitmapw[32*i+ 6] &= ~smap[32*threadIdx.x+ 6];
    d_bitmapw[32*i+ 7] &= ~smap[32*threadIdx.x+ 7];
    d_bitmapw[32*i+ 8] &= ~smap[32*threadIdx.x+ 8];
    d_bitmapw[32*i+ 9] &= ~smap[32*threadIdx.x+ 9];
    d_bitmapw[32*i+10] &= ~smap[32*threadIdx.x+10];
    d_bitmapw[32*i+11] &= ~smap[32*threadIdx.x+11];
    d_bitmapw[32*i+12] &= ~smap[32*threadIdx.x+12];
    d_bitmapw[32*i+13] &= ~smap[32*threadIdx.x+13];
    d_bitmapw[32*i+14] &= ~smap[32*threadIdx.x+14];
    d_bitmapw[32*i+15] &= ~smap[32*threadIdx.x+15];
    d_bitmapw[32*i+16] &= ~smap[32*threadIdx.x+16];
    d_bitmapw[32*i+17] &= ~smap[32*threadIdx.x+17];
    d_bitmapw[32*i+18] &= ~smap[32*threadIdx.x+18];
    d_bitmapw[32*i+19] &= ~smap[32*threadIdx.x+19];
    d_bitmapw[32*i+20] &= ~smap[32*threadIdx.x+20];
    d_bitmapw[32*i+21] &= ~smap[32*threadIdx.x+21];
    d_bitmapw[32*i+22] &= ~smap[32*threadIdx.x+22];
    d_bitmapw[32*i+23] &= ~smap[32*threadIdx.x+23];
    d_bitmapw[32*i+24] &= ~smap[32*threadIdx.x+24];
    d_bitmapw[32*i+25] &= ~smap[32*threadIdx.x+25];
    d_bitmapw[32*i+26] &= ~smap[32*threadIdx.x+26];
    d_bitmapw[32*i+27] &= ~smap[32*threadIdx.x+27];
    d_bitmapw[32*i+28] &= ~smap[32*threadIdx.x+28];
    d_bitmapw[32*i+29] &= ~smap[32*threadIdx.x+29];
    d_bitmapw[32*i+30] &= ~smap[32*threadIdx.x+30];
    d_bitmapw[32*i+31] &= ~smap[32*threadIdx.x+31];
    return;
#else                                            // NVVP reported        us,    % global load/store efficiency
    __syncthreads();    // Make sure everybody has stored their results
    d_bitmapw[32*i-32*threadIdx.x+              threadIdx.x] &= ~smap[              threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+   blockDim.x+threadIdx.x] &= ~smap[   blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+ 2*blockDim.x+threadIdx.x] &= ~smap[ 2*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+ 3*blockDim.x+threadIdx.x] &= ~smap[ 3*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+ 4*blockDim.x+threadIdx.x] &= ~smap[ 4*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+ 5*blockDim.x+threadIdx.x] &= ~smap[ 5*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+ 6*blockDim.x+threadIdx.x] &= ~smap[ 6*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+ 7*blockDim.x+threadIdx.x] &= ~smap[ 7*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+ 8*blockDim.x+threadIdx.x] &= ~smap[ 8*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+ 9*blockDim.x+threadIdx.x] &= ~smap[ 9*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+10*blockDim.x+threadIdx.x] &= ~smap[10*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+11*blockDim.x+threadIdx.x] &= ~smap[11*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+12*blockDim.x+threadIdx.x] &= ~smap[12*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+13*blockDim.x+threadIdx.x] &= ~smap[13*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+14*blockDim.x+threadIdx.x] &= ~smap[14*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+15*blockDim.x+threadIdx.x] &= ~smap[15*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+16*blockDim.x+threadIdx.x] &= ~smap[16*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+17*blockDim.x+threadIdx.x] &= ~smap[17*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+18*blockDim.x+threadIdx.x] &= ~smap[18*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+19*blockDim.x+threadIdx.x] &= ~smap[19*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+20*blockDim.x+threadIdx.x] &= ~smap[20*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+21*blockDim.x+threadIdx.x] &= ~smap[21*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+22*blockDim.x+threadIdx.x] &= ~smap[22*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+23*blockDim.x+threadIdx.x] &= ~smap[23*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+24*blockDim.x+threadIdx.x] &= ~smap[24*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+25*blockDim.x+threadIdx.x] &= ~smap[25*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+26*blockDim.x+threadIdx.x] &= ~smap[26*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+27*blockDim.x+threadIdx.x] &= ~smap[27*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+28*blockDim.x+threadIdx.x] &= ~smap[28*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+29*blockDim.x+threadIdx.x] &= ~smap[29*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+30*blockDim.x+threadIdx.x] &= ~smap[30*blockDim.x+threadIdx.x];
    d_bitmapw[32*i-32*threadIdx.x+31*blockDim.x+threadIdx.x] &= ~smap[31*blockDim.x+threadIdx.x];
    return;
#endif

  }
}


__global__ void __launch_bounds__(256, 4) rcv_sieve_primes(
        unsigned int tidoffseta,/* Offset from tid to first tid's element in tree */
        unsigned int tidoffsetz,/* Offset from tid to just past last tid's element in tree */
        unsigned int *d_plist,  /* pointer to list of primes */
        unsigned int pcount,    /* number of elements in list */
        int96        kstart,    /* lowest k-value in upcoming sieve */
        unsigned int kcount,    /* number of bits in upcoming sieve */
        unsigned int *d_bdelta, /* pointer to starting k-values per prime */
        unsigned int *d_kncount,/* pointer to count of k-values per prime */
        unsigned int *d_ktree,  /* pointer to tree of count of k-values per prime */
        unsigned int *d_bitmapw /* bitmap for the sieve, 32-bit words */
        )
{
  unsigned int i;
  unsigned int j;
  unsigned int k;
  unsigned int l;

  unsigned int pcountpow2;      /* next power of 2 >= pcount */

  for (pcountpow2=1; pcountpow2 < pcount; pcountpow2 = pcountpow2+pcountpow2)
    ;

  unsigned int ndeeper;         /* number of elements 1 level deeper in tree */
  ndeeper = pcount+pcount - pcountpow2;

  i = blockDim.x * blockIdx.x + threadIdx.x;
  i += tidoffseta;              /* Work the tree starting at this offset */

  if ((i < d_ktree[1]) && (i < tidoffsetz))  // Root of tree contains total threads.
  {
    j = 1;      // Root of tree.  Contains no useful data
    while (j < pcount)
    {
      j += j;
                // j is even.  Node we seek must be j or j+1 (or descendents)
      if (i >= d_ktree[j])
      {
        i -= d_ktree[j];
        j += 1;
      }
      //if (i >= d_ktree[j])
      //  ***ERROR***   // Cannot occur
    }

    if (j < pcountpow2)
      l = j + ndeeper - pcount;
    else
      l = j - pcountpow2;


                        // l now indexes the prime we are sieving
                        // i contains the instance of sieving with this prime

    k = d_bdelta[l] + i*d_plist[l];  // Get the bit (relative to the entire sieve) we are to clear


    // Following two methods both work.  For repeatability, we prefer atomicAnd.
    // Not entirely certain about undocumented CUDA behavior, but we don't
    // believe atomicAnd slows us down.  CUDA generally claims stalls only
    // occur when you use the result of an instruction.  We are careful to
    // *avoid* use of the atomicAnd result.  Only stall should occur if another
    // thread coincidentally needs to reference the same word before hardware
    // (memory controller?) completes the operation.
    // BTW, all CUDA since Compute Capability 1.1 support atomicAnd.
#if 1
    if (k < kcount)
      if (d_bitmapw[k>>5] & 0x00000001<<(k&31)) // Is our bit still on?
        atomicAnd(&d_bitmapw[k>>5], ~(0x00000001<<(k&31))); // If yes, turn it off.  If no, save atomic op

                        // Note:  Avoid unlocked word-size AND operation.
                        // The probability of a collission is 4 times as
                        // large, compared with byte-size AND operation.
                        // So, we'll run more excess divisibility tests.
#else    // This section uses byte-wide bitmap
    if (k < kcount)
      if (((unsigned char *)d_bitmapw)[k>>3] & 0x01<<(k&7)) // Is our bit still on?
        ((unsigned char *)d_bitmapw)[k>>3] &= ~(0x01<<(k&7));  // If yes, turn it off.  If no, save a memory write.

                        // Note:  The above is not locked.  Other
                        // concurrent threads may be changing the same byte.
                        // If we lose an occasional clear, it's no big deal.
                        // We'll just run a few extra divisibility tests.
                        // However, please check any code or resources that
                        // depend on the law of large numbers.
#endif
  }

}



// This kernel simply initializes a small array used as a set of atomic
// indexes into a large array which will collect our linearized candidate.
__global__ void rcv_reset_atomic_indexes(
        unsigned int width,          /* width of atomic index array */
        unsigned int *d_xaindexes)   /* atomic index into array */
{
  unsigned int i;
  
  // One thread, per index should be launched for this kernel, please.)
  // Note:  As implemented in this program, the array is 1 column wide
  i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < width)                /* Excess threads do not participate */
    d_xaindexes[i] = 0;         /* each atomic index is initialized to zero */
}


// This kernel converts a completed set of sieve bits to an array of k-values
// for the trial factor component.
__global__ void __launch_bounds__(256, 4) rcv_linearize_sieve(
        int96        kstart,    /* lowest k-value in current sieve */
        unsigned int kcount,    /* number of bits in current sieve */
        unsigned int *d_bitmapw,/* bitmap for the sieve, 32-bit words */
        unsigned int *d_karray, /* linear array of k-values */
        unsigned int kasize,    /* number of spots in output array */
        unsigned int *d_kaindex /* atomic allocation index into karray */
        )
{
  unsigned int i;
  __shared__ volatile unsigned short bitcount[256];  // Each thread of our block puts bit-counts here
#define LINEARIZE_SMEM_BOUND (32*256*37/100)         // Space for 256 Threads Per Blocks at 37% sieve rate.
                                                     // at MAX_SIEVE_PRIMES =  304, 35.35% survival rate.
                                                     // at MAX_SIEVE_PRIMES =  559, 32.42% survival rate.
                                                     // at MAX_SIEVE_PRIMES = 2500, 26.96% survival rate.
  __shared__ unsigned short smem[LINEARIZE_SMEM_BOUND];
  __shared__ unsigned int   kaix;                    // Index into caller's array for our candidates

  // One thread, per 32-bit word of bitmap should be launched for this kernel, please.)
  i = blockDim.x * blockIdx.x + threadIdx.x;

#if 0
  if (i==0)
    printf("Reached rcv_linearize_sieve\n");
#endif

  if (i < (kcount>>5))	// Excess threads don't participate
  {

    // Count number of bits in use
    {
      unsigned int t;
      t = d_bitmapw[i];
      t = (t&0x55555555) + ((t>> 1)&0x55555555);  // Generate sixteen 2-bit sums
      t = (t&0x33333333) + ((t>> 2)&0x33333333);  // Generate eight 3-bit sums
      t = (t&0x07070707) + ((t>> 4)&0x07070707);  // Generate four 4-bit sums
      t = (t&0x000f000f) + ((t>> 8)&0x000f000f);  // Generate two 5-bit sums
      t = (t&0x0000001f) + ((t>>16)&0x0000001f);  // Generate one 6-bit sum

      bitcount[threadIdx.x] = t;	// Tell everybody how much space my thread needs
    }

    __syncthreads();    // Synchronization required!

    // CAUTION:  Following requires 256 threads per block

    // First five tallies remain within one warp.  Should be in lock-step.
    if (!(i&1))       // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
      bitcount[threadIdx.x] += bitcount[threadIdx.x + 1];

    if (!(i&2))        // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
      bitcount[threadIdx.x] += bitcount[threadIdx.x + 2 & ~1];

    if (!(i&4))        // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
      bitcount[threadIdx.x] += bitcount[threadIdx.x + 4 & ~3];

    if (!(i&8))        // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
      bitcount[threadIdx.x] += bitcount[threadIdx.x + 8 & ~7];

    if (!(i&16))       // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
      bitcount[threadIdx.x] += bitcount[threadIdx.x + 16 & ~15];

    // Further tallies are across warps.  Must synchronize
    __syncthreads();   // Synchronization required!
    if (!(i&32))       // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
      bitcount[threadIdx.x] += bitcount[threadIdx.x + 32 & ~31];

    __syncthreads();   // Synchronization required!
    if (!(i&64))       // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
      bitcount[threadIdx.x] += bitcount[threadIdx.x + 64 & ~63];

    __syncthreads();   // Synchronization required!
    if (!(i&128))       // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
      bitcount[threadIdx.x] += bitcount[threadIdx.x + 128 & ~127];

    // At this point, bitcount[...] should contain the total number of bits for the indexed
    // thread plus all high-numbered threads.  I.e., bitcount[0] is the total count.


    // Atomically allocate space in final array for list of k-values
    // One thread allocates space for entire thread block.  Should minimize contention.
    {
      if (threadIdx.x == 0)     // First thread of the thread block?
      {
        kaix = atomicAdd(d_kaindex, bitcount[0]);  // Obtain space in final array for our k-values
        if (kaix + bitcount[0] >= kasize)
          asm("{trap; \n\t}");    // Trap if allocation exceeds array size
        while (kaix + bitcount[0] >= kasize)
          ;    // If trap fails, cause a kernel timeout.
      }
      __syncthreads();
    }

    if (bitcount[0] > LINEARIZE_SMEM_BOUND)    // Will we overshoot our shared memory?
    {
      // This should *rarely* happen -- as when tweaking the code.
      // We revert to a simpler, slower method that doesn't use shared memory
      unsigned int k;
      unsigned int bitmapw;
      unsigned int mykaix;        // karray index for this thread

      k = 32*i;                   // This thread's starting k-value
      bitmapw = d_bitmapw[i];     // 32-bit word containing this thread's bits
      mykaix = kaix + bitcount[0] - bitcount[threadIdx.x];  // Storage index to hold this thread's k-values

      for (int j=0; j<32; j+=1)
      {
        if (bitmapw&1)
          d_karray[mykaix++] = k+j;  // Store current k-value to global memory array
        bitmapw >>= 1;               // Shift sieve bits over
      }

      return;     // No fanfare.  But skip the fast method, which follows
    }


    // Work through one 32-bit word
    {
      unsigned int k;
      unsigned int bitmapw;
      unsigned int six;

      __syncthreads();
      // k = 32*i;                 // This thread's starting k-value
      k = 32*threadIdx.x;          // This thread's starting k-value
      bitmapw = d_bitmapw[i];   // 32-bit word containing this thread's bits
      six = bitcount[0] - bitcount[threadIdx.x];  // Storage index to hold this thread's k-values

      // Unroll this loop, please
      for (int j=0; j<32; j+=1)
      {
        if (bitmapw&1)
          smem[six++] = k+j;   // Store current k-value to shared memory array
        bitmapw >>= 1;         // Shift sieve bits over
      }

    }

    // Here, all warps of our thread block have placed their candidates in shared memory.

    // The role of threads changes to maximize performance in copying
    // those candidates to global memory
    __syncthreads();              // The smem we read probably is not the smem we wrote!

    {
      unsigned int mykaix;        // karray index for this thread
               int six;           // smem index for this thread

      mykaix = (kaix/32)*32 + threadIdx.x;    // Align our threads to karray alignment
                                              // (thread 32n+5 will access word 32m+5)
      six = mykaix - kaix;    // six==0 where mykaix==kaix

      // First set of transfers may not involve lowest thread IDs
      if ((six >= 0) && (six < bitcount[0]))
      {
        d_karray[mykaix] = smem[six]+32*(i-threadIdx.x);    // Copy a candidate
      }

      mykaix += blockDim.x;
      six += blockDim.x;

      // Copy any additional candidates
      for ( ; six < bitcount[0]; six += blockDim.x, mykaix += blockDim.x)
      {
        d_karray[mykaix] = smem[six]+32*(i-threadIdx.x);    // Copy a candidate
      }

    }

  }
}
