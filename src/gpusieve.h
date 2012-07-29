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


typedef struct                  // Context information for each stream's work
{
  int stream;          // Space for cuda Stream Handles
  int linearized_event; // Event placed after linearization function
  unsigned int   exp;           // Mersenne exponent
  unsigned int   kclass;        // Class number
  unsigned int   csieve_bits;   // Number of bits in current sieve
           int96 ckstart;       // Lowest k-value for current sieve
           int96 cbstart;       // Lowest b-value for current sieve
           int96 cbend;         // Highest b-value for current sieve
} sievecontext;

