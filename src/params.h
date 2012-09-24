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
*/


// In Catalyst 11.x, x>=10 and 12.y, not all parameters were passed to the kernel
// -> replace user-defined struct with uint8
#define WA_FOR_CATALYST11_10_BUG


/*
SIEVE_SIZE_LIMIT is the maximum segment size of the sieve.
too small => to much overhead
too big => doesn't fit into (fast) CPU-caches
The size given here is in kiB (1024 bytes). A good starting point is the size
of your CPUs L1-Data cache.
This is just the upper LIMIT of the SIEVE_SIZE, the actual sieve size depends
on some other factors as well, but you don't have to worry about.

If this #define is not set, an ini-file key SieveSizeLimit will be evaluated to
set it. This allows for adjusting the SieveSize, but may be up to 3% slower
than an equal SIEVE_SIZE_LIMIT #define.

*/

#define SIEVE_SIZE_LIMIT 36


/* EXTENDED_SELFTEST will add about 30k additional tests to the -st2 test */
// #define EXTENDED_SELFTEST


/******************
** DEBUG options **
******************/

/* print some more timing information */
//#define VERBOSE_TIMING


/* print detailed timing information of the sieve*/
//#define VERBOSE_SIEVE_TIMING


/* do some checks on the mod/div routines */
//#define CHECKS_MODBASECASE


/* print stream, kernel schedule and h_ktab usage */
//#define DEBUG_STREAM_SCHEDULE

/* perform a sanity check on the h_ktab usage */
//#define DEBUG_STREAM_SCHEDULE_CHECK


/* disable sieve code to measure raw GPU performance */
//#define RAW_GPU_BENCH


/* issue lots of additional trace output from the C-part of the program
   (see mfakto_kernels.cl - TRACE_KERNEL and TRACE_TID for how to trace the
   kernel execution, kernel trace does not require a rebuild) */
//#define DETAILED_INFO


/* enable the OpenCL built-in performance measurement. This will print the
   pure times needed to copy the data over, and to test the FC's of the chunk
   (pure run time per kernel invokation) - best measure to compare changes in
   the kernels, or drivers. */
#define CL_PERFORMANCE_INFO


/* Tell the OpenCL compiler to create debuggable code for the Kernels */
//#define CL_DEBUG

/******************************************************************************
*******************************************************************************
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*******************************************************************************
******************************************************************************/

#ifndef _MSC_VER
  #define MFAKTO_VERSION "mfakto 0.13pre1" /* DO NOT CHANGE! */
#else
  #define MFAKTO_VERSION "mfakto 0.13pre1-Win" /* DO NOT CHANGE! */
#endif


/*
If MORE_CLASSES is defined then the TF process is split into 4620
(4 * 3*5*7*11) classes. Otherwise it will be split into 420 (4 * 3*5*7)
classes. With 4620 the siever runs a bit more efficent at the cost of 10 times
more sieve initializations. This will allow to increase SIEVE_PRIMES a little
bit further. This starts to become useful on my system for e.g. TF M66xxxxxx
from 2^66 to 2^67. The OpenCL version, mfakto, requires MORE_CLASSES be defined.
*/

#define MORE_CLASSES



/*
THREADS_PER_BLOCK is not needed for OpenCL - it dynamically uses the device's maximum
*/

//#define THREADS_PER_BLOCK 256



/*
SIEVE_PRIMES defines how far we sieve the factor candidates.
The first <SIEVE_PRIMES> odd primes are sieved.
The optimal value depends greatly on the speed of the CPU (one core) and the
speed of the CPU.
The actual configuration is done in mfakto.ini.
The following lines define the min, default and max value.
*/

#define SIEVE_PRIMES_MIN        256 /* DO NOT CHANGE! */
#define SIEVE_PRIMES_DEFAULT  25000 /* DO NOT CHANGE! */
#define SIEVE_PRIMES_MAX    1000000 /* DO NOT CHANGE! */

/*
primes[1000]=7927      k_tab(2M)=6976698
primes[3511]=32749     largest prime in 15 bits, k_tab(2M)=8073831
primes[6541]=65521     largest prime in 16 bits, k_tab(2M)=8610744
primes[25000]=287137   k_tab(2M)=9765293
primes[100000]=1299721 k_tab(2M)=10940976
primes[200000]=2750161 (22 bits) largest prime sieved here, k_tab(2M)=11506233
primes[300000]=4256249 (23) k_tab(2M) = 11827037
primes[500000]=7368791 (23) k_tab(2M) = 12207668
primes[1000000]=15485867 (24) k_tab(2M) = 12733916
*/


/* the first SIEVE_SPLIT primes have a special code in sieve.c. This defines
when the siever switches between those two code variants.
Needs to be less than SIEVE_PRIMES_MIN.*/

#define SIEVE_SPLIT 250 /* DO NOT CHANGE! */



/*
The number of streams used by mfakto. No distinction between CPU and GPU streams anymore
The actual configuration is done in mfakto.ini. This ini-file contains
a small description, too
The following lines define the min, default and max value.
*/

#define NUM_STREAMS_MIN     1 /* DO NOT CHANGE! */
#define NUM_STREAMS_DEFAULT 3 /* DO NOT CHANGE! */
#define NUM_STREAMS_MAX     10 /* DO NOT CHANGE! */


/* set NUM_CLASSES and SIEVE_SIZE depending on MORE_CLASSES and SIEVE_SIZE_LIMIT
   MORE_CLASSES is required for mfakto now */
#ifdef MORE_CLASSES
  #define NUM_CLASSES 4620 /* 2 * 2 * 3 * 5 * 7 * 11 */
#ifdef SIEVE_SIZE_LIMIT
  #define SIEVE_SIZE ((SIEVE_SIZE_LIMIT<<13) - (SIEVE_SIZE_LIMIT<<13) % (13*17*19*23))
#endif
#else
# error "mfakto requires MORE_CLASSES be defined."
#endif
