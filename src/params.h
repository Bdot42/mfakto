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


/* EXTENDED_SELFTEST will add about 30k additional tests to the -st and -st2 tests */
#define EXTENDED_SELFTEST


/******************
** DEBUG options **
******************/

/* print detailed timing information of the CPU sieve*/
//#define VERBOSE_SIEVE_TIMING


/* do some checks on the mod/div routines */
//#define CHECKS_MODBASECASE


/* print stream, kernel schedule and h_ktab usage */
//#define DEBUG_STREAM_SCHEDULE

/* perform a sanity check on the h_ktab usage */
//#define DEBUG_STREAM_SCHEDULE_CHECK


/* disable sieve code to measure raw GPU performance
   if kernel invocation fails with "Error -4: Enqueuing kernel(clEnqueueNDRangeKernel)",
   then the GPUSieveProcessSize is too large - set it to 8 for this mode.*/
//#define RAW_GPU_BENCH


/* issue lots of additional trace output from the C-part of the program
   (see mfakto_kernels.cl - TRACE_KERNEL and TRACE_TID for how to trace the
   kernel execution, kernel trace does not require a rebuild) */
//#define DETAILED_INFO


/* enable the OpenCL built-in performance measurement. This will print the
   pure times needed to copy the data over, and to test the FC's of the chunk
   (pure run time per kernel invokation) - best measure to compare changes in
   the kernels, or drivers. */
//#define CL_PERFORMANCE_INFO


/* Tell the OpenCL compiler to create debuggable code for the Kernels */
//#define CL_DEBUG

/* in order to more efficiently trace/debug the kernels, this define makes
   the factor to be found the first candidate during the selftest, so that
   debugging/tracing thread 0 of the first block should yield a factor found result
   (this works only for the CPU sieve as the GPU sieve reverts the order) */
//#define DEBUG_FACTOR_FIRST


/******************************************************************************
*******************************************************************************
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*******************************************************************************
******************************************************************************/

#define SHORT_MFAKTO_VERSION "0.16-beta.1" /* DO NOT CHANGE! */

#ifdef _MSC_VER
  #define MFAKTO_VERSION "mfakto " SHORT_MFAKTO_VERSION "-Win" /* DO NOT CHANGE! */
#elif defined __MINGW32__
#define MFAKTO_VERSION "mfakto " SHORT_MFAKTO_VERSION "-MGW" /* DO NOT CHANGE! */
#else
  #define MFAKTO_VERSION "mfakto " SHORT_MFAKTO_VERSION /* DO NOT CHANGE! */
#endif


// MORE_CLASSES and SIEVE_SIZE are used for CPU-sieving only. GPU-sieving uses a config setting
/*
If MORE_CLASSES is defined then the TF process is split into 4620
(4 * 3*5*7*11) classes. Otherwise it will be split into 420 (4 * 3*5*7)
classes. With 4620 the siever runs a bit more efficient at the cost of 10 times
more sieve initializations. This will allow to increase SIEVE_PRIMES a little
bit further. This starts to become useful on my system for e.g. TF M66xxxxxx
from 2^66 to 2^67. The OpenCL version, mfakto, requires MORE_CLASSES be defined.
*/

#define MORE_CLASSES



/*
THREADS_PER_BLOCK is not needed for OpenCL - it dynamically uses the device's maximum
All GPU-sieve kernels also have 256 threads per block.
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


#ifdef CL_PERFORMANCE_INFO
#define QUEUE commandQueuePrf
#else
#define QUEUE commandQueue
#endif
/*
The number of streams used by mfakto. No distinction between CPU and GPU streams anymore
The actual configuration is done in mfakto.ini. This ini-file contains
a small description, too
The following lines define the min, default and max value.
*/

#define NUM_STREAMS_MIN     1 /* DO NOT CHANGE! */
#define NUM_STREAMS_DEFAULT 3 /* DO NOT CHANGE! */
#define NUM_STREAMS_MAX     10 /* DO NOT CHANGE! */

// MORE_CLASSES and SIEVE_SIZE are used for CPU-sieving only. GPU-sieving uses a config setting
/* set NUM_CLASSES and SIEVE_SIZE depending on MORE_CLASSES and SIEVE_SIZE_LIMIT
   MORE_CLASSES is required for mfakto's CPU sieve */
#ifdef MORE_CLASSES
  #define NUM_CLASSES 4620 /* 2 * 2 * 3 * 5 * 7 * 11 */
#ifdef SIEVE_SIZE_LIMIT
  #define SIEVE_SIZE ((SIEVE_SIZE_LIMIT<<13) - (SIEVE_SIZE_LIMIT<<13) % (13*17*19*23))
#endif
#else
# error "mfakto requires MORE_CLASSES be defined."
#endif


/*
GPU_SIEVE_PRIMES defines how far we sieve the factor candidates on the GPU.
The first <GPU_SIEVE_PRIMES> primes are sieved.

GPU_SIEVE_SIZE defines how big of a GPU sieve we use (in M bits).

GPU_SIEVE_PROCESS_SIZE defines how far many bits of the sieve each TF block processes (in K bits).
Larger values may lead to less wasted cycles by reducing the number of times all threads in a warp
are not TFing a candidate.  However, more shared memory is used which may reduce occupancy.
Smaller values should lead to a more responsive system (each kernel takes less time to execute).

The actual configuration is done in mfaktc.ini.
The following lines define the min, default and max value.
*/

#define GPU_SIEVE_PRIMES_MIN                54 /* GPU sieving code can work (inefficiently) with very small numbers */
#define GPU_SIEVE_PRIMES_DEFAULT         82486 /* Default is to sieve primes up to about 1.05M */
#define GPU_SIEVE_PRIMES_MAX           1075766 /* Primes to 16,742,237.  GPU sieve should be able to handle up to 16M. */

#define GPU_SIEVE_SIZE_MIN                   4 /* A 4M bit sieve seems like a reasonable minimum */
#define GPU_SIEVE_SIZE_DEFAULT              64 /* Default is a 64M bit sieve */
#define GPU_SIEVE_SIZE_MAX                 128 /* We've only tested up to 128M bits.  The GPU sieve code may be able to go higher. */

#define GPU_SIEVE_PROCESS_SIZE_MIN           8 /* Processing 8K bits in each block is minimum (256 threads * 1 word of 32 bits) */
#define GPU_SIEVE_PROCESS_SIZE_DEFAULT      16 /* Default is processing 16K bits */
#define GPU_SIEVE_PROCESS_SIZE_MAX          32 /* Upper limit is 64K, since we store k values as "short". Shared memory requirements limit usable values */

