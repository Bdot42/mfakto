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

/* 60bit (4x 15bit) integer
D=d0 + d1*(2^15) + d2*(2^30) ... */
typedef struct
{
  cl_uint d0,d1,d2,d3;
}int60;

/* 120bit (8x 15bit) integer
D=d0 + d1*(2^15) + d2*(2^30) + ... */
typedef struct
{
  cl_uint d0,d1,d2,d3,d4,d5,d6,d7;
}int120;

/* 75bit (5x 15bit) integer
D=d0 + d1*(2^15) + d2*(2^30) ... */
typedef struct
{
  cl_uint d0,d1,d2,d3,d4;
}int75;

/* 150bit (10x 15bit) integer
D=d0 + d1*(2^15) + d2*(2^30) + ... */
typedef struct
{
  cl_uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150;

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct
{
  cl_uint d0,d1,d2;
}int72;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct
{
  cl_uint d0,d1,d2,d3,d4,d5;
}int144;


/* int72 and int96 are the same but this way the compiler warns
when an int96 is passed to a function designed to handle 72 bit int.
The applies to int144 and int192, too. */

/* 96bit (3x 32bit) integer
D= d0 + d1*(2^32) + d2*(2^64) */
typedef struct
{
  cl_uint d0,d1,d2;
}int96;

/* 192bit (6x 32bit) integer
D=d0 + d1*(2^32) + d2*(2^64) + ... */
typedef struct
{
  cl_uint d0,d1,d2,d3,d4,d5;
}int192;

enum STREAM_STATUS
{
  UNUSED,
  PREPARED,
  RUNNING,
  DONE
};

enum MODES
{
  MODE_NORMAL,
  MODE_SELFTEST_SHORT,
  MODE_SELFTEST_HALF,
  MODE_SELFTEST_FULL,
  MODE_PERFTEST
};

enum GPUKernels
{
  AUTOSELECT_KERNEL = 0,
  _TEST_MOD_,
  _95BIT_64_OpenCL,
  _71BIT_MUL24,
  _63BIT_MUL24,
  BARRETT72_MUL24,
  BARRETT79_MUL32,
  BARRETT92_MUL32,
  BARRETT58_MUL15,
  BARRETT73_MUL15,
  UNKNOWN_KERNEL, /* what comes after this one will not be loaded automatically*/
  _64BIT_64_OpenCL,
  BARRETT92_64_OpenCL,
  CL_SIEVE_INIT,
  CL_SIEVE,
  _95BIT_MUL32  /* not yet there */
};

enum PRINT_PARM // cCpgtenrswWdTUHulM .. CcpgtenrswWdTUHMlu
{
  CLASS_ID,       //  %C - class ID (n/4620)
  CLASS_NUM,      //  %c - class number (n/960)
  PCT_COMPLETE,   //  %p - percent complete (%)
  GHZ,            //  %g - GHz-days/day (GHz)
  TIME_PER_CLASS, //  %t - time per class (s)
  ETA,            //  %e - eta (d/h/m/s)
  CANDIDATES,     //  %n - number of candidates (M/G)
  RATE,           //  %r - rate (M/s)
  SIEVE_PRIMES,   //  %s - SievePrimes
  CPU_WAIT_TIME,  //  %w - CPU wait time for GPU (us)
  CPU_WAIT_PCT,   //  %W - CPU wait % (%)
  DATE_SHORT,     //  %d - date (Mon nn)
  TIME_SHORT,     //  %T - time (HH:MM)
  USER,           //  %U - username (as configured)
  HOST,           //  %H - hostname (as configured), ComputerID
  EXP,            //  %M - the exponent being worked on
  LOWER_LIMIT,    //  %l - the lower bit-limit
  UPPER_LIMIT,    //  %u - the upper bit-limit

  NUM_PRINT_PARM  //
};

typedef struct
{
  cl_uint  pos;  /* the position where this parameter shall appear, 0=unused */
  char     out[16]; /* fixed size output string */
  char     parm; /* the parameter, one of cCpgtenrswWdTuhf */
} print_parameter;

typedef struct
{
  cl_event copy_events[NUM_STREAMS_MAX];
  cl_event exec_events[NUM_STREAMS_MAX];
  cl_uint *h_ktab[NUM_STREAMS_MAX];
  cl_mem   d_ktab[NUM_STREAMS_MAX];
  cl_uint *h_RES;
  cl_mem   d_RES;
  enum STREAM_STATUS stream_status[NUM_STREAMS_MAX];
  enum GPUKernels    preferredKernel;
  /* for GPU sieving: */
  cl_uint *h_primes;
  cl_mem   d_primes;
  cl_uint *h_k_mult;
  cl_mem   d_k_mult;
  cl_uint *h_savestate;
  cl_mem   d_savestate;


  cl_uint sieve_primes, sieve_primes_adjust, sieve_primes_max, sieve_primes_max_global, sieve_gpu, sieve_size;
  cl_uint num_streams;
  
  enum MODES mode;
  cl_uint checkpoints, checkpointdelay, stages, stopafterfactor;
  cl_uint threads_per_grid_max, threads_per_grid;

#ifdef CHECKS_MODBASECASE
  cl_mem   d_modbasecase_debug;
  cl_uint *h_modbasecase_debug;
#endif  

  cl_uint vectorsize;
  cl_uint printmode;
  cl_uint class_counter;		/* needed for ETA calculation */
  cl_uint allowsleep;
  cl_uint small_exp;
  cl_uint quit; 
  char * p_ptr[20];         /* pointers to the formatted output string arrays, in the order they are being used, allow 20 of them */
  print_parameter p_par[NUM_PRINT_PARM]; /* to hold the position and strings of each possible parameter */
  char print_line[512];
  char V5UserID[51];
  char ComputerID[51];
  char workfile[51];		/* allow filenames up to 50 chars... */
  char inifile[51];		/* allow filenames up to 50 chars... */
  char resultsfile[51];
}mystuff_t;			/* FIXME: proper name needed */

typedef struct
{
    char d_name[128], d_ver[128], v_name[128], dr_version[128], exts[2048];
    cl_ulong gl_cache, gl_mem, l_mem;
    cl_uint max_clock, units, w_dim;
    size_t wg_size, wi_sizes[10], maxThreadsPerBlock, maxThreadsPerGrid;
} OpenCL_deviceinfo_t;

typedef struct _kernel_info
{
  enum GPUKernels kernel_id;
  char            kernelname[32];
  cl_uint         bit_min, bit_max;
  cl_kernel       kernel;
} kernel_info_t;


#define RET_ERROR 1000000001
#define RET_QUIT  1000000002

