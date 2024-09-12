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

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#ifndef __MY_TYPES_H
#define __MY_TYPES_H
#include <stdio.h>
#include "params.h"
#if defined __APPLE__
  #include "OpenCL/cl.h"
#else
  #include "CL/cl.h"
#endif

/* 60bit (4x 15bit) integer
D=d0 + d1*(2^15) + d2*(2^30) ... */
typedef struct _int60
{
  cl_uint d0,d1,d2,d3;
} int60;

/* 120bit (8x 15bit) integer
D=d0 + d1*(2^15) + d2*(2^30) + ... */
typedef struct _int120
{
  cl_uint d0,d1,d2,d3,d4,d5,d6,d7;
} int120;

/* 75bit (5x 15bit) integer
D=d0 + d1*(2^15) + d2*(2^30) ... */
typedef struct _int75
{
  cl_uint d0,d1,d2,d3,d4;
} int75;

/* 150bit (10x 15bit) integer
D=d0 + d1*(2^15) + d2*(2^30) + ... */
typedef struct _int150
{
  cl_uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
} int150;

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct _int72
{
  cl_uint d0,d1,d2;
} int72;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct _int144
{
  cl_uint d0,d1,d2,d3,d4,d5;
} int144;


/* int72 and int96 are the same but this way the compiler warns
when an int96 is passed to a function designed to handle 72 bit int.
The applies to int144 and int192, too. */

/* 96bit (3x 32bit) integer
D= d0 + d1*(2^32) + d2*(2^64) */
typedef struct _int96
{
  cl_uint d0,d1,d2;
} int96;

/* 192bit (6x 32bit) integer
D=d0 + d1*(2^32) + d2*(2^64) + ... */
typedef struct _int192
{
  cl_uint d0,d1,d2,d3,d4,d5;
} int192;

enum STREAM_STATUS
{
  UNUSED,
  PREPARED,
  RUNNING,
  DONE
};

enum MODES
{
  MODE_PERFTEST,
  MODE_NORMAL,
  MODE_SELFTEST_SHORT,
  MODE_SELFTEST_QUICK,
  MODE_SELFTEST_FULL
};

enum EXIT_VALUES
{
  ERR_OK = 0,
  ERR_PARAM,
  ERR_INIT,
  ERR_MEM,
  ERR_SELFTEST,
  ERR_RUNTIME
};

enum GPUKernels
{
  AUTOSELECT_KERNEL = 0,
  _TEST_MOD_,
  _71BIT_MUL24,
  _63BIT_MUL24,
  BARRETT79_MUL32,
  BARRETT77_MUL32,
  BARRETT76_MUL32,
  BARRETT92_MUL32,
  BARRETT88_MUL32,
  BARRETT87_MUL32,
  BARRETT73_MUL15,
  BARRETT69_MUL15,
  BARRETT70_MUL15,
  BARRETT71_MUL15,
  BARRETT88_MUL15,
  BARRETT83_MUL15,
  BARRETT82_MUL15,
  BARRETT74_MUL15,
  MG62,
  MG88,
  UNKNOWN_KERNEL, /* what comes after this one will not be loaded automatically*/
  _64BIT_64_OpenCL,
  BARRETT92_64_OpenCL,
  CL_CALC_BIT_TO_CLEAR,  // loaded if GPU sieving enabled
  CL_CALC_MOD_INV,       // loaded if GPU sieving enabled
  CL_SIEVE,              // loaded if GPU sieving enabled
  BARRETT79_MUL32_GS,
  BARRETT77_MUL32_GS,
  BARRETT76_MUL32_GS,
  BARRETT92_MUL32_GS,
  BARRETT88_MUL32_GS,
  BARRETT87_MUL32_GS,
  BARRETT73_MUL15_GS,
  BARRETT69_MUL15_GS,
  BARRETT70_MUL15_GS,
  BARRETT71_MUL15_GS,
  BARRETT88_MUL15_GS,
  BARRETT83_MUL15_GS,
  BARRETT82_MUL15_GS,
  BARRETT74_MUL15_GS,
  UNKNOWN_GS_KERNEL  /* not yet there */
};

typedef enum GPUKernels GPUKernels;

enum GPU_types
{
  GPU_AUTO,
  GPU_VLIW4,
  GPU_VLIW5,
  GPU_GCN,   // low and mid-level GCN with slow DP 1:16
  GPU_GCN2,  // high-end GCN with faster DP 1:4
  GPU_GCN3,  // newer GCN with improved int32 operations (e.g. R290)
  GPU_GCN4,
  GPU_GCN5,
  GPU_GCNF,  // R VII
  GPU_RDNA,
  GPU_APU,
  GPU_CPU,
  GPU_NVIDIA,
  GPU_INTEL,
  GPU_UNKNOWN   // must be the last one
};

typedef struct _GPU_type
{
  enum GPU_types gpu_type;
  unsigned int   CE_per_multiprocessor;
  char           gpu_name[8];
} GPU_type;

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

typedef struct _print_parameter
{
  cl_uint  pos;  /* the position where this parameter shall appear, 0=unused */
  char     out[16]; /* fixed size output string */
  char     parm; /* the parameter, one of cCpgtenrswWdTuhf */
} print_parameter;

typedef struct _stats_t
{
  char     progressheader[256];       /* userconfigureable progress header */
  char     progressformat[256];       /* userconfigureable progress line */
  cl_uint  class_number;              /* the number of the last processed class */
  cl_uint  grid_count;                /* number of grids processed in the last processed class */
  cl_ulong class_time;                /* time (in ms) needed to process the last processed class */
  unsigned long long bit_level_time;  /* time (in ms) since the bitlevel was started */
  cl_ulong cpu_wait_time;             /* time (ms) CPU was waiting for the GPU */
  float    cpu_wait;                  /* percentage CPU was waiting for the GPU */
  cl_uint  output_counter;            /* count how often the status line was written since last headline */
  cl_uint  class_counter;             /* number of finished classes of the current job */
  double   ghzdays;                   /* primenet GHZdays for the current assignment (current stage) */
  char     kernelname[32];
}stats_t;

typedef struct _mystuff_t
{
  cl_event copy_events[NUM_STREAMS_MAX];
  cl_event exec_events[NUM_STREAMS_MAX];
  cl_uint *h_ktab[NUM_STREAMS_MAX];
  cl_mem   d_ktab[NUM_STREAMS_MAX];
  cl_uint *h_RES;
  cl_mem   d_RES;
  enum STREAM_STATUS stream_status[NUM_STREAMS_MAX];
  enum GPU_types gpu_type;
  /* for GPU sieving: */
  cl_uint *h_bitarray;
  cl_mem   d_bitarray;
  cl_uint *h_sieve_info;
  cl_mem   d_sieve_info;
  cl_uint *h_calc_bit_to_clear_info;
  cl_mem   d_calc_bit_to_clear_info;

  cl_uint  more_classes;                    /* 0= 420 classes, 1= 4620 classes */
  cl_uint  num_classes;                     /* 420 / 4620 classes */

  cl_uint  exponent;                        /* the exponent we're currently working on */
  cl_uint  bit_min;                         /* where do we start TFing */
  cl_uint  bit_max_assignment;              /* the upper size of factors we're searching for */
  cl_uint  bit_max_stage;                   /* as above, but only for the current stage */

  cl_uint  sieve_primes;                    /* the actual number of odd primes using for sieving */
  cl_uint  sieve_primes_adjust;             /* allow automated adjustment of sieve_primes? */
  cl_uint  sieve_primes_upper_limit;        /* the upper limit of sieve_primes for the current exponent */
  cl_uint  sieve_primes_min, sieve_primes_max; /* user configurable sieve_primes min/max */
  cl_uint  sieve_size;

  cl_uint  gpu_sieving;			             /* TRUE if we're letting the GPU do the sieving */
  cl_uint  gpu_sieve_size;			         /* Size (in bits) of the GPU sieve.  4..128M bits. */
  cl_uint  gpu_sieve_processing_size;	   /* The number of GPU sieve bits each thread in a kernel will process.  8,16,24,32K bits. */
  cl_uint  gpu_sieve_min_exp;			       /* minimum exponent for the sieve_primes we have */

  cl_uint  flush;                        /* GPU sieving only: flush the queue after # kernels, 0=off */
  cl_uint  num_streams;

  enum MODES mode;
  cl_uint checkpoints, checkpointdelay, stages, stopafterfactor;
  cl_uint threads_per_grid_max, threads_per_grid;

#ifdef CHECKS_MODBASECASE
  cl_mem   d_modbasecase_debug;
  cl_uint *h_modbasecase_debug;
#endif

  cl_uint  vectorsize;
  cl_uint  printmode;
//  cl_uint  allowsleep;    // not used in mfakto (yet)
  cl_uint  small_exp;
  cl_uint  print_timestamp;
  cl_uint  quit;
  cl_ulong cpu_mask;         /* CPU affinity mask for the siever thread */
  cl_int   verbosity;        /* -1 = uninitialized, 0 = reduced number of screen printfs, 1= default, >= 2 = some additional printfs */
  cl_int   logging;
  cl_uint  selftestsize;
  cl_uint  force_rebuild;    /* 1: delete the previous binfile */

  stats_t  stats;            /* stats for the status line */

  char workfile[51];         /* allow filenames up to 50 chars... */
  char inifile[51];	         /* allow filenames up to 50 chars... */
  char resultfile[51];
  char jsonresultfile[51];
  char logfile[51];
  FILE *logfileptr;
  char V5UserID[51];         /* primenet V5UserID and ComputerID */
  char ComputerID[51];       /* currently only used for screen/result output */
  char assignment_key[100]; /* the assignment ID */
  char factors_string[500];            /* store factors in global state */
  char CompileOptions[151];  /* additional compile options */
  char binfile[51];          /* compiled kernels file to use, empty if not desired */

  cl_uint override_v;        /* override INI file when setting verbosity */

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
  cl_uint         bit_min, bit_max, stages;
  cl_kernel       kernel;
} kernel_info_t;


#define RET_ERROR 1000000001
#define RET_QUIT  1000000002

#endif
