/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)
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

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct
{
  unsigned int d0,d1,d2;
}int72;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct
{
  unsigned int d0,d1,d2,d3,d4,d5;
}int144;


/* int72 and int96 are the same but this way the compiler warns
when an int96 is passed to a function designed to handle 72 bit int.
The applies to int144 and int192, too. */

/* 96bit (3x 32bit) integer
D= d0 + d1*(2^32) + d2*(2^64) */
typedef struct
{
  unsigned int d0,d1,d2;
}int96;

/* 192bit (6x 32bit) integer
D=d0 + d1*(2^32) + d2*(2^64) + ... */
typedef struct
{
  unsigned int d0,d1,d2,d3,d4,d5;
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
  MODE_SELFTEST_FULL
};

typedef struct
{
  cl_event copy_events[NUM_STREAMS_MAX];
  cl_event exec_events[NUM_STREAMS_MAX];
  unsigned int *h_ktab[NUM_STREAMS_MAX];
  cl_mem        d_ktab[NUM_STREAMS_MAX];
  unsigned int *h_RES;
  cl_mem        d_RES;
  enum STREAM_STATUS stream_status[NUM_STREAMS_MAX];
  
  int sieve_primes, sieve_primes_adjust, sieve_primes_max;
  char workfile[51];		/* allow filenames up to 50 chars... */
  int num_streams;
  
  enum MODES mode;
  int checkpoints, stages, stopafterfactor;
  int threads_per_grid_max, threads_per_grid;

#ifdef CHECKS_MODBASECASE
  unsigned int *d_modbasecase_debug;
  unsigned int *h_modbasecase_debug;
#endif  

  int vectorsize;
  int printmode;
  int class_counter;		/* needed for ETA calculation */
  int allowsleep;
  int quit; 
}mystuff_t;			/* FIXME: proper name needed */

typedef struct
{
    char d_name[128], d_ver[128], v_name[128], dr_version[128], exts[2048];
    cl_ulong gl_cache, gl_mem, l_mem;
    cl_uint max_clock, units, w_dim;
    size_t wg_size, wi_sizes[10], maxThreadsPerBlock, maxThreadsPerGrid;
} OpenCL_deviceinfo_t;

enum GPUKernels
{
  AUTOSELECT_KERNEL = 0,
  _TEST_MOD_,
  _64BIT_64_OpenCL,
  _95BIT_64_OpenCL,
  BARRETT92_64_OpenCL,
  _71BIT_MUL24,
  _71BIT_MUL24_4,
  _71BIT_MUL24_8,
  BARRETT79_MUL32,
  BARRETT92_MUL32,
  UNKNOWN_KERNEL, /* what comes after this one will not be loaded */
  _95BIT_MUL32  /* not yet there */
};

typedef struct _kernel_info
{
  enum GPUKernels kernel_id;
  char            kernelname[32];
  int             bit_min, bit_max;
  cl_kernel       kernel;
} kernel_info_t;


#define RET_ERROR 1000000001
#define RET_QUIT  1000000002

