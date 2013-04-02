/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2013  Oliver Weihe (o.weihe@t-online.de)
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

#include <stdio.h>
#include <stdlib.h>
#ifndef _MSC_VER
  #include <unistd.h>
  #define _GNU_SOURCE
  #include <sched.h>
#endif
#include <string.h>
#include <errno.h>
#include <time.h>
#include <CL/cl.h>

#include "params.h"
#include "my_types.h"
#include "compatibility.h"
#include "sieve.h"
#include "read_config.h"
#include "parse.h"
#include "timer.h"
#include "checkpoint.h"
#include "signal_handler.h"
#include "filelocking.h"
#include "perftest.h"
#include "mfakto.h"
#include "gpusieve.h"
#include "output.h"


int gpu_sieve_main (int argc, char** argv);

mystuff_t mystuff;

extern OpenCL_deviceinfo_t deviceinfo;
extern kernel_info_t       kernel_info[];
struct GPU_type gpu_types[]={
  {GPU_AUTO,     0,  "AUTO"},
  {GPU_VLIW4,   64,  "VLIW4"},
  {GPU_VLIW5,   80,  "VLIW5"},
  {GPU_GCN,     64,  "GCN"},
  {GPU_CPU,      1,  "CPU"},
  {GPU_APU,     80,  "APU"},
  {GPU_NVIDIA,   8,  "NVIDIA"},
  {GPU_INTEL,    1,  "INTEL"},
  {GPU_UNKNOWN,  0,  "UNKNOWN"}
};

unsigned long long int calculate_k(unsigned int exp, int bits)
/* calculates biggest possible k in "2 * exp * k + 1 < 2^bits" */
{
  unsigned long long int k = 0, tmp_low, tmp_hi;
  
  if((bits > 65) && exp < (unsigned int)(1U << (bits - 65))) k = 0; // k would be >= 2^64...
  else if(bits <= 64)
  {
    tmp_low = 1ULL << (bits - 1);
    tmp_low--;
    k = tmp_low / exp;
  }
  else if(bits <= 96)
  {
    tmp_hi = 1ULL << (bits - 33);
    tmp_hi--;
    tmp_low = 0xFFFFFFFFULL;
    
    k = tmp_hi / exp;
    tmp_low += (tmp_hi % exp) << 32;
    k <<= 32;
    k += tmp_low / exp;
  }
  
  if(k == 0)k = 1;
  return k;
}


int kernel_possible(int kernel, cl_uint exp, cl_uint bit_min, cl_uint bit_max)
/* returns 1 if the selected kernel can handle the assignment, 0 otherwise
The variables exp, bit_min and bit_max must be a valid assignment! */
{
  int ret = 0;
  kernel_info_t k = kernel_info[kernel];

  if (bit_min >= k.bit_min  &&  bit_max <= k.bit_max  &&  (k.stages || (bit_max - bit_min) == 1)) ret = 1;
  
  exp++;	//exp is currently unused in this function...
  
  return ret;
}


int class_needed(unsigned int exp, unsigned long long int k_min, int c)
{
/*
checks whether the class c must be processed or can be ignored at all because
all factor candidates within the class c are a multiple of 3, 5, 7 or 11 (11
only if MORE_CLASSES is definied) or are 3 or 5 mod 8

k_min *MUST* be aligned in that way that k_min is in class 0!
*/
  if( ((2 * (exp %  8) * ((k_min + c) %  8)) %  8 !=  2) && \
      ((2 * (exp %  8) * ((k_min + c) %  8)) %  8 !=  4) && \
      ((2 * (exp %  3) * ((k_min + c) %  3)) %  3 !=  2) && \
      ((2 * (exp %  5) * ((k_min + c) %  5)) %  5 !=  4) && \
      ((2 * (exp %  7) * ((k_min + c) %  7)) %  7 !=  6))
#ifdef MORE_CLASSES        
  if(  (2 * (exp % 11) * ((k_min + c) % 11)) % 11 != 10 )
#endif
  {
    return 1;
  }

  return 0;
}

typedef GPUKernels kernel_precedence[UNKNOWN_KERNEL];

GPUKernels find_fastest_kernel(mystuff_t *mystuff)
{
  /* searches the kernel precedence list of the GPU for the first on that is capable of running the assignment */
  static kernel_precedence kernel_precedences [] = {
    /* sorted list of all kernels per GPU type, fastest first */
    {
/*  GPU_AUTO,   */  
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL },
    {
/*  GPU_VLIW4,  */
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL },
    {
/*  GPU_VLIW5,  */
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL },
    {
/*  GPU_GCN,    */
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      /*// 7850@975MHz, v=2 / v=4
          if (st_data[ind].bit_min <= 63)                                    kernels[j++] = _63BIT_MUL24;      // 168.0
    if ((st_data[ind].bit_min >= 61) && (st_data[ind].bit_min < 72))   kernels[j++] = _71BIT_MUL24;      // 178.2 / 176.0
    if ((st_data[ind].bit_min >= 64) && (st_data[ind].bit_min < 76))   kernels[j++] = BARRETT76_MUL32;   // 259.7 / 259.1
    if ((st_data[ind].bit_min >= 64) && (st_data[ind].bit_min < 77))   kernels[j++] = BARRETT77_MUL32;   // 251.1 / 248.0
    if ((st_data[ind].bit_min >= 64) && (st_data[ind].bit_min < 79))   kernels[j++] = BARRETT79_MUL32;   // 221.9 / 221.3
    if ((st_data[ind].bit_min >= 65) && (st_data[ind].bit_min < 87))   kernels[j++] = BARRETT87_MUL32;   // 228.0 / 227.3
    if ((st_data[ind].bit_min >= 65) && (st_data[ind].bit_min < 88))   kernels[j++] = BARRETT88_MUL32;   // 219.9 / 217.6
    if ((st_data[ind].bit_min >= 65) && (st_data[ind].bit_min < 92))   kernels[j++] = BARRETT92_MUL32;   // 198.4 / 197.8
    if ((st_data[ind].bit_min >= 63) && (st_data[ind].bit_min < 70))   kernels[j++] = BARRETT70_MUL24;   // 210.5 / 204.9
    if ((st_data[ind].bit_min >= 60) && (st_data[ind].bit_min < 69))   kernels[j++] = BARRETT69_MUL15;   // 360.0 / 354.1
    if ((st_data[ind].bit_min >= 60) && (st_data[ind].bit_min < 69))   kernels[j++] = BARRETT70_MUL15;   // 360.0 / 354.2
    if ((st_data[ind].bit_min >= 60) && (st_data[ind].bit_min < 70))   kernels[j++] = BARRETT71_MUL15;   // 335.1 / 332.0
    if ((st_data[ind].bit_min >= 60) && (st_data[ind].bit_min < 73))   kernels[j++] = BARRETT73_MUL15;   // 294.6 / 291.4
    if ((st_data[ind].bit_min >= 60) && (st_data[ind].bit_min < 82))   kernels[j++] = BARRETT82_MUL15;   // 261.4 / 202.7
    if ((st_data[ind].bit_min >= 60) && (st_data[ind].bit_min < 83))   kernels[j++] = BARRETT83_MUL15;   // 244.8 / 183.1
    if ((st_data[ind].bit_min >= 60) && (st_data[ind].bit_min < 88))   kernels[j++] = BARRETT88_MUL15;   // 219.5 / 169.6
    if ((st_data[ind].bit_min >= 10) && (st_data[ind].bit_min < 62))   kernels[j++] = MG62;              // 154.0 /  53.5
    if ((st_data[ind].bit_min >= 74) && (st_data[ind].bit_min < 89))   kernels[j++] = MG88;              // 163.8 /  57.4
*/

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL },
    {
/*  GPU_CPU,    */
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL },
    {
/*  GPU_APU,    */
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL },
    {
/*  GPU_NVIDIA, */
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL },
    {
/*  GPU_INTEL,  */
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL },
    {
/*  GPU_UNKNOWN */
      BARRETT69_MUL15,  //                   244M/s  vec=4  on HD5770
      BARRETT70_MUL15,  //                   240M/s  vec=4
      BARRETT71_MUL15,  //                   226M/s  vec=4
      BARRETT77_MUL32,  //                   203M/s  vec=2
      BARRETT73_MUL15,  // 288M/s            196M/s  vec=4
      BARRETT70_MUL24,  // 321M/s on HD5870, 183M/s  vec=4
      BARRETT92_MUL32,
      BARRETT88_MUL32,
      BARRETT87_MUL32,
      _71BIT_MUL24,     // 258M/s            130M/s
      BARRETT79_MUL32,  // 255M/s            128M/s
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,
      _63BIT_MUL24,     // 236M/s
      BARRETT92_MUL32,  // 205M/s            103M/s
      MG62,
      MG88,

      UNKNOWN_KERNEL,
      UNKNOWN_KERNEL }
    
  };

  kernel_precedence *k = &kernel_precedences[mystuff->gpu_type]; // select the row for the GPU we're running on / we're configured for
  GPUKernels         use_kernel = AUTOSELECT_KERNEL;
  cl_uint            i;

  for (i = 0; i < UNKNOWN_KERNEL && (*k)[i] < UNKNOWN_KERNEL; i++)
  {
    if (kernel_possible((*k)[i], mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage))
    {
      use_kernel = (*k)[i];
      break;
    }
  }
  return use_kernel;
}

int tf(mystuff_t *mystuff, int class_hint, unsigned long long int k_hint, enum GPUKernels use_kernel)
/*
tf M<mystuff->exponent> from 2^<mystuff->bit_min> to 2^<mystuff->mystuff->bit_max_stage>

kernel: see my_types.h -> enum GPUKernels

return value (mystuff->mode = MODE_NORMAL):
number of factors found
RET_ERROR any CL function returned an error
RET_QUIT if early exit was requested by SIGINT

 

return value (mystuff->mode > MODE_NORMAL), i.e. selftest:
0 for a successful selftest (known factor was found)
1 no factor found
2 wrong factor returned
RET_ERROR any CL function returned an error

other return value 
-1 unknown mode
*/
{
  unsigned int cur_class, max_class = NUM_CLASSES-1, i, count = 0;
  unsigned long long int k_min, k_max, k_range, tmp;
  unsigned int f_hi, f_med, f_low;
  struct timeval timer;
  time_t time_last_checkpoint, time_add_file_check=0;
  int factorsfound = 0, numfactors = 0, restart = 0, do_checkpoint = mystuff->checkpoints;

  int retval = 0, add_file_exists = 0;
    
  cl_ulong time_run, time_est;

  mystuff->stats.output_counter = 0; /* reset output counter, needed for status headline */
  mystuff->stats.ghzdays = primenet_ghzdays(mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage);

  if(mystuff->mode != MODE_SELFTEST_SHORT)printf("Starting trial factoring M%u from 2^%d to 2^%d (%.2fGHz-days)\n",
    mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage, mystuff->stats.ghzdays);
  timer_init(&timer);
  time(&time_last_checkpoint);

  mystuff->stats.class_counter = 0;
  
  k_min=calculate_k(mystuff->exponent,mystuff->bit_min);
  k_max=calculate_k(mystuff->exponent,mystuff->bit_max_stage);
  
  if(mystuff->mode > MODE_NORMAL) // any selftest mode
  {
/* a shortcut for the selftest, bring k_min and k_max "close" to the known factor */
    if(NUM_CLASSES == 420)k_range = 10000000000ULL;
    else                  k_range = 100000000000ULL;
    if(mystuff->mode == MODE_SELFTEST_SHORT)k_range /= 5; /* even smaller ranges for the "small" selftest */
    if((k_max - k_min) > (3ULL * k_range))
    {
      tmp = k_hint - (k_hint % k_range) - k_range;
      if(tmp > k_min) k_min = tmp;

      tmp += 3ULL * k_range;
      if((tmp < k_max) || (k_max < k_min)) k_max = tmp; /* check for k_max < k_min enables some selftests where k_max >= 2^64 but the known factor itself has a k < 2^64 */
    }
#ifdef DEBUG_FACTOR_FIRST
    // The following line is just for debugging: it makes sure that the factor to be found is the first k being tested (so the first trace should show finding the factor)
    k_min = k_hint - 2*4620;
#endif
  }

  k_min -= k_min % NUM_CLASSES;	/* k_min is now 0 mod NUM_CLASSES */

  if(mystuff->mode != MODE_SELFTEST_SHORT && mystuff->verbosity >= 2)
  {
    printf("  k_min = %llu - k_max = %llu\n", k_min, k_max);
  }

  if(use_kernel == AUTOSELECT_KERNEL)
  {  // this is the speed order for VLIW4, Cayman (HD6970)
    use_kernel = find_fastest_kernel(mystuff);

    if(use_kernel == AUTOSELECT_KERNEL)
    {
      printf("ERROR: No suitable kernel found for bit_min=%d, bit_max=%d.\n",
                 mystuff->bit_min, mystuff->bit_max_stage);
      return RET_ERROR;
    }
  }

  sprintf(mystuff->stats.kernelname, kernel_info[use_kernel].kernelname);

  if(mystuff->mode != MODE_SELFTEST_SHORT && mystuff->verbosity >= 1)printf("Using GPU kernel \"%s\"\n", mystuff->stats.kernelname);

  if(mystuff->mode == MODE_NORMAL)
  {
    if((mystuff->checkpoints > 0) && (checkpoint_read(mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage, &cur_class, &factorsfound) == 1))
    {
      printf("\nFound a valid checkpoint file.\n");
      if(mystuff->verbosity >= 1) printf("  last finished class was: %d\n", cur_class);
      if(mystuff->verbosity >= 2) printf("  found %d factor%s already\n", factorsfound, factorsfound == 1 ? "" : "s");
      printf("\n");
      cur_class++; // the checkpoint contains the last completely processed class!

/* calculate the number of classes which are already processed. This value is needed to estimate ETA */
      for(i = 0; i < cur_class; i++)
      {
        if(class_needed(mystuff->exponent, k_min, i))mystuff->stats.class_counter++;
      }
      restart = mystuff->stats.class_counter;
    }
    else
    {
      cur_class=0;
    }
  }
  else // mystuff->mode != MODE_NORMAL
  {
    cur_class = class_hint % NUM_CLASSES;
    max_class = cur_class;
  }

  if (mystuff->gpu_sieving == 1)
  {
    gpusieve_init_exponent(mystuff);
  }

  for(; cur_class <= max_class; cur_class++)
  {
    if(class_needed(mystuff->exponent, k_min, cur_class))
    {
      mystuff->stats.class_number = cur_class;
      if(mystuff->quit)
      {
/* check if quit is requested. Because this is at the begining of the class
   we can be sure that if RET_QUIT is returned the last class hasn't
   finished. The signal handler which sets mystuff->quit not active during
   selftests so we need to check for RET_QUIT only when doing real work. */
        if(mystuff->printmode == 1)printf("\n");
        return RET_QUIT;
      }
      else
      {
        count++;
        mystuff->stats.class_counter++;

        if (mystuff->gpu_sieving == 1)
        {
          use_kernel = BARRETT77_MUL32_GS; // TODO let it select suitable kernels
          gpusieve_init_class(mystuff, k_min+cur_class);
          if ((use_kernel >= BARRETT79_MUL32_GS) && (use_kernel <= BARRETT82_MUL15_GS))
          {
            numfactors = tf_class_opencl (k_min+cur_class, k_max, mystuff, use_kernel);
          }
          else
          {
            printf("ERROR: Unknown GPU sieve kernel selected (%d)!\n", use_kernel);  return RET_ERROR;
          }
        }
        else
        {
          sieve_init_class(mystuff->exponent, k_min+cur_class, mystuff->sieve_primes);
          if ((use_kernel >= _71BIT_MUL24) && (use_kernel < UNKNOWN_KERNEL))
          {
            numfactors = tf_class_opencl (k_min+cur_class, k_max, mystuff, use_kernel);
          }
          else
          {
            printf("ERROR: Unknown kernel selected (%d)!\n", use_kernel);  return RET_ERROR;
          }
        }

        if (numfactors == RET_ERROR)
        {
          printf("ERROR from tf_class.\n");
          return RET_ERROR;
        }
        factorsfound+=numfactors;

        if(mystuff->mode == MODE_NORMAL)
        {
          time_t now = time(NULL);
          if (add_file_exists)
          {
            if (now > time_add_file_check + 300)   // do not process the add file until it is 5 minutes old
            {
              process_add_file(mystuff->workfile);
              add_file_exists = 0;
            } // else just wait until after the next class
          }
          else
          {
            add_file_exists = add_file_available(mystuff->workfile);
            time_add_file_check = now;
          }

          if (mystuff->checkpoints > 0)
          {
            if ( ((mystuff->checkpoints > 1) && (--do_checkpoint == 0)) ||
                 ((mystuff->checkpoints == 1) && (now - time_last_checkpoint > (time_t) mystuff->checkpointdelay)) ||
                   mystuff->quit )
            {
              checkpoint_write(mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage, cur_class, factorsfound);
              do_checkpoint = mystuff->checkpoints;
              time_last_checkpoint = now;
            }
          }
          if((mystuff->stopafterfactor >= 2) && (factorsfound > 0) && (cur_class != max_class))cur_class = max_class + 1;
        }
      }
      fflush(NULL);
    }
  }
  if(mystuff->mode != MODE_SELFTEST_SHORT && mystuff->printmode == 1)printf("\n");
  print_result_line(mystuff, factorsfound);

  if(mystuff->mode == MODE_NORMAL)
  {
    retval = factorsfound;
    if(mystuff->checkpoints > 0)checkpoint_delete(mystuff->exponent);
  }
  else // mystuff->mode != MODE_NORMAL
  {
    if(mystuff->h_RES[0] == 0)
    {
      printf("ERROR: selftest failed for M%u (%s)\n", mystuff->exponent, kernel_info[use_kernel].kernelname);
      printf("  no factor found\n");
      retval = 1;
    }
    else // mystuff->h_RES[0] > 0
    {
/*
calculate the value of the known factor in f_{hi|med|low} and compare with the
results from the selftest.
k_max and k_min are used as 64bit temporary integers here...
*/    
      f_hi    = (k_hint >> 63);
      f_med   = (k_hint >> 31) & 0xFFFFFFFFULL;
      f_low   = (k_hint <<  1) & 0xFFFFFFFFULL; /* f_{hi|med|low} = 2 * k_hint */
      
      k_max   = (unsigned long long int)mystuff->exponent * f_low;
      f_low   = (k_max & 0xFFFFFFFFULL) + 1;
      k_min   = (k_max >> 32);

      k_max   = (unsigned long long int)mystuff->exponent * f_med;
      k_min  += k_max & 0xFFFFFFFFULL;
      f_med   = k_min & 0xFFFFFFFFULL;
      k_min >>= 32;
      k_min  += (k_max >> 32);

      f_hi  = (unsigned int ) (k_min + (mystuff->exponent * f_hi)); /* f_{hi|med|low} = 2 * k_hint * exp +1 */
      
      if ((use_kernel == _71BIT_MUL24) || (use_kernel == _63BIT_MUL24) || (use_kernel == BARRETT70_MUL24)) /* these kernels use 24bit per int */
      {
        f_hi  <<= 16;
        f_hi   += f_med >> 16;

        f_med <<= 8;
        f_med  += f_low >> 24;
        f_med  &= 0x00FFFFFF;
        
        f_low  &= 0x00FFFFFF;
      }
      else if (((use_kernel >= BARRETT73_MUL15) && (use_kernel <= BARRETT82_MUL15)) || (use_kernel == MG88))
      {
        // 30 bits per reported result int
        f_hi  <<= 4;
        f_hi   += f_med >> 28;

        f_med <<= 2;
        f_med  += f_low >> 30;
        f_med  &= 0x3FFFFFFF;
        
        f_low  &= 0x3FFFFFFF;
      }
      k_min=0; /* using k_min for counting the number of matches here */
      for(i=0; (i<mystuff->h_RES[0]) && (i<10); i++)
      {
        if(mystuff->h_RES[i*3 + 1] == f_hi  && \
           mystuff->h_RES[i*3 + 2] == f_med && \
           mystuff->h_RES[i*3 + 3] == f_low) k_min++;
      }
      if(k_min != 1) /* the factor should appear ONCE */
      {
        printf("ERROR: selftest failed for M%u (%s)\n", mystuff->exponent, kernel_info[use_kernel].kernelname);
        printf("  expected result: %08X %08X %08X\n", f_hi, f_med, f_low);
        for(i=0; (i<mystuff->h_RES[0]) && (i<10); i++)
        {
          printf("  reported result: %08X %08X %08X\n", mystuff->h_RES[i*3 + 1], mystuff->h_RES[i*3 + 2], mystuff->h_RES[i*3 + 3]);
        }
        retval = 2;
      }
      else
      {
        if(mystuff->mode != MODE_SELFTEST_SHORT)printf("selftest for M%u passed (%s)!\n", mystuff->exponent, kernel_info[use_kernel].kernelname);
      }
    }
  }

  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    time_run = timer_diff(&timer)/1000;
    time_est = time_run;
    
    if(restart == 0)printf("tf(): total time spent: ");
    else            printf("tf(): time spent since restart:   ");

/*  restart == 0 ==> time_est = time_run */

    if(time_run > 86400000ULL)printf("%" PRIu64 "d ",   time_run / 86400000ULL);
    if(time_run > 3600000ULL) printf("%2" PRIu64 "h ", (time_run /  3600000ULL) % 24ULL);
    if(time_run > 60000ULL)   printf("%2" PRIu64 "m ", (time_run /    60000ULL) % 60ULL);
                              printf("%2" PRIu64 ".%03" PRIu64 "s", (time_run / 1000ULL) % 60ULL, time_run % 1000ULL);
    if(restart != 0)
    {
      time_est = (time_run * mystuff->stats.class_counter ) / (cl_ulong)(mystuff->stats.class_counter-restart);
      printf("\n      estimated total time spent: ");
      if(time_est > 86400000ULL)printf("%" PRIu64 "d ",   time_est / 86400000ULL);
      if(time_est > 3600000ULL) printf("%2" PRIu64 "h ", (time_est /  3600000ULL) % 24ULL);
      if(time_est > 60000ULL)   printf("%2" PRIu64 "m ", (time_est /    60000ULL) % 60ULL);
                                printf("%2" PRIu64 ".%03" PRIu64 "s", (time_est / 1000ULL) % 60ULL, time_est % 1000ULL);
    }
    if(mystuff->mode == MODE_NORMAL) printf(" (%.2f GHz-days / day)", mystuff->stats.ghzdays * 86400000.0 / (double) time_est);
    printf("\n\n");
  }
  return retval;
}


int selftest(mystuff_t *mystuff, enum MODES type)
/*
type = 1: small selftest (this is executed EACH time mfakto is started)
type = 2: half selftest
type = 3: full selftest

return value
0 selftest passed
1 selftest failed
RET_ERROR we might have a serios problem
*/
{
#include "selftest-data.h"
  
  int i, j, tf_res, st_success=0, st_nofactor=0, st_wrongfactor=0, st_unknown=0;

  unsigned int num_selftests=0, total_selftests=sizeof(st_data) / sizeof(st_data[0]);
  int f_class, selftests_to_run;
  int retval=1, ind;
  enum GPUKernels kernels[UNKNOWN_KERNEL];
  unsigned int index[] = {    46, 1518, 2,   25,   39,   57,   // some factors below 2^71 (test the 71/75 bit kernel depending on compute capability)
                             70,   72,   73,  82,  88,   // some factors below 2^75 (test 75 bit kernel)
                            106,  355,  358,  666,   // some very small factors
                           1547, 1552, 1556, 1557    // some factors below 2^95 (test 95 bit kernel)
                         };                          // mfakto special case (25-bit factor)
  // save the SievePrimes ini value as the selftest may lower it to fit small test-exponents
  unsigned int sieve_primes_save = mystuff->sieve_primes;
  unsigned int kernel_index;

  if (type == MODE_SELFTEST_FULL)
    selftests_to_run = total_selftests;
  else
    selftests_to_run = 1559;

  register_signal_handler(mystuff);

  for(i=0; i<selftests_to_run; ++i)
  {
    if(type == MODE_SELFTEST_SHORT)
    {
      if (i < (sizeof(index)/sizeof(index[0])))
      {
        ind = index[i];
        printf("########## testcase %d/%d (#%d) ##########\r", i+1, (int) (sizeof(index)/sizeof(index[0])), ind);
      }
      else
        break; // short test done
    }
    else // treat type <> 1 as full test
    {
      printf("########## testcase %d/%d ##########\n", i+1, selftests_to_run);
      ind = i;
    }
    f_class = (int)(st_data[ind].k % NUM_CLASSES);
    mystuff->exponent           = st_data[ind].exp;
    mystuff->bit_min            = st_data[ind].bit_min;
    mystuff->bit_max_assignment = st_data[ind].bit_min + 1;
    mystuff->bit_max_stage      = mystuff->bit_max_assignment;

/* create a list which kernels can handle this testcase */
    j = 0;                                                                                           


    if (mystuff->gpu_sieving == 0)
    {
      for (kernel_index = _63BIT_MUL24; kernel_index < UNKNOWN_KERNEL; ++kernel_index)
      {
        if(kernel_possible(kernel_index, mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage)) kernels[j++] = kernel_index;
      }
      // careful to not sieve out small test candidates
      mystuff->sieve_primes_upper_limit = sieve_sieve_primes_max(mystuff->exponent, mystuff->sieve_primes_max);
      if (mystuff->sieve_primes > mystuff->sieve_primes_upper_limit)
        mystuff->sieve_primes = mystuff->sieve_primes_upper_limit;
    }
    else
    {
      for (kernel_index = BARRETT79_MUL32_GS; kernel_index <= BARRETT82_MUL15_GS; ++kernel_index)
      {
        if(kernel_possible(kernel_index, mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage)) kernels[j++] = kernel_index;
      }
    }

    while(j>0)
    {
      num_selftests++;
      tf_res=tf(mystuff, f_class, st_data[ind].k, kernels[--j]);
            if(tf_res == 0)st_success++;
      else if(tf_res == 1)st_nofactor++;
      else if(tf_res == 2)st_wrongfactor++;
      else if(tf_res == RET_ERROR) return RET_ERROR; /* bail out, we might have a serios problem */
      else           st_unknown++;
#ifdef DETAILED_INFO
      printf("Test %d finished, so far suc: %d, no: %d, wr: %d, unk: %d\n", num_selftests, st_success, st_nofactor, st_wrongfactor, st_unknown);
      fflush(NULL);
#endif
      if (mystuff->quit) break;
    }
    if (mystuff->quit) break;
  }

  printf("Selftest statistics                          \n");
  printf("  number of tests           %d\n", num_selftests);
  printf("  successful tests          %d\n", st_success);
  if(st_nofactor > 0)   printf("  no factor found           %d\n", st_nofactor);
  if(st_wrongfactor > 0)printf("  wrong factor reported     %d\n", st_wrongfactor);
  if(st_unknown > 0)    printf("  unknown return value      %d\n", st_unknown);
  printf("\n");

  // restore SievePrimes ini value
  mystuff->sieve_primes = sieve_primes_save;

  if(st_success == num_selftests)
  {
    printf("selftest PASSED!\n\n");
    retval=0;
  }
  else
  {
    printf("selftest FAILED!\n\n");
  }

  return retval;
}

int main(int argc, char **argv)
{
  unsigned int exponent = 1;
  int bit_min = -1, bit_max = -1;
  int parse_ret = -1;
  int devicenumber = 0;

  int i = 1, tmp = 0;
  char *ptr;
  int use_worktodo = 1;
  
  mystuff.mode = MODE_NORMAL;
  mystuff.quit = 0;
  mystuff.verbosity = -1;
  mystuff.bit_min = -1;
  mystuff.bit_max_assignment = -1;
  mystuff.bit_max_stage = -1;
  mystuff.gpu_sieving = 0;
  mystuff.gpu_sieve_size = GPU_SIEVE_SIZE_DEFAULT * 1024 * 1024;		/* Size (in bits) of the GPU sieve.  Default is 128M bits. */
  mystuff.gpu_sieve_primes = GPU_SIEVE_PRIMES_DEFAULT;				/* Default to sieving primes below about 1.05M */
  mystuff.gpu_sieve_processing_size = GPU_SIEVE_PROCESS_SIZE_DEFAULT * 1024;	/* Default to 8K bits processed by each block in a Barrett kernel. */
  strcpy(mystuff.resultfile, "results.txt");
  strcpy(mystuff.inifile, "mfakto.ini");

  while(i<argc)
  {
    if((!strcmp((char*)"-h", argv[i])) || (!strcmp((char*)"--help", argv[i])))
    {
      print_help(argv[0]);
      return ERR_OK;
    }
    else if(!strcmp((char*)"-v", argv[i]))
    {
      if(i+1 >= argc)
      {
        printf("ERROR: no verbosity level specified for option \"-v\"\n");
        return ERR_PARAM;
      }
      tmp = (int)strtol(argv[i+1], &ptr, 10);
      if(*ptr || errno || tmp != strtol(argv[i+1], &ptr, 10) )
      {
        printf("ERROR: can't parse verbosity level for option \"-v\"\n");
        return ERR_PARAM;
      }
      i++;
      
      if(tmp > 2)
      {
        printf("WARNING: maximum verbosity level is 2\n");
        tmp = 2;
      }
      
      if(tmp < 0)
      {
        printf("WARNING: minumum verbosity level is 0\n");
        tmp = 0;
      }

      mystuff.verbosity = tmp;
    }
    else if(!strcmp((char*)"-d", argv[i]))
    {
      if(i+1 >= argc)
      {
        printf("ERROR: no device number specified for option \"-d\"\n");
        return ERR_PARAM;
      }
      if (argv[i+1][0] == 'c')  // run on CPU
      {
        devicenumber = -1;
      }
      else if (argv[i+1][0] == 'g')  // run on GPU
      {
        devicenumber = 10000;
      }
      else
      {
        devicenumber = strtol(argv[i+1],&ptr,10);
        if(*ptr || errno || devicenumber != strtol(argv[i+1],&ptr,10) )
        {
          printf("ERROR: can't parse <device number> for option \"-d\"\n");
          return ERR_PARAM;
	      }
      }
      i++;
    }
    else if(!strcmp((char*)"-tf", argv[i]))
    {
      if(i+3 >= argc)
      {
        printf("ERROR: missing parameters for option \"-tf\"\n");
        return ERR_PARAM;
      }
      exponent=(unsigned int)strtoul(argv[i+1],&ptr,10);
      if(*ptr || errno || (unsigned long)exponent != strtoul(argv[i+1],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <exp> for option \"-tf\"\n");
        return ERR_PARAM;
      }
      bit_min=(int)strtol(argv[i+2],&ptr,10);
      if(*ptr || errno || (long)bit_min != strtol(argv[i+2],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <min> for option \"-tf\"\n");
        return ERR_PARAM;
      }
      bit_max=(int)strtol(argv[i+3],&ptr,10);
      if(*ptr || errno || (long)bit_max != strtol(argv[i+3],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <max> for option \"-tf\"\n");
        return ERR_PARAM;
      }
      if(!valid_assignment(exponent, bit_min, bit_max, mystuff.verbosity))
      {
        return ERR_PARAM;
      }
      use_worktodo = 0;
      parse_ret = 0;
      i += 3;
    }
    else if(!strcmp((char*)"-st", argv[i]))
    {
      mystuff.mode = MODE_SELFTEST_HALF;
    }
    else if(!strcmp((char*)"-st2", argv[i]))
    {
      mystuff.mode = MODE_SELFTEST_FULL;
    }
    else if(!strcmp((char*)"-i", argv[i]) || !strcmp((char*)"--inifile", argv[i]))
    {
      i++;
      strncpy(mystuff.inifile, argv[i], 50);
      mystuff.inifile[50]='\0';
    }
    else if(!strcmp((char*)"--perftest", argv[i]))
    {
      read_config(&mystuff);
      if ((i+1)<argc)
        tmp = (int)strtol(argv[i+1],&ptr,10);
      else
        tmp = 0;
      perftest(tmp, devicenumber);  
      return ERR_OK;
    }
    else if(!strcmp((char*)"--timertest", argv[i]))
    {
      timertest();
      return ERR_OK;
    }
    else if(!strcmp((char*)"--sleeptest", argv[i]))
    {
      sleeptest();
      return ERR_OK;
    }
    else if(!strcmp((char*)"--CLtest", argv[i]))
    {
      read_config(&mystuff);
      CL_test(devicenumber);
      return ERR_OK;
    }
    else if(!strcmp((char*)"--gpusievetest", argv[i]))
    {
      read_config(&mystuff);
      init_CL(mystuff.num_streams, devicenumber);
//      gpu_sieve_main(argc-i, &argv[i]);  
      return ERR_OK;
    }
    else
    {
      fprintf(stderr, "ERROR: unknown option '%s'\n", argv[i]);
      return ERR_PARAM;
    }
    i++;
  }

  printf("%s (%dbit build)\n\n", MFAKTO_VERSION, (int)(sizeof(void*)*8));

  read_config(&mystuff);
  
/* print current configuration */
  if(mystuff.verbosity >= 1)
  {
    printf("Compiletime options\n");
    if (mystuff.gpu_sieving == 0)
    {
#ifdef SIEVE_SIZE_LIMIT
      printf("  SIEVE_SIZE_LIMIT          %dkiB\n", SIEVE_SIZE_LIMIT);
      printf("  SIEVE_SIZE                %dbits\n", SIEVE_SIZE);
#endif
      printf("  SIEVE_SPLIT               %d\n", SIEVE_SPLIT);
    }
  }
  if(mystuff.gpu_sieving == 0 && SIEVE_SPLIT > mystuff.sieve_primes_min)
  {
    printf("ERROR: SIEVE_SPLIT must be <= SievePrimesMin\n");
    return ERR_PARAM;
  }
  if(mystuff.verbosity >= 1)
  {
#ifdef MORE_CLASSES
    printf("  MORE_CLASSES              enabled\n");
#else
    printf("  MORE_CLASSES              disabled\n");
#endif

#ifdef USE_DEVICE_PRINTF
    printf("  USE_DEVICE_PRINTF         enabled (DEBUG option)\n");
#endif
#ifdef CHECKS_MODBASECASE
    printf("  CHECKS_MODBASECASE        enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_STREAM_SCHEDULE
    printf("  DEBUG_STREAM_SCHEDULE     enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_STREAM_SCHEDULE_CHECK
    printf("  DEBUG_STREAM_SCHEDULE_CHECK\n                            enabled (DEBUG option)\n");
#endif
#ifdef RAW_GPU_BENCH
    printf("  RAW_GPU_BENCH             enabled (DEBUG option)\n");
#endif
#ifdef DETAILED_INFO
    printf("  DETAILED_INFO             enabled (DEBUG option)\n");
#endif
#ifdef CL_PERFORMANCE_INFO
    printf("  CL_PERFORMANCE_INFO       enabled (DEBUG option)\n");
#endif
  }

  if(init_CL(mystuff.num_streams, devicenumber)!=CL_SUCCESS)
  {
    printf("init_CL(%d, %d) failed\n", mystuff.num_streams, devicenumber);
    return ERR_INIT;
  }

  if (mystuff.gpu_type == GPU_AUTO)
  {
    // try to auto-detect the type of GPU
    if (strstr(deviceinfo.d_name, "Capeverde")  ||    // 7750, 7770, 8760, 8740
        strstr(deviceinfo.d_name, "Pitcairn")   ||    // 7850, 7870, 8870
        strstr(deviceinfo.d_name, "Newzealand") ||    // 7990
        strstr(deviceinfo.d_name, "Oland")      ||    // 8670, 8570
        strstr(deviceinfo.d_name, "Malta")      ||    // ?
        strstr(deviceinfo.d_name, "Tahiti"))          // 7950, 7970, 8970, 8950
    {
      mystuff.gpu_type = GPU_GCN;
    }
    else if (strstr(deviceinfo.d_name, "Cayman")   ||  // 6950, 6970
             strstr(deviceinfo.d_name, "Antilles"))    // 6990
    {
      mystuff.gpu_type = GPU_VLIW4;
    }
    else if (strstr(deviceinfo.d_name, "WinterPark")  ||  // 6370D (E2-3200), 6410D (A4-3300, A4-3400)
             strstr(deviceinfo.d_name, "BeaverCreek") ||  // 6530D (A6-3500, A6-3600, A6-3650, A63670K), 6550D (A8-3800, A8-3850, A8-3870K)
             strstr(deviceinfo.d_name, "Zacate")      ||  // 6320 (E-450)
             strstr(deviceinfo.d_name, "Ontario")     ||  // 6290 (C-60)
             strstr(deviceinfo.d_name, "Wrestler"))       // 6250 (C-30, C-50), 6310 (E-240, E-300, E-350)
    {
      mystuff.gpu_type = GPU_APU;
    }
    else if (strstr(deviceinfo.d_name, "Caicos")   ||  // 6450, 7450, 7470
             strstr(deviceinfo.d_name, "Cedar")    ||  // 7350, 5450
             strstr(deviceinfo.d_name, "Redwood")  ||  // 5550, 5570, 5670
             strstr(deviceinfo.d_name, "Turks")    ||  // 6570, 6670, 7570, 7670
             strstr(deviceinfo.d_name, "Juniper")  ||  // 6750, 6770, 5750, 5770
             strstr(deviceinfo.d_name, "Cypress")  ||  // 5830, 5850, 5870
             strstr(deviceinfo.d_name, "Hemlock")  ||  // 5970
             strstr(deviceinfo.d_name, "Barts"))       // 6790, 6850, 6870
    {
      mystuff.gpu_type = GPU_VLIW5;
    }
    else if (strstr(deviceinfo.d_name, "RV7"))         // 4xxx (ATI RV 7xx)
    {
      mystuff.gpu_type = GPU_VLIW5;
      gpu_types[mystuff.gpu_type].CE_per_multiprocessor = 40; // though VLIW5, only 40 instead of 80 compute elements
    }
    else if (strstr(deviceinfo.d_name, "CPU")           ||
             strstr(deviceinfo.v_name, "GenuineIntel")  ||
             strstr(deviceinfo.v_name, "AuthenticAMD"))
    {
      mystuff.gpu_type = GPU_CPU;
    }
    else if (strstr(deviceinfo.v_name, "NVIDIA"))
    {
      mystuff.gpu_type = GPU_NVIDIA;  // not (yet) working
    }
    else if (strstr(deviceinfo.d_name, "Intel(R) HD Graphics"))
    {
      mystuff.gpu_type = GPU_INTEL;  // not (yet) working
    }
    else
    {
      printf("WARNING: Unknown GPU name, assuming GCN type. Please post the device "
          "name \"%s (%s)\" to http://www.mersenneforum.org/showthread.php?t=15646 "
          "to have it added to mfakto. Set GPUType in %s to select a GPU type yourself "
          "and avoid this warning.\n", deviceinfo.d_name, deviceinfo.v_name, mystuff.inifile);
      mystuff.gpu_type = GPU_GCN;
    }
  }

  mystuff.threads_per_grid = mystuff.threads_per_grid_max;
  if(mystuff.threads_per_grid > deviceinfo.maxThreadsPerGrid)
  {
    mystuff.threads_per_grid = (cl_uint)deviceinfo.maxThreadsPerGrid;
  }
  // threads_per_grid is the number of FC's per kernel invocation. It must be divisible by the vectorsize
  // as only threads_per_grid / vectorsize threads will actually be started.
  mystuff.threads_per_grid -= mystuff.threads_per_grid % (mystuff.vectorsize * deviceinfo.maxThreadsPerBlock);

  if(mystuff.verbosity >= 1)
  {
    printf("\nOpenCL device info\n");
    printf("  name                      %s (%s)\n", deviceinfo.d_name, deviceinfo.v_name);
    printf("  device (driver) version   %s (%s)\n", deviceinfo.d_ver, deviceinfo.dr_version);
    printf("  maximum threads per block %d\n", (int)deviceinfo.maxThreadsPerBlock);
    printf("  maximum threads per grid  %d\n", (int)deviceinfo.maxThreadsPerGrid);
    printf("  number of multiprocessors %d (%d compute elements)\n", deviceinfo.units, deviceinfo.units * gpu_types[mystuff.gpu_type].CE_per_multiprocessor);
    printf("  clock rate                %dMHz\n", deviceinfo.max_clock);

    printf("\nAutomatic parameters\n");

    printf("  threads per grid          %d\n", mystuff.threads_per_grid);
    printf("  optimizing kernels for    %s\n\n", gpu_types[mystuff.gpu_type].gpu_name);
  }

  if ((mystuff.gpu_type == GPU_GCN) && (mystuff.vectorsize > 3))
  {
    printf("\nWARNING: Your GPU was detected as GCN (Graphics Core Next). "
      "These chips perform very slow with vector sizes of 4 or higher. "
      "Please change to VectorSize=2 in %s and restart mfakto for optimal performance.\n\n",
      mystuff.inifile);
  }

  if (init_CLstreams())
  {
    printf("ERROR: init_CLstreams (malloc buffers?) failed\n");
    return ERR_MEM;
  }
  if (mystuff.gpu_sieving == 0)
  {
    // do not set the CPU affinity earlier as the OpenCL initialization will
    // start some control threads which we do not want to bind to a certain CPU
    // no need to do this if we're sieving on the GPU
    if (mystuff.cpu_mask)
    {
#ifdef _MSC_VER
      SetThreadAffinityMask(GetCurrentThread(), mystuff.cpu_mask);
#else
      sched_setaffinity(0, sizeof(mystuff.cpu_mask), mystuff.cpu_mask);
#endif
    }
#ifdef SIEVE_SIZE_LIMIT
    sieve_init();
#else
    sieve_init(mystuff.sieve_size, mystuff.sieve_primes_max);
#endif
    mystuff.sieve_primes_upper_limit = mystuff.sieve_primes_max;
  }

  if(mystuff.mode == MODE_NORMAL)
  {

/* before we start real work run a small selftest */  
    mystuff.mode = MODE_SELFTEST_SHORT;
    if(mystuff.verbosity >= 1) printf("running a simple selftest ...\n");
    //if (selftest(&mystuff, MODE_SELFTEST_SHORT) != 0) return ERR_SELFTEST; /* selftest failed :( */ // TODO re-enable selftest
    mystuff.mode = MODE_NORMAL;
    /* allow for ^C */
    register_signal_handler(&mystuff);
 
    do
    {
      if (use_worktodo) parse_ret = get_next_assignment(mystuff.workfile, &((mystuff.exponent)), &((mystuff.bit_min)), 
                                                            &((mystuff.bit_max_assignment)), NULL, mystuff.verbosity);
      else
      {
        mystuff.exponent           = exponent;
        mystuff.bit_min            = bit_min;
        mystuff.bit_max_assignment = bit_max;
      }

      if (parse_ret == OK)
      {
        if(mystuff.verbosity >= 1)printf("got assignment: exp=%u bit_min=%d bit_max=%d (%.2f GHz-days)\n", mystuff.exponent, mystuff.bit_min, mystuff.bit_max_assignment, primenet_ghzdays(mystuff.exponent, mystuff.bit_min, mystuff.bit_max_assignment));

        mystuff.bit_max_stage = mystuff.bit_max_assignment;

        if (mystuff.gpu_sieving == 0)
        {
          mystuff.sieve_primes_upper_limit = sieve_sieve_primes_max(mystuff.exponent, mystuff.sieve_primes_max);
          if(mystuff.sieve_primes > mystuff.sieve_primes_upper_limit)
          {
            mystuff.sieve_primes = mystuff.sieve_primes_upper_limit;
            printf("WARNING: SievePrimes is too big for the current assignment, lowering to %u\n", mystuff.sieve_primes_upper_limit);
            printf("         It is not allowed to sieve primes which are equal or bigger than the \n");
            printf("         exponent itself!\n");
          }
        }
        if(mystuff.stages == 1)
        {
          while( ((calculate_k(mystuff.exponent, mystuff.bit_max_stage) - calculate_k(mystuff.exponent, mystuff.bit_min)) > (250000000ULL * NUM_CLASSES)) && ((mystuff.bit_max_stage - mystuff.bit_min) > 1) )mystuff.bit_max_stage--;
        }
        tmp = 0;
        while(mystuff.bit_max_stage <= mystuff.bit_max_assignment && !mystuff.quit)
        {
          tmp = tf(&mystuff, 0, 0, AUTOSELECT_KERNEL);
          if(tmp == RET_ERROR) return ERR_RUNTIME; /* bail out, we might have a serios problem  */

          if(tmp != RET_QUIT)
          {

            if( (mystuff.stopafterfactor > 0) && (tmp > 0) )
            {
              mystuff.bit_max_stage = mystuff.bit_max_assignment;
            }

            if(use_worktodo)
            {
              if(mystuff.bit_max_stage == mystuff.bit_max_assignment)parse_ret = clear_assignment(mystuff.workfile, mystuff.exponent, mystuff.bit_min, mystuff.bit_max_assignment, 0);
              else                                                   parse_ret = clear_assignment(mystuff.workfile, mystuff.exponent, mystuff.bit_min, mystuff.bit_max_assignment, mystuff.bit_max_stage);

                   if(parse_ret == CANT_OPEN_WORKFILE)   printf("ERROR: clear_assignment() / modify_assignment(): can't open \"%s\"\n", mystuff.workfile);
              else if(parse_ret == CANT_OPEN_TEMPFILE)   printf("ERROR: clear_assignment() / modify_assignment(): can't open \"__worktodo__.tmp\"\n");
              else if(parse_ret == ASSIGNMENT_NOT_FOUND) printf("ERROR: clear_assignment() / modify_assignment(): assignment not found in \"%s\"\n", mystuff.workfile);
              else if(parse_ret == CANT_RENAME)          printf("ERROR: clear_assignment() / modify_assignment(): can't rename workfiles\n");
              else if(parse_ret != OK)                   printf("ERROR: clear_assignment() / modify_assignment(): Unknown error (%d)\n", parse_ret);
            }

            mystuff.bit_min = mystuff.bit_max_stage;
            mystuff.bit_max_stage++;
          }
        }
      }
      else if(parse_ret == CANT_OPEN_FILE)             printf("ERROR: get_next_assignment(): can't open \"%s\"\n", mystuff.workfile);
      else if(parse_ret == VALID_ASSIGNMENT_NOT_FOUND) printf("ERROR: get_next_assignment(): no valid assignment found in \"%s\"\n", mystuff.workfile);
      else if(parse_ret != OK)                         printf("ERROR: get_next_assignment(): Unknown error (%d)\n", parse_ret);
    }
    while(parse_ret == OK && use_worktodo && !mystuff.quit);
  }
  else // mystuff.mode != MODE_NORMAL
  {
    if (0 != selftest(&mystuff, mystuff.mode))
    {
      printf ("Error exit as selftest failed\n");
      cleanup_CL();
      sieve_free();
      return ERR_SELFTEST;
    }
  }

  cleanup_CL();

  sieve_free();

  return ERR_OK;
}
